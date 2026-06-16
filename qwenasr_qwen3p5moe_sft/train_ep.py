#!/usr/bin/env python3
"""EP Training Script - Custom training loop with Expert Parallelism.

Supports LMDB-based data loading with dynamic batching by token count.
"""
import os
import json
import logging
import time
import math
import re
import struct
import numpy as np
import collections
import lmdb
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from model_ep import EPSpeechTranslationModel, save_ep_checkpoint, load_ep_checkpoint, load_lora_checkpoint, apply_lora_to_model
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor,
)
from torch.utils.data import Dataset
from protofiles import ASRProto
from file_io import LengthsFileReader

logger = logging.getLogger(__name__)
LMDB_TEXT_TOKEN_REPLACEMENTS = {
    "<unused0>": "<|box_start|>",
}


def _compress_audio_pad_runs(text):
    return re.sub(r'(<\|audio_pad\|>){2,}', lambda m: f"<|audio_pad|>*{m.group(0).count('<|audio_pad|>')}", text)


def _parse_loss_bucket_edges(bucket_spec):
    if not bucket_spec:
        return []
    edges = []
    for part in bucket_spec.split(","):
        part = part.strip()
        if not part:
            continue
        edges.append(int(part))
    return sorted(set(edges))


def _format_loss_bucket_label(edges, bucket_idx):
    if not edges:
        return "all"
    if bucket_idx == 0:
        return f"<= {edges[0]}"
    if bucket_idx == len(edges):
        return f"> {edges[-1]}"
    return f"{edges[bucket_idx - 1] + 1}-{edges[bucket_idx]}"


def _infer_checkpoint_world_size(checkpoint_dir):
    """Infer the world size used to create a checkpoint.

    Prefer metadata written by newer checkpoints. Fall back to counting
    per-rank train_state files so older checkpoints can still be handled.
    """
    meta_path = os.path.join(checkpoint_dir, "ep_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        world_size = metadata.get("world_size")
        if world_size is not None:
            return int(world_size)

    state_file_pattern = re.compile(r"^train_state_rank(\d+)\.pt$")
    state_ranks = []
    for name in os.listdir(checkpoint_dir):
        match = state_file_pattern.match(name)
        if match:
            state_ranks.append(int(match.group(1)))

    if not state_ranks:
        return None

    expected_ranks = set(range(max(state_ranks) + 1))
    if set(state_ranks) == expected_ranks:
        return max(state_ranks) + 1
    return None


def _get_feat_extract_output_lengths(input_lengths):
    """Calculate audio token count from mel frames"""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


# ---------------------------------------------------------------------------
# LMDB helpers
# ---------------------------------------------------------------------------
class LmdbEnvWrapper:
    """Lazy-opening LMDB environment wrapper with LRU cache.

    For large-scale training (hundreds of LMDBs), keeps at most `max_open` environments
    open simultaneously to avoid file descriptor exhaustion. Uses LRU eviction.
    Supports fork-safe re-opening for DataLoader workers.
    """
    _env_cache = collections.OrderedDict()  # (lmdb_path, pid) -> lmdb.Environment
    _env_count_by_pid = collections.defaultdict(int)  # pid -> count of open envs
    _max_open = 2048  # max simultaneously open LMDB environments per process

    @classmethod
    def set_max_open(cls, n):
        cls._max_open = n

    def __init__(self, lmdb_path, increment_key=True):
        self.lmdb_path = lmdb_path
        self.increment_key = increment_key

    def _open(self):
        pid = os.getpid()
        cache_key = (self.lmdb_path, pid)

        # Already open: move to end (most recently used)
        if cache_key in LmdbEnvWrapper._env_cache:
            LmdbEnvWrapper._env_cache.move_to_end(cache_key)
            return

        # Evict LRU entries if at capacity
        while LmdbEnvWrapper._env_count_by_pid[pid] >= LmdbEnvWrapper._max_open:
            # Find oldest entry for this pid
            for k in LmdbEnvWrapper._env_cache:
                if k[1] == pid:
                    evicted_env = LmdbEnvWrapper._env_cache.pop(k)
                    evicted_env.close()
                    LmdbEnvWrapper._env_count_by_pid[pid] -= 1
                    break
            else:
                break  # no entries for this pid

        # Use actual file size as map_size to minimize virtual address space
        file_size = os.path.getsize(self.lmdb_path) if os.path.exists(self.lmdb_path) else 1 << 30
        map_size = max(file_size * 2, 1 << 20)  # at least 1MB, 2x file size for headroom

        LmdbEnvWrapper._env_cache[cache_key] = lmdb.open(
            self.lmdb_path, map_size=map_size, max_dbs=2,
            subdir=False, readonly=True, lock=False, readahead=False,
        )
        LmdbEnvWrapper._env_count_by_pid[pid] += 1

    def _get_env(self):
        pid = os.getpid()
        cache_key = (self.lmdb_path, pid)
        self._open()
        # Move to end on access (LRU)
        LmdbEnvWrapper._env_cache.move_to_end(cache_key)
        return LmdbEnvWrapper._env_cache[cache_key]

    def _format_key(self, index):
        if self.increment_key:
            return f"{index:011d}".encode()
        return str(index).encode()

    def read(self, index):
        """Read a single entry by index. Returns raw bytes value."""
        env = self._get_env()
        key = self._format_key(index)
        with env.begin(write=False) as txn:
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Key {key} (index={index}) not found in {self.lmdb_path}")
        return value

    def close(self):
        pid = os.getpid()
        cache_key = (self.lmdb_path, pid)
        if cache_key in LmdbEnvWrapper._env_cache:
            LmdbEnvWrapper._env_cache.pop(cache_key).close()


def parse_proto_int32(raw_bytes):
    """Parse ASRProto -> np.int32 array."""
    proto = ASRProto.FromString(raw_bytes)
    return np.frombuffer(proto.data, dtype=np.int32).copy()


def parse_proto_wav(raw_bytes):
    """Parse ASRProto -> float32 audio samples (16kHz PCM16)."""
    proto = ASRProto.FromString(raw_bytes)
    pcm = np.frombuffer(proto.data, dtype=np.int16).copy()
    return pcm.astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Single LMDB dataset entry (one dataset block in train.json)
# ---------------------------------------------------------------------------
class LmdbDatasetEntry:
    """Represents one dataset entry from train.json. Opens all LMDB envs lazily."""

    # Known fields with their parsing modes
    FIELD_PARSERS = {
        'wav': 'wav',           # PCM audio
        'ed_label': 'int32',    # translation token ids
        'ce_label': 'int32',    # phone sequence
        'ed_label_kws': 'int32',  # keywords token ids
        'ed_label_src': 'int32',  # source text token ids
        'ed_label_asr': 'int32',  # ASR text token ids
        'ed_label_acc': 'int32',  # ASR accuracy value
    }

    def __init__(self, dataset_config: dict):
        self.config = dataset_config
        self.data_num = dataset_config.get('data_num', 0)
        self.lmdb_envs = {}     # field_name -> LmdbEnvWrapper
        self.field_metas = {}   # field_name -> json metadata

        # Range support: [start, end) selects a subset of the LMDB
        self.range_start = 0
        self.range_end = None  # None means use all data
        if 'range' in dataset_config:
            r = dataset_config['range']
            self.range_start = r[0]
            self.range_end = r[1]

        # Open each LMDB field defined in config
        for field_name, json_path in dataset_config.items():
            if field_name in ('data_num', 'range'):
                continue
            if not isinstance(json_path, str) or not json_path.endswith('.json'):
                continue
            if not os.path.exists(json_path):
                logger.warning(f"LMDB json not found: {json_path}, skipping field '{field_name}'")
                continue

            with open(json_path, 'r') as f:
                meta = json.load(f)

            self.field_metas[field_name] = meta
            lmdb_path = meta['lmdb_path']
            increment_key = meta.get('increment_key', 'true').lower() == 'true'
            self.lmdb_envs[field_name] = LmdbEnvWrapper(lmdb_path, increment_key)

        # Load lengths from wav meta
        self.lengths = None
        if 'wav' in self.field_metas:
            wav_meta = self.field_metas['wav']
            lengths_file = wav_meta.get('metas', {}).get('lengths_file')
            if lengths_file and os.path.exists(lengths_file):
                reader = LengthsFileReader(lengths_file).open()
                all_lengths = [reader.read(i) for i in range(reader.n)]
                reader.close()
                # Apply range to lengths
                end = self.range_end if self.range_end is not None else len(all_lengths)
                self.lengths = all_lengths[self.range_start:end]

        # Determine languages from metadata
        self.wav_language = self.field_metas.get('wav', {}).get('language', 'chinese')
        self.target_language = self.field_metas.get('ed_label', {}).get('language', 'english')

        # Update data_num based on range
        if self.range_end is not None:
            self.data_num = self.range_end - self.range_start
        elif self.lengths:
            self.data_num = len(self.lengths)
        # else: keep original data_num

    def has_field(self, field_name):
        return field_name in self.lmdb_envs

    def read_raw(self, field_name, index):
        """Read raw proto bytes from a specific LMDB field.

        index is local (0-based within this entry's range).
        Automatically offset by range_start to get the LMDB global key.
        """
        if field_name not in self.lmdb_envs:
            return None
        global_index = index + self.range_start
        return self.lmdb_envs[field_name].read(global_index)

    def read_parsed(self, field_name, index):
        """Read and parse a field. Returns appropriate type based on field."""
        raw = self.read_raw(field_name, index)
        if raw is None:
            return None

        # Determine parser: use known parsers or default to int32
        parser_type = self.FIELD_PARSERS.get(field_name, 'int32')
        if parser_type == 'wav':
            return parse_proto_wav(raw)
        else:
            return parse_proto_int32(raw)

    def get_length(self, index):
        """Get pre-computed length for a sample."""
        if self.lengths is not None and index < len(self.lengths):
            return self.lengths[index]
        return 0

    def close(self):
        for env in self.lmdb_envs.values():
            env.close()


# ---------------------------------------------------------------------------
# KWS parsing
# ---------------------------------------------------------------------------
def parse_kws_text(kws_text: str) -> list:
    """Parse keyword pairs from decoded kws text.

    Input format examples:
        "00003488923 将供诉人翻译为plaintiff:将刘立新翻译为her:"
        "" (empty)

    Returns list of (src, tgt) tuples, e.g. [("供诉人", "plaintiff"), ("刘立新", "her")]
    """
    if not kws_text or not kws_text.strip():
        return []

    # Split by ':'
    parts = kws_text.strip().split(':')
    pairs = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Match pattern "将X翻译为Y" (with optional leading ID/spaces)
        m = re.search(r'将(.+?)翻译为(.+)', part)
        if m:
            src = m.group(1).strip()
            tgt = m.group(2).strip()
            if src and tgt:
                pairs.append((src, tgt))
    return pairs


# ---------------------------------------------------------------------------
# LMDB Dataset
# ---------------------------------------------------------------------------
class LmdbSpeechTranslationDataset(Dataset):
    """LMDB-based speech translation dataset.

    Reads audio, translations, ASR text, keywords etc. from multiple LMDB databases.
    Supports flexible addition of new LMDB fields without code changes.

    NOTE: LMDB token data was encoded with a different tokenizer (GemmaTokenizer).
    We need a separate lmdb_tokenizer to decode the stored int32 token IDs back to text,
    then use the training tokenizer (Qwen3.5) for chat template and final tokenization.
    """

    def __init__(self, data_json_path, tokenizer, feature_extractor,
                 max_tokens_persample=None, asr_acc_threshold=1.0,
                 lmdb_tokenizer_path=None):
        self.tokenizer = tokenizer  # Qwen3.5 training tokenizer
        self.feature_extractor = feature_extractor

        # LMDB decode tokenizer (GemmaTokenizer used when building the LMDB)
        if lmdb_tokenizer_path:
            self.lmdb_tokenizer = AutoTokenizer.from_pretrained(lmdb_tokenizer_path)
        else:
            self.lmdb_tokenizer = None
        self.max_tokens_persample = max_tokens_persample
        self.asr_acc_threshold = asr_acc_threshold

        # Audio special token IDs
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_end|>")
        self.audio_pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")

        # Load dataset entries from train.json
        with open(data_json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.entries = []                      # list of LmdbDatasetEntry
        self.cumulative_sizes = []             # cumulative raw sample counts
        self.filtered_cumulative_sizes = []    # cumulative kept sample counts
        self.valid_indices = []                # filtered global sample indices
        self.all_lengths = []                  # filtered flat lengths for batching
        self.raw_total_samples = 0
        self.filtered_samples = 0

        total = 0
        kept_total = 0
        for ds_config in config['datasets']:
            entry = LmdbDatasetEntry(ds_config)
            self.entries.append(entry)
            entry_start = total
            total += entry.data_num
            self.cumulative_sizes.append(total)
            kept_in_entry = 0

            if entry.lengths is None:
                if self.max_tokens_persample is not None:
                    raise ValueError(
                        f"Dataset entry starting at raw index {entry_start} is missing wav lengths_file, "
                        "required for --max_tokens_persample filtering"
                    )
                for local_idx in range(entry.data_num):
                    self.valid_indices.append(entry_start + local_idx)
                    self.all_lengths.append(0)
                    kept_in_entry += 1
                kept_total += kept_in_entry
                self.filtered_cumulative_sizes.append(kept_total)
                continue

            for local_idx, sample_len in enumerate(entry.lengths):
                global_idx = entry_start + local_idx
                if self.max_tokens_persample is not None and sample_len > self.max_tokens_persample:
                    self.filtered_samples += 1
                    continue
                self.valid_indices.append(global_idx)
                self.all_lengths.append(sample_len)
                kept_in_entry += 1

            kept_total += kept_in_entry
            self.filtered_cumulative_sizes.append(kept_total)

        self.raw_total_samples = total
        self.total_samples = len(self.valid_indices)
        self.kept_ratio = (self.total_samples / self.raw_total_samples) if self.raw_total_samples > 0 else 0.0
        self.filtered_ratio = (self.filtered_samples / self.raw_total_samples) if self.raw_total_samples > 0 else 0.0

        if self.total_samples == 0:
            raise ValueError(
                f"All {self.raw_total_samples} samples were filtered out by "
                f"--max_tokens_persample={self.max_tokens_persample}"
            )

    def _decode_lmdb_tokens(self, token_ids, skip_special=True):
        """Decode int32 token IDs from LMDB using the LMDB tokenizer."""
        ids = token_ids.tolist()
        if self.lmdb_tokenizer is not None:
            text = self.lmdb_tokenizer.decode(ids, skip_special_tokens=skip_special)
        else:
            text = self.tokenizer.decode(ids, skip_special_tokens=skip_special)

        for src_token, dst_token in LMDB_TEXT_TOKEN_REPLACEMENTS.items():
            text = text.replace(src_token, dst_token)
        return text

    def __len__(self):
        return self.total_samples

    def _locate_raw(self, global_idx):
        """Map global index to (entry_idx, local_idx). Uses bisect for O(log N)."""
        import bisect
        i = bisect.bisect_right(self.cumulative_sizes, global_idx)
        if i >= len(self.cumulative_sizes):
            raise IndexError(f"Index {global_idx} out of range (raw_total={self.raw_total_samples})")
        local_idx = global_idx - (self.cumulative_sizes[i - 1] if i > 0 else 0)
        return i, local_idx

    def _build_prompt(self, wav_lang, tgt_lang, asr_text, kws_pairs):
        """Build the translation prompt string."""
        lang_map = {'chinese': '中文', 'english': '英文'}
        src_lang_str = lang_map.get(wav_lang, wav_lang)
        tgt_lang_str = lang_map.get(tgt_lang, tgt_lang)

        prompt = f"参考识别内容（可能为空），识别:【{asr_text}】，将音频里面{src_lang_str}翻译成{tgt_lang_str}||"

        # Append keyword pairs
        for src, tgt in kws_pairs:
            prompt += f"{src}->{tgt}||"

        return prompt

    def __getitem__(self, idx):
        raw_idx = self.valid_indices[idx]
        entry_idx, local_idx = self._locate_raw(raw_idx)
        entry = self.entries[entry_idx]

        # 1. Read audio from LMDB
        audio_float = entry.read_parsed('wav', local_idx)
        inputs = self.feature_extractor(
            audio_float,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.squeeze(0)  # (128, padded_T)
        attention_mask = inputs.attention_mask.squeeze(0)
        feature_lens = int(attention_mask.sum().item())
        input_features = input_features[:, :feature_lens].contiguous()

        audio_token_count = int(_get_feat_extract_output_lengths(feature_lens))

        # 2. Read translation (ed_label) -> token ids -> text
        trans_tokens = entry.read_parsed('ed_label', local_idx)
        translation = self._decode_lmdb_tokens(trans_tokens, skip_special=True)

        # 3. Read ASR text and acc
        asr_text = ""
        if entry.has_field('ed_label_asr') and entry.has_field('ed_label_acc'):
            acc_arr = entry.read_parsed('ed_label_acc', local_idx)
            # acc is stored as int32 percentage (0-100)
            acc_val = float(acc_arr[0]) if len(acc_arr) > 0 else 0.0
            # Normalize: if acc > 1, treat as percentage
            if acc_val > 1.0:
                acc_val = acc_val / 100.0

            if acc_val >= self.asr_acc_threshold:
                asr_tokens = entry.read_parsed('ed_label_asr', local_idx)
                asr_text = self._decode_lmdb_tokens(asr_tokens, skip_special=True)

        # 4. Read keywords
        kws_pairs = []
        if entry.has_field('ed_label_kws'):
            kws_tokens = entry.read_parsed('ed_label_kws', local_idx)
            # Empty kws is array([0]) -> skip
            if not (len(kws_tokens) == 1 and kws_tokens[0] == 0):
                kws_text = self._decode_lmdb_tokens(kws_tokens, skip_special=True)
                kws_pairs = parse_kws_text(kws_text)

        # 5. Build prompt
        prompt = self._build_prompt(entry.wav_language, entry.target_language, asr_text, kws_pairs)

        # 6. Build chat template with audio tokens
        # Audio tokens are appended AFTER the prompt text
        audio_tokens_str = f"<|audio_start|>{'<|audio_pad|>' * audio_token_count}<|audio_end|>"
        user_content = prompt + "\n" + audio_tokens_str

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": translation},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # 7. Tokenize
        encodings = self.tokenizer(text, truncation=False, padding=False)
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # 8. Build labels (only compute loss on translation)
        labels = self._build_labels(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": input_features,
            "feature_lens": feature_lens,
            "sample_len": entry.get_length(local_idx),
        }

    def _build_labels(self, input_ids):
        """Mask all tokens before assistant response content."""
        labels = list(input_ids)

        assistant_start_str = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        assistant_tokens = self.tokenizer.encode(assistant_start_str, add_special_tokens=False)

        assistant_pos = None
        for i in range(len(input_ids) - len(assistant_tokens) + 1):
            if input_ids[i:i + len(assistant_tokens)] == assistant_tokens:
                assistant_pos = i + len(assistant_tokens)
                break

        if assistant_pos is not None:
            for i in range(assistant_pos):
                labels[i] = -100
        else:
            # Fallback: try without <think> tags
            assistant_start_str2 = "<|im_start|>assistant\n"
            assistant_tokens2 = self.tokenizer.encode(assistant_start_str2, add_special_tokens=False)
            for i in range(len(input_ids) - len(assistant_tokens2) + 1):
                if input_ids[i:i + len(assistant_tokens2)] == assistant_tokens2:
                    assistant_pos = i + len(assistant_tokens2)
                    break
            if assistant_pos is not None:
                for i in range(assistant_pos):
                    labels[i] = -100
            else:
                logger.warning("Could not find assistant start position, masking nothing")

        return labels


# ---------------------------------------------------------------------------
# Dynamic Batch Sampler (by total tokens)
# ---------------------------------------------------------------------------
class DynamicBatchSampler(Sampler):
    """Groups samples into batches where total token count <= batch_tokens.

    Works with distributed training: each rank gets a different subset of batches.
    Uses dataset entry boundaries to keep chunked shuffling IO-friendly.
    Also applies length-aware sortish bucketing and wave-balanced dispatch so
    each global step sees batches with more similar costs across ranks.
    """

    def __init__(self, lengths, batch_tokens, world_size=1, rank=0,
                 shuffle=True, seed=0, drop_last=False, max_batch_size=None,
                 cumulative_sizes=None, shuffle_chunk_entries=8,
                 shuffle_block_samples=5000, sortish_window_size=1024,
                 sampler_mode="legacy"):
        self.lengths = np.array(lengths, dtype=np.int64)  # numpy for fast argsort
        self.batch_tokens = batch_tokens
        self.max_batch_size = max_batch_size
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        # Dataset entry boundaries for locality-aware shuffling
        self.cumulative_sizes = cumulative_sizes
        self.shuffle_chunk_entries = max(1, int(shuffle_chunk_entries))
        self.shuffle_block_samples = max(1, int(shuffle_block_samples))
        self.sortish_window_size = max(1, int(sortish_window_size))
        self.sampler_mode = sampler_mode

        # Pre-compute batches for length estimation
        self._num_batches = self._estimate_num_batches()

    def _estimate_num_batches(self):
        """Estimate total number of batches (for __len__)."""
        sorted_indices = np.argsort(self.lengths).tolist()
        batches = self._build_batches(sorted_indices)
        # Distribute across ranks and pad to ensure all ranks have same count
        max_batches_per_rank = (len(batches) + self.world_size - 1) // self.world_size
        return max_batches_per_rank if max_batches_per_rank > 0 else 1

    def _build_batches(self, indices):
        """Group indices into batches by total token count and max_batch_size."""
        batches = []
        current_batch = []
        current_tokens = 0

        for idx in indices:
            sample_len = self.lengths[idx]
            if sample_len <= 0:
                sample_len = 1  # avoid zero-length samples

            # Check if adding this sample would exceed limits
            exceeds_tokens = current_batch and current_tokens + sample_len > self.batch_tokens
            exceeds_size = self.max_batch_size and len(current_batch) >= self.max_batch_size

            if exceeds_tokens or exceeds_size:
                batches.append(current_batch)
                current_batch = [idx]
                current_tokens = sample_len
            else:
                current_batch.append(idx)
                current_tokens += sample_len

        if current_batch and not self.drop_last:
            batches.append(current_batch)

        return batches

    def _sortish_indices(self, indices):
        """Sort short windows by length while keeping window order random."""
        if self.sortish_window_size <= 1 or len(indices) <= 1:
            return indices

        sorted_indices = []
        for start in range(0, len(indices), self.sortish_window_size):
            window = indices[start:start + self.sortish_window_size]
            window = sorted(window, key=lambda idx: self.lengths[idx], reverse=True)
            sorted_indices.extend(window)
        return sorted_indices

    def _batch_cost(self, batch):
        return int(sum(max(1, self.lengths[idx]) for idx in batch))

    def _balance_batch_waves(self, batches, rng):
        """Arrange batches in world-size waves with similar costs."""
        if len(batches) <= self.world_size:
            return batches

        batch_infos = [{"batch": batch, "cost": self._batch_cost(batch)} for batch in batches]
        rng.shuffle(batch_infos)
        batch_infos.sort(key=lambda item: item["cost"], reverse=True)

        waves = []
        for start in range(0, len(batch_infos), self.world_size):
            wave = batch_infos[start:start + self.world_size]
            rng.shuffle(wave)
            waves.append(wave)

        wave_order = rng.permutation(len(waves))
        ordered_batches = []
        for wave_idx in wave_order:
            ordered_batches.extend(item["batch"] for item in waves[wave_idx])
        return ordered_batches

    def __iter__(self):
        if self.shuffle and self.sampler_mode == "global_random":
            rng = np.random.RandomState(self.seed + self.epoch)
            indices = rng.permutation(len(self.lengths)).tolist()
        elif self.shuffle and self.cumulative_sizes is not None:
            # Chunked-locality shuffle with interleaving:
            #
            # Problem: 264 LMDBs, global shuffle causes constant env switching (slow IO).
            #          Pure per-LMDB sequential causes poor randomness.
            #
            # Solution: 3-level shuffle
            #   1. Group 8 LMDBs into a "chunk", shuffle within chunk (IO locality)
            #   2. Split each chunk into small blocks of ~5000 samples
            #   3. Shuffle all blocks globally (randomness across all LMDBs)
            #
            # Effect: at any point during training, samples come from diverse LMDBs,
            # but within a ~5000-sample block, only 8 LMDBs are accessed (few env switches).
            rng = np.random.RandomState(self.seed + self.epoch)
            n_entries = len(self.cumulative_sizes)
            chunk_size = min(self.shuffle_chunk_entries, n_entries)  # LMDBs per chunk
            block_size = self.shuffle_block_samples                  # samples per block

            # Build per-entry shuffled indices
            entry_indices_list = []
            prev = 0
            for cum in self.cumulative_sizes:
                entry_indices = np.arange(prev, cum)
                rng.shuffle(entry_indices)
                entry_indices_list.append(entry_indices)
                prev = cum

            # Shuffle entry order, group into chunks
            entry_order = rng.permutation(n_entries)

            # Build chunks, then split into small blocks
            blocks = []
            for chunk_start in range(0, n_entries, chunk_size):
                chunk_entries = entry_order[chunk_start:chunk_start + chunk_size]
                chunk_indices = np.concatenate([entry_indices_list[i] for i in chunk_entries])
                rng.shuffle(chunk_indices)
                # Split this chunk into blocks
                for blk_start in range(0, len(chunk_indices), block_size):
                    blocks.append(chunk_indices[blk_start:blk_start + block_size])

            # Shuffle block order globally → interleave chunks
            block_order = rng.permutation(len(blocks))
            indices = np.concatenate([blocks[i] for i in block_order]).tolist()
        elif self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            indices = rng.permutation(len(self.lengths)).tolist()
        else:
            indices = np.argsort(self.lengths).tolist()

        if self.shuffle and self.sampler_mode == "balanced":
            indices = self._sortish_indices(indices)

        # Build batches
        batches = self._build_batches(indices)

        if self.shuffle:
            rng2 = np.random.RandomState(self.seed + self.epoch + 1000)
            if self.sampler_mode == "balanced":
                # Build world-size "waves" of similar-cost batches so each rank sees a
                # comparable amount of work at the same global step.
                batches = self._balance_batch_waves(batches, rng2)
            else:
                # Legacy/global_random behavior: randomize batch order globally after
                # batch construction, without extra length-aware balancing.
                batch_order = rng2.permutation(len(batches))
                batches = [batches[i] for i in batch_order]

        # Distribute batches across ranks (round-robin)
        rank_batches = batches[self.rank::self.world_size]

        # Pad to ensure all ranks have the same number of batches (critical for EP all-to-all sync)
        max_batches_per_rank = (len(batches) + self.world_size - 1) // self.world_size
        while len(rank_batches) < max_batches_per_rank:
            # Pad with the last batch (or first batch if empty)
            if rank_batches:
                rank_batches.append(rank_batches[-1])
            elif batches:
                rank_batches.append(batches[0])

        # Yield individual indices; DataLoader with batch_size=1 will call __getitem__ per index,
        # but we need to yield entire batches. Use a flat list and let the collator handle it.
        # Actually, we yield lists (batches) so DataLoader should use batch_sampler.
        for batch in rank_batches:
            yield batch

    def __len__(self):
        return self._num_batches

    def set_epoch(self, epoch):
        self.epoch = epoch


# ---------------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------------
@dataclass
class DataCollator:
    """Original Pad-based DataCollator (kept for compatibility)."""
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Pad text sequences
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": [], "sample_lens": []}

        for f in features:
            seq_len = min(len(f["input_ids"]), max_len)
            ids = f["input_ids"][:seq_len]
            mask = f["attention_mask"][:seq_len]
            labs = f["labels"][:seq_len]
            pad_len = max_len - seq_len
            batch["input_ids"].append(ids + [self.tokenizer.pad_token_id] * pad_len)
            batch["attention_mask"].append(mask + [0] * pad_len)
            batch["labels"].append(labs + [-100] * pad_len)
            batch["sample_lens"].append(f["sample_len"])

        # Concat audio features by real lengths (no padding waste)
        audio_list = []
        feature_lens = []
        for f in features:
            audio_list.append(f["input_features"])   # each is (128, real_len)
            feature_lens.append(f["feature_lens"])

        batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
        batch["attention_mask"] = torch.tensor(batch["attention_mask"], dtype=torch.long)
        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long)
        batch["sample_lens"] = torch.tensor(batch["sample_lens"], dtype=torch.long)
        batch["input_features"] = torch.cat(audio_list, dim=1)  # (128, total_len)
        batch["feature_lens"] = torch.tensor(feature_lens, dtype=torch.long)

        return batch


@dataclass
class PackedDataCollator:
    """Pack-based DataCollator - packs multiple samples into a single sequence.

    Eliminates padding waste by concatenating sequences with cu_seqlens boundaries.
    Compatible with FlashAttention-2 varlen mode.
    """
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # ========== Text sequences: Pack format ==========
        packed_input_ids = []
        packed_labels = []
        cu_seqlens = [0]  # Cumulative sequence lengths, starts at 0
        position_ids = []
        sample_lens = []

        for f in features:
            seq_len = len(f["input_ids"])
            packed_input_ids.extend(f["input_ids"])
            packed_labels.extend(f["labels"])
            cu_seqlens.append(cu_seqlens[-1] + seq_len)
            # Position IDs: independent counting within each sample
            position_ids.extend(list(range(seq_len)))
            sample_lens.append(f["sample_len"])

        # ========== Audio features: concat by real lengths (same as before) ==========
        audio_list = []
        feature_lens = []
        for f in features:
            audio_list.append(f["input_features"])
            feature_lens.append(f["feature_lens"])

        batch = {
            "input_ids": torch.tensor(packed_input_ids, dtype=torch.long),  # (total_len,)
            "position_ids": torch.tensor(position_ids, dtype=torch.long),   # (total_len,)
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),      # (batch_size+1,)
            "labels": torch.tensor(packed_labels, dtype=torch.long),        # (total_len,)
            "sample_lens": torch.tensor(sample_lens, dtype=torch.long),     # (batch_size,)
            "input_features": torch.cat(audio_list, dim=1),                 # (128, total_audio_len)
            "feature_lens": torch.tensor(feature_lens, dtype=torch.long),   # (batch_size,)
        }
        return batch


def setup_distributed():
    """Initialize distributed training."""
    # torchrun sets these env vars
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Verify multi-node setup
    if world_size > 1:
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "13525")
        print(f"[Rank {rank}] Connecting to {master_addr}:{master_port}, world_size={world_size}", flush=True)

    dist.init_process_group(backend="hccl")

    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    return rank, world_size, local_rank, device


def create_ep_group(world_size):
    """Create EP process group (all ranks in one EP group)."""
    ranks = list(range(world_size))
    ep_group = dist.new_group(ranks)
    return ep_group


def get_non_expert_params(model):
    """Get parameters that are NOT expert-specific (need gradient sync)."""
    non_expert_params = []
    expert_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'local_experts.gate_up_proj' in name or 'local_experts.down_proj' in name:
            expert_params.append(param)
        else:
            non_expert_params.append(param)
    return non_expert_params, expert_params


def sync_gradients(params, group_size, group=None):
    """Flatten grads into one buffer, single all-reduce, then unflatten."""
    grads = []
    shapes = []
    numels = []
    for param in params:
        if param.grad is not None:
            grads.append(param.grad.data.reshape(-1))
            shapes.append(param.grad.shape)
            numels.append(param.grad.numel())
        else:
            shapes.append(None)
            numels.append(0)

    if not grads:
        return

    flat = torch.cat(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
    flat /= group_size

    offset = 0
    for i, param in enumerate(params):
        if shapes[i] is not None:
            n = numels[i]
            param.grad.data = flat[offset:offset + n].reshape(shapes[i])
            offset += n


def sync_non_expert_gradients_dp(non_expert_params, dp_size, dp_group):
    """Sync non-expert gradients across DP group only."""
    sync_gradients(non_expert_params, dp_size, dp_group)


def get_cosine_annealing_schedule(optimizer, warmup_steps, lr_decay_iters, min_lr, max_lr):
    """Cosine Annealing LR scheduler with optional warmup.

    Args:
        warmup_steps: linear warmup steps (0 = no warmup, start from max_lr)
        lr_decay_iters: total cosine decay steps (after warmup)
        min_lr: minimum learning rate at end of decay
        max_lr: maximum learning rate (= optimizer's base lr)
    """
    def lr_lambda(current_step):
        # Phase 1: warmup (linear from 0 to max_lr)
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        # Phase 2: cosine decay (from max_lr to min_lr)
        decay_step = current_step - warmup_steps
        if decay_step >= lr_decay_iters:
            return min_lr / max_lr
        progress = float(decay_step) / float(max(1, lr_decay_iters))
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr + (max_lr - min_lr) * coeff) / max_lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_path", type=str, default="/repo1/yjjiang11/work/vibecoding/Qwen3.5/qwen3.5_MOE_AST_sft/models/Qwen/Qwen3_5_CausalLM")
    parser.add_argument("--asr_path", type=str, default="/repo1/yjjiang11/work/vibecoding/Qwen3.5/qwen3.5_MOE_AST_sft/models/Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--tokenizer_path", type=str, default="/repo1/yjjiang11/work/vibecoding/Qwen3.5/qwen3.5_MOE_AST_sft/models/Qwen/Qwen3___5-35B-A3B")
    parser.add_argument("--train_data_path", type=str, default="/repo1/yjjiang11/work/vibecoding/Qwen3.5/qwenasr_qwen3p5moe_sft_lmdb/lmdbdata/train.json")
    parser.add_argument("--eval_data_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="output_ep")
    parser.add_argument("--batch_tokens", type=int, default=1024, help="Max total tokens per batch (sum of lengths)")
    parser.add_argument("--max_batch_size", type=int, default=None, help="Max number of samples per batch (hard limit to prevent OOM)")
    parser.add_argument("--shuffle_chunk_entries", type=int, default=8,
                        help="LMDB entries per locality-preserving shuffle chunk")
    parser.add_argument("--shuffle_block_samples", type=int, default=5000,
                        help="Approximate sample count per shuffled locality block")
    parser.add_argument("--sortish_window_size", type=int, default=1024,
                        help="Window size for length-aware sortish bucketing before batch building")
    parser.add_argument(
        "--sampler_mode",
        type=str,
        default="legacy",
        choices=["global_random", "legacy", "balanced"],
        help=(
            "Batch sampler behavior: global_random matches acf84a5 global shuffle ordering; "
            "legacy matches 93cae8d chunk/block locality ordering; "
            "balanced enables sortish + wave balancing on top of legacy locality"
        ),
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--lr_decay_iters", type=int, default=0, help="Total cosine decay steps after warmup (0 = use total_steps - warmup_steps)")
    parser.add_argument("--min_lr", type=float, default=0.0, help="Minimum learning rate at end of cosine decay")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_tokens_persample", type=int, default=None,
                        help="Filter out samples whose wav lengths_file value exceeds this threshold")
    parser.add_argument("--asr_acc_threshold", type=float, default=2.0, help="ASR accuracy threshold for filling recognition text into prompt")
    parser.add_argument("--lmdb_tokenizer_path", type=str, default="/repo1/yjjiang11/work/vibecoding/Qwen3.5/qwenasr_qwen3p5moe_sft_lmdb/lmdbdata/res", help="Tokenizer used to encode LMDB text data (for decoding int32 tokens back to text)")
    parser.add_argument("--loss_bucket_edges", type=str, default="640,1280,1920,2600",
                        help="Comma-separated sample length edges used for bucketed loss logging")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint dir to resume training from")
    parser.add_argument("--ep_size", type=int, default=8, help="Expert parallel size (default: 8, use world_size for global EP)")
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA training (default: full finetuning)")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--use_packed_format", action="store_true", help="Use packed sequence format (eliminates padding waste, requires FA2)")
    parser.add_argument("--max_steps", type=int, default=0, help="If >0, stop after this many optimizer steps (for smoke tests / profiling)")
    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank, device = setup_distributed()

    # EP within node; all ranks still consume different data batches.
    # Expert shards are replicated across nodes, so same ep_rank forms a replica-sync group.
    ep_size = args.ep_size if args.ep_size > 0 else world_size
    if world_size % ep_size != 0:
        raise ValueError(f"world_size={world_size} must be divisible by ep_size={ep_size}")

    num_expert_replicas = world_size // ep_size
    expert_replica_rank = rank // ep_size
    ep_rank = rank % ep_size

    # Create output dir early so we can set up logging
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Logging: console + per-rank log file
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Console handler (rank 0 only prints INFO, others WARN)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if rank == 0 else logging.WARN)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    # File handler: each rank writes to output_dir/logs/rank{N}.log
    file_handler = logging.FileHandler(os.path.join(log_dir, f"rank{rank}.log"), mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    if rank == 0:
        logger.info(
            f"Training: world_size={world_size}, ep_size={ep_size}, "
            f"expert_replicas={num_expert_replicas}"
        )
        logger.info(f"Args: {args}")
        logger.info(
            "Training config:\n%s",
            json.dumps(vars(args), ensure_ascii=False, indent=2, sort_keys=True),
        )

    # Create EP group (within node) — for expert all-to-all communication
    ep_groups = []
    for i in range(num_expert_replicas):
        ep_group_ranks = list(range(i * ep_size, (i + 1) * ep_size))
        ep_groups.append(dist.new_group(ep_group_ranks))

    ep_group = ep_groups[expert_replica_rank]
    ep_group_ranks = list(range(expert_replica_rank * ep_size, (expert_replica_rank + 1) * ep_size))

    # Create expert replica groups across nodes: same ep_rank, different nodes.
    expert_replica_groups = []
    for i in range(ep_size):
        replica_group_ranks = list(range(i, world_size, ep_size))
        expert_replica_groups.append(dist.new_group(replica_group_ranks))

    expert_replica_group = expert_replica_groups[ep_rank]
    expert_replica_group_ranks = list(range(ep_rank, world_size, ep_size))

    # Load tokenizer and feature extractor
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.asr_path, trust_remote_code=True)
    audio_token_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")

    if rank == 0:
        logger.info(f"Audio token ID: {audio_token_id}")

    # Load model with EP
    model = EPSpeechTranslationModel(
        llm_path=args.llm_path,
        asr_path=args.asr_path,
        audio_token_id=audio_token_id,
        ep_size=ep_size,
        ep_rank=ep_rank,
        ep_group=ep_group,
        device=device,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # LoRA or full finetuning
    if args.use_lora:
        apply_lora_to_model(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
        # In LoRA mode, only LoRA params + audio encoder trainable params need optimization
        # No expert params to train, no expert gradient sync needed
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if rank == 0:
            t_count = sum(p.numel() for p in trainable_params)
            logger.info(f"LoRA mode: {t_count/1e6:.2f}M trainable params")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    else:
        # Full finetuning: separate non-expert and expert params
        non_expert_params, expert_params = get_non_expert_params(model)
        if rank == 0:
            ne_count = sum(p.numel() for p in non_expert_params)
            e_count = sum(p.numel() for p in expert_params)
            logger.info(f"Full finetuning: Non-expert {ne_count/1e6:.1f}M, Expert (local) {e_count/1e6:.1f}M")

        optimizer = torch.optim.AdamW(
            [
                {"params": non_expert_params, "lr": args.learning_rate, "weight_decay": args.weight_decay},
                {"params": expert_params, "lr": args.learning_rate, "weight_decay": args.weight_decay},
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    # Dataset: LMDB-based with dynamic batching
    train_dataset = LmdbSpeechTranslationDataset(
        args.train_data_path, tokenizer, feature_extractor,
        args.max_tokens_persample, args.asr_acc_threshold,
        lmdb_tokenizer_path=args.lmdb_tokenizer_path,
    )
    batch_sampler = DynamicBatchSampler(
        lengths=train_dataset.all_lengths,
        batch_tokens=args.batch_tokens,
        # Full data-parallel sharding: every rank gets a different batch.
        world_size=world_size,
        rank=rank,
        shuffle=True,
        seed=0,
        max_batch_size=args.max_batch_size,
        cumulative_sizes=train_dataset.filtered_cumulative_sizes,
        shuffle_chunk_entries=args.shuffle_chunk_entries,
        shuffle_block_samples=args.shuffle_block_samples,
        sortish_window_size=args.sortish_window_size,
        sampler_mode=args.sampler_mode,
    )

    # Choose data collator based on format
    if args.use_packed_format:
        if rank == 0:
            logger.info("Using PackedDataCollator (pack format - eliminates padding waste)")
        data_collator = PackedDataCollator(tokenizer=tokenizer)
    else:
        if rank == 0:
            logger.info("Using DataCollator (pad format - legacy)")
        data_collator = DataCollator(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler,
        collate_fn=data_collator, num_workers=4, pin_memory=True,
        persistent_workers=True,
    )

    logger.info(
        "Parallel topology | "
        f"rank={rank} local_rank={local_rank} "
        f"ep_rank={ep_rank} expert_replica_rank={expert_replica_rank} "
        f"ep_group={ep_group_ranks} expert_replica_group={expert_replica_group_ranks} "
        f"sampler_rank={rank}/{world_size}"
    )

    # LR scheduler
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.num_epochs
    lr_decay_iters = args.lr_decay_iters if args.lr_decay_iters > 0 else total_steps - args.warmup_steps
    scheduler = get_cosine_annealing_schedule(
        optimizer, args.warmup_steps, lr_decay_iters, args.min_lr, args.learning_rate
    )
    loss_bucket_edges = _parse_loss_bucket_edges(args.loss_bucket_edges)
    loss_bucket_labels = [
        _format_loss_bucket_label(loss_bucket_edges, idx)
        for idx in range(len(loss_bucket_edges) + 1)
    ]
    loss_bucket_edges_tensor = torch.tensor(loss_bucket_edges, dtype=torch.long, device=device)
    num_loss_buckets = len(loss_bucket_labels)

    if rank == 0:
        logger.info(
            f"Dataset stats | raw_total={train_dataset.raw_total_samples} "
            f"filtered={train_dataset.filtered_samples} ({train_dataset.filtered_ratio:.2%}) "
            f"active={len(train_dataset)} ({train_dataset.kept_ratio:.2%}) "
            f"max_tokens_persample={args.max_tokens_persample}"
        )
        logger.info(
            "Batch sampler | "
            f"sampler_mode={args.sampler_mode} "
            f"batch_tokens={args.batch_tokens} max_batch_size={args.max_batch_size} "
            f"shuffle_chunk_entries={args.shuffle_chunk_entries} "
            f"shuffle_block_samples={args.shuffle_block_samples} "
            f"sortish_window_size={args.sortish_window_size}"
        )
        logger.info(f"Loss buckets | edges={loss_bucket_edges if loss_bucket_edges else ['all']}")
        logger.info(
            f"Training: raw_total={train_dataset.raw_total_samples}, active={len(train_dataset)}, "
            f"{steps_per_epoch} steps/epoch, {total_steps} total steps"
        )

    # Resume from checkpoint
    resume_epoch = 0
    resume_global_step = 0
    resume_step_in_epoch = 0  # micro-steps to skip in the resumed epoch

    if args.resume_from_checkpoint:
        ckpt_dir = args.resume_from_checkpoint
        # Load model weights
        if args.use_lora:
            load_lora_checkpoint(model, ckpt_dir, ep_rank)
        else:
            load_ep_checkpoint(model, ckpt_dir, ep_rank)

        ckpt_world_size = _infer_checkpoint_world_size(ckpt_dir)
        can_resume_train_state = ckpt_world_size == world_size

        if rank == 0:
            if ckpt_world_size is None:
                logger.warning(
                    f"Could not infer checkpoint world_size from {ckpt_dir}; "
                    "loading model weights only (optimizer/scheduler reset)"
                )
            elif can_resume_train_state:
                logger.info(
                    f"Checkpoint world_size matches current run ({world_size}); "
                    "loading model weights and training state"
                )
            else:
                logger.warning(
                    f"Checkpoint world_size={ckpt_world_size} does not match current world_size={world_size}; "
                    "loading model weights only (optimizer/scheduler reset)"
                )

        # Load training state only when checkpoint topology matches current run.
        if can_resume_train_state:
            state_path = os.path.join(ckpt_dir, f"train_state_rank{rank}.pt")
            if os.path.exists(state_path):
                train_state = torch.load(state_path, map_location="cpu")
                optimizer.load_state_dict(train_state["optimizer"])
                scheduler.load_state_dict(train_state["scheduler"])
                resume_global_step = train_state["global_step"]
                resume_epoch = train_state["epoch"]
                # Calculate how many micro-steps to skip in the current epoch
                resume_step_in_epoch = resume_global_step * args.gradient_accumulation_steps - \
                    resume_epoch * steps_per_epoch * args.gradient_accumulation_steps
                if rank == 0:
                    logger.info(
                        f"Resumed from {ckpt_dir}: epoch={resume_epoch}, global_step={resume_global_step}, "
                        f"skipping {resume_step_in_epoch} micro-steps in epoch {resume_epoch+1}"
                    )
            else:
                logger.warning(
                    f"No train_state found at {state_path}, loading model weights only "
                    "(optimizer/scheduler reset)"
                )
        else:
            resume_epoch = 0
            resume_global_step = 0
            resume_step_in_epoch = 0

    # Training loop
    model.train()
    global_step = resume_global_step
    accumulated_loss = 0.0
    local_step_loss_sum = 0.0
    local_step_valid_tokens = 0
    local_window_loss_sum = 0.0
    local_window_valid_tokens = 0
    local_bucket_loss_sums = torch.zeros(num_loss_buckets, dtype=torch.float32, device=device)
    local_bucket_token_counts = torch.zeros(num_loss_buckets, dtype=torch.float32, device=device)
    local_bucket_sample_counts = torch.zeros(num_loss_buckets, dtype=torch.float32, device=device)

    # Profiler: track time for each stage (rank 0 only, every logging_steps)
    prof = {
        "data":     0.0,  # DataLoader fetch + H2D transfer
        "forward":  0.0,  # model forward
        "backward": 0.0,  # loss.backward
        "sync":     0.0,  # gradient all-reduce
        "optim":    0.0,  # optimizer.step + scheduler + zero_grad
    }
    prof_steps = 0
    prof_t = time.time()  # timestamp of last batch-end (used to measure data time)

    for epoch in range(resume_epoch, args.num_epochs):
        batch_sampler.set_epoch(epoch)
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            # Skip already-trained micro-steps when resuming
            if epoch == resume_epoch and step < resume_step_in_epoch:
                continue

            # [PROF] data time = time from last batch-end to now (DataLoader + H2D)
            if rank == 0:
                torch.npu.synchronize() if hasattr(torch, 'npu') else None
                t0 = time.time()
                prof["data"] += t0 - prof_t

            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward
            if rank == 0:
                torch.npu.synchronize() if hasattr(torch, 'npu') else None
                t1 = time.time()
            outputs = model(**batch)
            valid_label_tokens = int((batch["labels"] != -100).sum().item())
            local_step_valid_tokens += valid_label_tokens
            local_step_loss_sum += float(outputs.loss.detach().float().item() * valid_label_tokens)
            with torch.no_grad():
                # Handle both pack and pad formats for per-sample loss bucket calculation
                is_packed = "cu_seqlens" in batch

                shift_labels = F.pad(batch["labels"], (0, 1), value=-100)[..., 1:].contiguous()
                per_token_loss = F.cross_entropy(
                    outputs.logits.detach().float().reshape(-1, outputs.logits.shape[-1]),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                )  # (total_tokens,) for pack or (batch*seq,) for pad

                if is_packed:
                    # Pack format: manually aggregate per sample using cu_seqlens
                    cu_seqlens = batch["cu_seqlens"]
                    batch_size = len(cu_seqlens) - 1
                    sample_loss_sums = torch.zeros(batch_size, dtype=torch.float32, device=device)
                    sample_token_counts = torch.zeros(batch_size, dtype=torch.float32, device=device)

                    for b in range(batch_size):
                        start = cu_seqlens[b].item()
                        end = cu_seqlens[b+1].item()
                        sample_labels = shift_labels[start:end]
                        sample_per_token_loss = per_token_loss[start:end]
                        valid_mask = sample_labels != -100
                        sample_loss_sums[b] = (sample_per_token_loss * valid_mask.to(sample_per_token_loss.dtype)).sum()
                        sample_token_counts[b] = valid_mask.sum().float()
                else:
                    # Pad format: reshape and sum per sample
                    per_token_loss = per_token_loss.view_as(shift_labels)  # (batch, seq)
                    valid_mask = shift_labels != -100
                    sample_loss_sums = (per_token_loss * valid_mask.to(per_token_loss.dtype)).sum(dim=1).float()
                    sample_token_counts = valid_mask.sum(dim=1).float()

                sample_loss_sums = sample_loss_sums.to(dtype=local_bucket_loss_sums.dtype)
                sample_token_counts = sample_token_counts.to(dtype=local_bucket_token_counts.dtype)
                if len(loss_bucket_edges) > 0:
                    bucket_indices = torch.bucketize(batch["sample_lens"], loss_bucket_edges_tensor)
                else:
                    bucket_indices = torch.zeros_like(batch["sample_lens"])
                local_bucket_loss_sums.index_add_(0, bucket_indices, sample_loss_sums)
                local_bucket_token_counts.index_add_(0, bucket_indices, sample_token_counts)
                local_bucket_sample_counts.index_add_(
                    0, bucket_indices, torch.ones_like(sample_token_counts, dtype=local_bucket_sample_counts.dtype)
                )
            loss = outputs.loss / args.gradient_accumulation_steps
            accumulated_loss += loss.item()
            if rank == 0:
                torch.npu.synchronize() if hasattr(torch, 'npu') else None
                prof["forward"] += time.time() - t1

            # Debug: print top-1 prediction
            debug_step = global_step + 1
            if rank == 0 and (step + 1) % args.gradient_accumulation_steps == 0 and debug_step % 2 == 0:
                with torch.no_grad():
                    is_packed = "cu_seqlens" in batch

                    if is_packed:
                        # Pack format: extract first sample using cu_seqlens
                        cu_seqlens = batch["cu_seqlens"]
                        start = cu_seqlens[0].item()
                        end = cu_seqlens[1].item()
                        logits = outputs.logits[start:end]  # (seq_len, vocab_size)
                        input_ids = batch["input_ids"][start:end].cpu()
                        labels = batch["labels"][start:end].cpu()
                    else:
                        # Pad format: extract first sample directly
                        logits = outputs.logits[0]  # (seq_len, vocab_size)
                        input_ids = batch["input_ids"][0].cpu()
                        labels = batch["labels"][0].cpu()

                    # Transformers shifts internally: logits[i] predicts labels[i+1]
                    # So to get predictions for labels[i], we need logits[i-1]
                    pred_ids = logits.argmax(dim=-1).cpu()

                    # Get positions where labels != -100
                    valid_positions = (labels != -100).nonzero(as_tuple=True)[0]

                    # Shift back by 1 to get the logits that predict these labels
                    pred_positions = valid_positions - 1
                    pred_positions = pred_positions[pred_positions >= 0]  # Remove negative indices

                    pred_tokens = pred_ids[pred_positions].tolist()
                    pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=False).replace('\n','')
                    pred_text = _compress_audio_pad_runs(pred_text)
                    target_tokens = labels[valid_positions[-len(pred_positions):]]
                    if len(valid_positions) > 0:
                        first_target_pos = int(valid_positions[0].item())
                        prompt_ids = input_ids[:first_target_pos].tolist()
                    else:
                        prompt_ids = input_ids.tolist()
                    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False).replace('\n', '')
                    prompt_text = _compress_audio_pad_runs(prompt_text)
                    target_text = tokenizer.decode(target_tokens.tolist(), skip_special_tokens=False).replace('\n', '')
                    target_text = _compress_audio_pad_runs(target_text)
                    if len(pred_positions) > 0:
                        pred_token_tensor = pred_ids[pred_positions]
                        token_acc = (pred_token_tensor == target_tokens).float().mean().item()
                    else:
                        token_acc = float("nan")
                    logger.info(f"Step {debug_step} - Prompt: {prompt_text}")
                    logger.info(f"Step {debug_step} - Target: {target_text}")
                    logger.info(
                        f"Step {debug_step} - Top-1 pred: {pred_text} | "
                        f"top1_token_acc={token_acc:.4f}"
                    )

            # Backward
            if rank == 0:
                torch.npu.synchronize() if hasattr(torch, 'npu') else None
                t2 = time.time()
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if rank == 0:
                    torch.npu.synchronize() if hasattr(torch, 'npu') else None
                    prof["backward"] += time.time() - t2
                    t3 = time.time()

                if args.use_lora:
                    sync_gradients(trainable_params, world_size)
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                else:
                    sync_gradients(non_expert_params, world_size)
                    sync_gradients(expert_params, num_expert_replicas, expert_replica_group)
                    all_params = list(non_expert_params) + list(expert_params)
                    torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)

                if rank == 0:
                    torch.npu.synchronize() if hasattr(torch, 'npu') else None
                    prof["sync"] += time.time() - t3
                    t4 = time.time()

                # Step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                local_window_loss_sum += local_step_loss_sum
                local_window_valid_tokens += local_step_valid_tokens
                rank0_step_loss = accumulated_loss
                local_step_loss_sum = 0.0
                local_step_valid_tokens = 0

                if rank == 0:
                    torch.npu.synchronize() if hasattr(torch, 'npu') else None
                    prof["optim"] += time.time() - t4
                    prof_steps += 1

                if global_step % args.logging_steps == 0:
                    loss_stats = torch.tensor(
                        [local_window_loss_sum, float(local_window_valid_tokens)],
                        dtype=torch.float32,
                        device=device,
                    )
                    if dist.is_initialized():
                        dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
                    global_window_loss_sum = float(loss_stats[0].item())
                    global_window_valid_tokens = int(loss_stats[1].item())

                    if rank == 0:
                        lr = scheduler.get_last_lr()[0]
                        global_loss = (
                            global_window_loss_sum / global_window_valid_tokens
                            if global_window_valid_tokens > 0 else float("nan")
                        )
                        logger.info(
                            f"Epoch {epoch+1} | Step {global_step}/{total_steps} | "
                            f"global_loss={global_loss:.4f} | "
                            f"rank0_step_loss={rank0_step_loss:.4f} | "
                            f"window_valid_tokens={global_window_valid_tokens} | "
                            f"LR: {lr:.2e}"
                        )
                        bucket_stats = torch.cat(
                            [local_bucket_loss_sums, local_bucket_token_counts, local_bucket_sample_counts]
                        )
                    else:
                        bucket_stats = torch.cat(
                            [local_bucket_loss_sums, local_bucket_token_counts, local_bucket_sample_counts]
                        )
                    if dist.is_initialized():
                        dist.all_reduce(bucket_stats, op=dist.ReduceOp.SUM)
                    global_bucket_loss_sums = bucket_stats[:num_loss_buckets]
                    global_bucket_token_counts = bucket_stats[num_loss_buckets:2 * num_loss_buckets]
                    global_bucket_sample_counts = bucket_stats[2 * num_loss_buckets:]
                    if rank == 0:
                        bucket_parts = []
                        for idx, label in enumerate(loss_bucket_labels):
                            token_count = int(global_bucket_token_counts[idx].item())
                            sample_count = int(global_bucket_sample_counts[idx].item())
                            bucket_loss = (
                                global_bucket_loss_sums[idx].item() / token_count
                                if token_count > 0 else float("nan")
                            )
                            bucket_parts.append(
                                f"{label}: loss={bucket_loss:.4f}, samples={sample_count}, tokens={token_count}"
                            )
                        logger.info("[LOSS_BUCKETS] " + " | ".join(bucket_parts))
                    # Profile report
                    if rank == 0 and prof_steps > 0:
                        total_t = sum(prof.values())
                        logger.info(
                            f"[PROF] avg over {prof_steps} optim-steps | "
                            f"data={prof['data']/prof_steps*1000:.1f}ms  "
                            f"forward={prof['forward']/prof_steps*1000:.1f}ms  "
                            f"backward={prof['backward']/prof_steps*1000:.1f}ms  "
                            f"sync={prof['sync']/prof_steps*1000:.1f}ms  "
                            f"optim={prof['optim']/prof_steps*1000:.1f}ms  "
                            f"total={total_t/prof_steps*1000:.1f}ms"
                        )
                        # Reset counters
                        for k in prof: prof[k] = 0.0
                        prof_steps = 0
                    local_window_loss_sum = 0.0
                    local_window_valid_tokens = 0
                    local_bucket_loss_sums.zero_()
                    local_bucket_token_counts.zero_()
                    local_bucket_sample_counts.zero_()
                accumulated_loss = 0.0

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    save_ep_checkpoint(
                        model, rank, expert_replica_rank, ep_rank, ep_size, save_dir, tokenizer, lora_only=args.use_lora
                    )
                    train_state = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }
                    torch.save(train_state, os.path.join(save_dir, f"train_state_rank{rank}.pt"))
                    if rank == 0:
                        logger.info(f"Checkpoint saved at step {global_step}")

                # Early stop for smoke tests / profiling
                if args.max_steps > 0 and global_step >= args.max_steps:
                    if rank == 0:
                        logger.info(f"Reached max_steps={args.max_steps}, stopping early.")
                    break

            # [PROF] record end-of-batch timestamp for next data timing
            if rank == 0:
                torch.npu.synchronize() if hasattr(torch, 'npu') else None
                prof_t = time.time()

        epoch_time = time.time() - epoch_start
        if rank == 0:
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")

        # Early stop for smoke tests / profiling (break outer epoch loop too)
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # Final save
    save_ep_checkpoint(
        model, rank, expert_replica_rank, ep_rank, ep_size, args.output_dir, tokenizer, lora_only=args.use_lora
    )
    if rank == 0:
        logger.info("Training completed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
