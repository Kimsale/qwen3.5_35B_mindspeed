#!/usr/bin/env python3
"""EP inference: load sharded checkpoint and run inference on test set."""
import os
import json
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoFeatureExtractor
from model_ep import (
    EPSpeechTranslationModel,
    load_ep_checkpoint,
    load_lora_checkpoint,
    apply_lora_to_model,
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def load_checkpoint_auto(model, checkpoint_dir, ep_rank, ep_size, ep_group,
                          lora_rank=8, lora_alpha=16, lora_dropout=0.0):
    """Auto-detect checkpoint type (full / LoRA) and load accordingly."""
    meta_path = os.path.join(checkpoint_dir, "ep_metadata.json")
    lora_only = False
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        lora_only = meta.get("lora_only", False)

    if lora_only:
        if ep_rank == 0:
            logger.info(f"Detected LoRA checkpoint, injecting LoRA (rank={lora_rank}) then loading weights...")
        apply_lora_to_model(model, lora_rank, lora_alpha, lora_dropout)
        load_lora_checkpoint(model, checkpoint_dir, ep_rank)
    else:
        if ep_rank == 0:
            logger.info("Detected full checkpoint, loading shard weights...")
        load_ep_checkpoint(model, checkpoint_dir, ep_rank)

    dist.barrier(group=ep_group)
    if ep_rank == 0:
        mode = "LoRA" if lora_only else "full"
        logger.info(f"Checkpoint loaded [{mode}] from {checkpoint_dir}")


def main():
    # Init distributed
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")

    ep_size = world_size
    ep_rank = rank
    ep_group = dist.new_group(list(range(ep_size)))

    # Paths
    llm_path = "/data/yjjiang11/work/vibecoding/Qwen3.5/qwen3.5_MOE_AST_sft/models/Qwen/Qwen3_5_CausalLM"
    asr_path = "/data/yjjiang11/work/vibecoding/Qwen3.5/qwen3.5_MOE_AST_sft/models/Qwen/Qwen3-ASR-1.7B"
    tokenizer_path = "/data/yjjiang11/work/vibecoding/Qwen3.5/qwen3.5_MOE_AST_sft/models/Qwen/Qwen3___5-35B-A3B"
    test_data_path = "/data/yjjiang11/work/vibecoding/Qwen3.5/qwenasr_qwen3p5moe_sft_lmdb/testdata/test.jsonl"
    checkpoint_dir = "output_ep_fast_lora"
    # LoRA hyper-params (must match training)
    lora_rank = 32
    lora_alpha = 64

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    feature_extractor = AutoFeatureExtractor.from_pretrained(asr_path, trust_remote_code=True)
    audio_token_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")

    # Load model
    if rank == 0:
        logger.info("Loading model with EP...")
    model = EPSpeechTranslationModel(llm_path, asr_path, audio_token_id, ep_size, ep_rank, ep_group, device)
    load_checkpoint_auto(model, checkpoint_dir, ep_rank, ep_size, ep_group, lora_rank, lora_alpha)
    model.eval()

    # Load test data
    with open(test_data_path) as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    if rank == 0:
        logger.info(f"Loaded {len(test_data)} test samples")

    # Inference
    from train_ep import _get_feat_extract_output_lengths
    results = []
    with torch.no_grad():
        for i, item in enumerate(test_data[:10]):
            if rank == 0:
                logger.info(f"Processing sample {i+1}/10...")

            # Load audio
            try:
                import soundfile as sf
                audio, sr = sf.read(item["audio_path"], dtype='float32')
                if sr != 16000:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            except Exception:
                import librosa
                audio, sr = librosa.load(item["audio_path"], sr=16000)

            inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.squeeze(0)  # (128, T)
            feature_lens = input_features.shape[-1]

            # Build prompt
            prompt = item.get("sentence", "")
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

            # Insert audio tokens
            audio_token_count = int(_get_feat_extract_output_lengths(feature_lens))
            audio_tokens = f"<|audio_start|>{'<|audio_pad|>' * audio_token_count}<|audio_end|>"
            text = text.replace(prompt, prompt + "\n" + audio_tokens)

            # Tokenize
            encodings = tokenizer(text, return_tensors="pt")
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            # Audio encoding — use model.forward-compatible path (FA2 single sample)
            input_features = input_features.to(device=device, dtype=torch.bfloat16)
            feature_lens_t = torch.tensor([feature_lens], dtype=torch.int64, device=device)
            audio_outputs = model.audio_encoder(input_features, feature_lens=feature_lens_t)
            audio_embeds = audio_outputs.last_hidden_state.to(dtype=torch.bfloat16)

            # Text embeddings
            text_embeds = model.llm.get_input_embeddings()(input_ids)

            # Replace audio tokens with audio embeddings
            audio_positions = (input_ids[0] == audio_token_id).nonzero(as_tuple=True)[0]
            if len(audio_positions) > 0:
                audio_len = min(audio_embeds.shape[0], len(audio_positions))
                text_embeds[0, audio_positions[:audio_len]] = audio_embeds[:audio_len]

            # Generate
            eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            outputs = model.llm.generate(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False,
                eos_token_id=eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translation = generated_text.strip()

            if rank == 0:
                results.append({
                    "audio_path": item["audio_path"],
                    "sentence": prompt,
                    "reference": item.get("translation", ""),
                    "prediction": translation,
                })
                logger.info(f"  Reference: {item.get('translation', '')}")
                logger.info(f"  Prediction: {translation}")

    # Save results
    if rank == 0:
        output_file = "inference_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_file}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
