from typing import Optional, Dict, Any
import wave

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler


class BaseRandomBatchSampler(StatefulDistributedSampler):
    """
    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. Default: ``True``. (It is not implemented that the drop_last is false.)
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        data_sharding: bool = False,
    ):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.total_samples = len(dataset)
        self.micro_batch_size = batch_size
        self.consumed_samples = 0
        self.next_consumed_samples = None
        self.data_sharding = data_sharding
        self.epoch = 0
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * self.num_replicas
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size
        if not drop_last:
            raise ValueError("It is not implemented that the drop_last is false.")

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        # resume sampler
        if self.next_consumed_samples is not None:
            self.consumed_samples = self.next_consumed_samples
            self.next_consumed_samples = None

        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                           * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.num_replicas
            start_idx = self.rank * bucket_size
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.epoch)
                idx_range_bucket = torch.randperm(bucket_size, generator=g).tolist()
            else:
                idx_range_bucket = list(range(bucket_size))
            idx_range = [start_idx + x for x in idx_range_bucket[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) \
                                * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.epoch)
                idx_range_total = \
                    torch.randperm(full_bucket_size, generator=g).tolist()
            else:
                idx_range_total = list(range(full_bucket_size))
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.rank::self.num_replicas]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []

    def state_dict(self) -> Dict[str, Any]:
        return {self._YIELDED: self.consumed_samples}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self._YIELDED not in state_dict:
            raise ValueError("Invalid state_dict")
        if state_dict[self._YIELDED] < 0:
            raise ValueError("Cannot load state_dict with negative yielded value")
        self.next_consumed_samples = state_dict[self._YIELDED]


class LengthBucketBatchSampler(BaseRandomBatchSampler):
    """
    Batch sampler that keeps distributed global micro-batches close in text/audio
    length. This reduces dynamic-shape churn for multimodal SFT while preserving
    the same samples, model, and global batch semantics.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        data_sharding: bool = False,
        bucket_size_multiplier: int = 64,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            data_sharding=data_sharding,
        )
        if data_sharding:
            raise ValueError("LengthBucketBatchSampler does not support data_sharding=True.")
        self.bucket_size_multiplier = max(1, int(bucket_size_multiplier))
        self.lengths = [self._sample_length(idx) for idx in range(self.total_samples)]

    def _sample_length(self, idx: int) -> int:
        item = self.dataset[idx]
        length = len(item.get("input_ids", []) or [])
        for audio_path in item.get("audios", []) or []:
            try:
                with wave.open(audio_path, "rb") as audio_file:
                    length += audio_file.getnframes() // 320
            except (FileNotFoundError, wave.Error, EOFError, TypeError):
                continue
        return length

    def _epoch_global_batches(self, active_total_samples: int):
        global_batch_size = self.micro_batch_times_data_parallel_size
        indices = list(range(active_total_samples))
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        if not self.shuffle:
            indices.sort(key=lambda idx: self.lengths[idx])
            return [
                indices[start:start + global_batch_size]
                for start in range(0, len(indices), global_batch_size)
            ]

        shuffled = torch.randperm(len(indices), generator=generator).tolist()
        indices = [indices[idx] for idx in shuffled]
        bucket_size = global_batch_size * self.bucket_size_multiplier
        batches = []
        for start in range(0, len(indices), bucket_size):
            bucket = indices[start:start + bucket_size]
            bucket.sort(key=lambda idx: self.lengths[idx])
            bucket = bucket[:len(bucket) - (len(bucket) % global_batch_size)]
            for batch_start in range(0, len(bucket), global_batch_size):
                batches.append(bucket[batch_start:batch_start + global_batch_size])

        if not batches:
            return []
        order = torch.randperm(len(batches), generator=generator).tolist()
        return [batches[idx] for idx in order]

    def __iter__(self):
        if self.next_consumed_samples is not None:
            self.consumed_samples = self.next_consumed_samples
            self.next_consumed_samples = None

        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        batch_offset = current_epoch_samples // self.micro_batch_times_data_parallel_size

        global_batches = self._epoch_global_batches(active_total_samples)
        rank_start = self.rank * self.micro_batch_size
        rank_end = rank_start + self.micro_batch_size
        for global_batch in global_batches[batch_offset:]:
            batch = global_batch[rank_start:rank_end]
            if len(batch) != self.micro_batch_size:
                continue
            self.consumed_samples += self.micro_batch_times_data_parallel_size
            yield batch
