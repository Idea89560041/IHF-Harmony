from __future__ import annotations

import itertools
import random
from collections.abc import Iterator

from torch.utils.data import Sampler


class InfiniteRandomSampler(Sampler[int]):
    """Simple infinite sampler kept for experiments that prefer step-based loops."""

    def __init__(self, dataset_size: int, seed: int = 0) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        self.dataset_size = dataset_size
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        while True:
            indices = list(range(self.dataset_size))
            rng.shuffle(indices)
            yield from indices

    def __len__(self) -> int:
        return self.dataset_size


class SequentialShardSampler(Sampler[int]):
    """Shard evaluation samples across distributed ranks without padding."""

    def __init__(self, dataset_size: int, world_size: int, rank: int) -> None:
        self.indices = list(itertools.islice(range(dataset_size), rank, None, world_size))

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
