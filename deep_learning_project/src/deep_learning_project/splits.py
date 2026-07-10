"""Dataset split helpers."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence


def stratified_indices(
    labels: Sequence[int],
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[int(label)].append(index)

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []

    for indices in by_label.values():
        rng.shuffle(indices)
        n_total = len(indices)
        n_test = max(1, round(n_total * test_size))
        n_val = max(1, round(n_total * val_size))
        test.extend(indices[:n_test])
        val.extend(indices[n_test : n_test + n_val])
        train.extend(indices[n_test + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def random_row_splits(
    n_rows: int,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    indices = list(range(n_rows))
    random.Random(seed).shuffle(indices)
    n_test = max(1, round(n_rows * test_size))
    n_val = max(1, round(n_rows * val_size))
    test = indices[:n_test]
    val = indices[n_test : n_test + n_val]
    train = indices[n_test + n_val :]
    return train, val, test

