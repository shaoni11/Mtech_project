"""Train/validation/test split strategies for realistic generalization benchmarks.

Every split function returns ``(train_idx, val_idx, test_idx)`` as NumPy integer
arrays of positions into the input sequence.

- ``stratified_random_split``: per-label random split (optimistic baseline).
- ``group_split``: whole groups (scaffolds, targets, plates, weeks) never cross
  split boundaries.
- ``stratified_group_split``: group split done separately inside each label, so
  every class keeps held-out groups (e.g. unseen compounds per MoA class).
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Hashable, Sequence

import numpy as np


SplitIndices = tuple[np.ndarray, np.ndarray, np.ndarray]


def _to_arrays(train: list[int], val: list[int], test: list[int], rng: random.Random) -> SplitIndices:
    for part in (train, val, test):
        rng.shuffle(part)
    return (
        np.array(train, dtype=np.int64),
        np.array(val, dtype=np.int64),
        np.array(test, dtype=np.int64),
    )


def stratified_random_split(
    labels: Sequence[Hashable], test_size: float, val_size: float, seed: int
) -> SplitIndices:
    rng = random.Random(seed)
    by_label: dict[Hashable, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for indices in by_label.values():
        indices = indices[:]
        rng.shuffle(indices)
        n_total = len(indices)
        n_test = max(1, round(n_total * test_size))
        n_val = max(1, round(n_total * val_size))
        if n_total - n_test - n_val < 1:
            n_test = 1 if n_total >= 2 else 0
            n_val = 1 if n_total >= 3 else 0
        test.extend(indices[:n_test])
        val.extend(indices[n_test : n_test + n_val])
        train.extend(indices[n_test + n_val :])
    return _to_arrays(train, val, test, rng)


def group_split(
    groups: Sequence[Hashable],
    test_size: float,
    val_size: float,
    seed: int,
    balanced: bool = True,
) -> SplitIndices:
    """Split so that all rows of a group land in the same partition.

    With ``balanced=True`` (Chemprop-style balanced scaffold split), groups larger
    than half the validation or test budget are placed first so they end up in
    train, and the remaining groups are shuffled. With ``balanced=False`` all
    groups are shuffled, which suits few large groups such as targets or weeks.
    """
    index_by_group: dict[Hashable, list[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        index_by_group[group].append(idx)
    if len(index_by_group) < 3:
        raise ValueError(f"Group split needs at least 3 groups, found {len(index_by_group)}.")

    n_total = len(groups)
    n_test = round(n_total * test_size)
    n_val = round(n_total * val_size)
    n_train = n_total - n_test - n_val

    rng = random.Random(seed)
    group_sets = list(index_by_group.values())
    if balanced:
        big = [g for g in group_sets if len(g) > n_val / 2 or len(g) > n_test / 2]
        small = [g for g in group_sets if not (len(g) > n_val / 2 or len(g) > n_test / 2)]
        rng.shuffle(big)
        rng.shuffle(small)
        ordered = big + small
    else:
        ordered = group_sets[:]
        rng.shuffle(ordered)

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for group_indices in ordered:
        if len(train) + len(group_indices) <= n_train:
            train.extend(group_indices)
        elif n_val > 0 and n_val - len(val) >= n_test - len(test):
            # Send the group to whichever held-out split is furthest below its target size.
            val.extend(group_indices)
        else:
            test.extend(group_indices)

    if not train or not test or (val_size > 0 and not val):
        raise ValueError(
            "Group split produced an empty partition "
            f"(train={len(train)}, val={len(val)}, test={len(test)}). "
            "There are too few or too unevenly sized groups; try another seed or split."
        )
    return _to_arrays(train, val, test, rng)


def stratified_group_split(
    groups: Sequence[Hashable],
    labels: Sequence[Hashable],
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[SplitIndices, list[Hashable]]:
    """Hold out whole groups inside every label.

    Each group is assigned to its majority label. A label with k groups sends
    ``max(1, round(k * test_size))`` groups to test and, when at least two groups
    remain, ``max(1, round(k * val_size))`` to validation. Labels with a single
    group cannot hold out an unseen group, so their rows are split randomly;
    those labels are returned so callers can report them.
    """
    rng = random.Random(seed)
    label_votes: dict[Hashable, Counter] = defaultdict(Counter)
    index_by_group: dict[Hashable, list[int]] = defaultdict(list)
    for idx, (group, label) in enumerate(zip(groups, labels)):
        index_by_group[group].append(idx)
        label_votes[group][label] += 1

    groups_by_label: dict[Hashable, list[Hashable]] = defaultdict(list)
    for group in sorted(index_by_group, key=str):
        groups_by_label[label_votes[group].most_common(1)[0][0]].append(group)

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    row_split_labels: list[Hashable] = []
    for label in sorted(groups_by_label, key=str):
        label_groups = groups_by_label[label][:]
        rng.shuffle(label_groups)
        if len(label_groups) == 1:
            row_split_labels.append(label)
            indices = index_by_group[label_groups[0]]
            sub_train, sub_val, sub_test = stratified_random_split(
                [label] * len(indices), test_size, val_size, rng.randrange(1 << 30)
            )
            train.extend(indices[i] for i in sub_train)
            val.extend(indices[i] for i in sub_val)
            test.extend(indices[i] for i in sub_test)
            continue

        n_groups = len(label_groups)
        n_test = max(1, round(n_groups * test_size))
        n_val = max(1, round(n_groups * val_size)) if n_groups - n_test >= 2 else 0
        for position, group in enumerate(label_groups):
            if position < n_test:
                test.extend(index_by_group[group])
            elif position < n_test + n_val:
                val.extend(index_by_group[group])
            else:
                train.extend(index_by_group[group])

    return _to_arrays(train, val, test, rng), row_split_labels


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Bemis-Murcko scaffold SMILES; acyclic molecules share the empty scaffold."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Scaffold splits require RDKit: pip install rdkit") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"invalid:{smiles}"
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def scaffold_groups(smiles_list: Sequence[str]) -> list[str]:
    cache: dict[str, str] = {}
    scaffolds = []
    for smiles in smiles_list:
        if smiles not in cache:
            cache[smiles] = murcko_scaffold(smiles)
        scaffolds.append(cache[smiles])
    return scaffolds


def split_overlap(
    keys: Sequence[Hashable], train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray
) -> dict[str, int]:
    """Count distinct keys per split and how many test/val keys also occur in train."""
    train_keys = {keys[i] for i in train_idx.tolist()}
    val_keys = {keys[i] for i in val_idx.tolist()}
    test_keys = {keys[i] for i in test_idx.tolist()}
    return {
        "train_unique": len(train_keys),
        "val_unique": len(val_keys),
        "test_unique": len(test_keys),
        "val_seen_in_train": len(val_keys & train_keys),
        "test_seen_in_train": len(test_keys & train_keys),
    }
