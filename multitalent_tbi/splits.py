from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .data import CaseRecord, discover_cases, make_stratification_labels


def _case_id_aliases(case_id: str) -> list[str]:
    aliases = {case_id}
    stripped = case_id.lstrip("0")
    if stripped:
        aliases.add(stripped)
    if case_id.isdigit():
        normalized = str(int(case_id))
        aliases.add(normalized)
        aliases.add(case_id.zfill(max(len(case_id), len(normalized))))
    digits = "".join(character for character in case_id if character.isdigit())
    if digits:
        aliases.add(digits)
        aliases.add(str(int(digits)))
        aliases.add(digits.lstrip("0") or "0")
    return sorted(aliases, key=len, reverse=True)


def build_splits(
    dataset_dir: str | Path,
    num_folds: int,
    seed: int,
    output_path: str | Path,
) -> list[dict[str, list[str]]]:
    records = discover_cases(dataset_dir)
    labels = make_stratification_labels(records)
    ids = [record.case_id for record in records]

    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits: list[dict[str, list[str]]] = []
    ids_array = np.asarray(ids)
    labels_array = np.asarray(labels)

    for train_index, val_index in splitter.split(ids_array, labels_array):
        splits.append(
            {
                "train": ids_array[train_index].tolist(),
                "val": ids_array[val_index].tolist(),
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"num_folds": num_folds, "seed": seed, "splits": splits}, handle, indent=2)
    return splits


def load_splits(path: str | Path) -> list[dict[str, list[str]]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["splits"]


def split_records(records: list[CaseRecord], split: dict[str, list[str]]) -> tuple[list[CaseRecord], list[CaseRecord]]:
    mapping: dict[str, CaseRecord] = {}
    for record in records:
        for alias in _case_id_aliases(record.case_id):
            mapping.setdefault(alias, record)

    def resolve(case_ids: Iterable[str]) -> list[CaseRecord]:
        resolved: list[CaseRecord] = []
        missing: list[str] = []
        for case_id in case_ids:
            record = mapping.get(case_id)
            if record is None:
                record = mapping.get(case_id.lstrip("0"))
            if record is None and case_id.isdigit():
                record = mapping.get(str(int(case_id)))
            if record is None:
                missing.append(case_id)
            else:
                resolved.append(record)
        if missing:
            available_preview = sorted({record.case_id for record in records})[:10]
            raise KeyError(
                f"Missing case ids in dataset: {missing[:10]}... "
                f"Available ids example: {available_preview}"
            )
        return resolved

    train_records = resolve(split["train"])
    val_records = resolve(split["val"])
    return train_records, val_records
