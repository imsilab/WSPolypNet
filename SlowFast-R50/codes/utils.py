from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


CSV_COLUMNS = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "corloc_0.3",
    "corloc_0.5",
    "corloc_0.7",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pth")
    torch.save(state, temporary)
    os.replace(temporary, path)


def update_csv(path: Path, metrics: dict[str, float | int]) -> None:
    """Atomically insert/replace one epoch so interruption never erases prior rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[int, dict[str, str]] = {}
    if path.exists():
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
                raise RuntimeError(
                    f"Unexpected CSV columns in {path}: {reader.fieldnames}"
                )
            for row in reader:
                rows[int(row["epoch"])] = row
    epoch = int(metrics["epoch"])
    rows[epoch] = {column: str(metrics[column]) for column in CSV_COLUMNS}

    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row_epoch in sorted(rows):
            writer.writerow(rows[row_epoch])
    os.replace(temporary, path)
