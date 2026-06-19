from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import torch
from torch.utils.data import Dataset

from .transforms import to_unigaze_tensor


def read_manifest(manifest_path: str | Path) -> list[dict]:
    path = Path(manifest_path)
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("ok", True):
                records.append(record)
    return records


class CalibrationDataset(Dataset):
    def __init__(self, manifest_path: str | Path, records: Iterable[dict] | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        self.session_dir = self.manifest_path.parent
        self.records = list(records) if records is not None else read_manifest(self.manifest_path)
        self.records = [record for record in self.records if self._usable(record)]
        if not self.records:
            raise ValueError(f"no usable records in {self.manifest_path}")

    @staticmethod
    def _usable(record: dict) -> bool:
        required = [
            "normalized_face_path",
            "head_pose_pitch_yaw",
            "target_x_norm",
            "target_y_norm",
        ]
        return all(key in record for key in required)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        image_path = self.session_dir / record["normalized_face_path"]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return {
            "image": to_unigaze_tensor(image_rgb),
            "head_pose": torch.as_tensor(record["head_pose_pitch_yaw"], dtype=torch.float32),
            "target": torch.tensor(
                [record["target_x_norm"], record["target_y_norm"]],
                dtype=torch.float32,
            ),
            "viewport": torch.tensor(
                [
                    float(record.get("viewport_width", 1.0)),
                    float(record.get("viewport_height", 1.0)),
                ],
                dtype=torch.float32,
            ),
        }


def split_records(records: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(records), generator=generator).tolist()
    val_count = max(1, int(round(len(records) * val_ratio))) if len(records) > 1 else 0
    val_indexes = set(order[:val_count])
    train = [record for index, record in enumerate(records) if index not in val_indexes]
    val = [record for index, record in enumerate(records) if index in val_indexes]
    if not train:
        return records, []
    return train, val
