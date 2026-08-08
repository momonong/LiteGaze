"""Frozen final-fit and inference helpers for EyePoseTinyCNN-v1."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from scripts.gaze_diversity.train import (
    _loader,
    _normalize_images,
    _predict,
    configure_torch_runtime,
)


def train_or_load_final_candidate(
    *,
    train_images: np.ndarray,
    train_poses: np.ndarray,
    train_targets: np.ndarray,
    seed: int,
    config: dict[str, Any],
    checkpoint_path: Path,
    integrity: dict[str, str],
    resource_checkpoint: Callable[[int, str], None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit exactly 11 epochs or load an integrity-bound completed checkpoint."""
    torch = configure_torch_runtime(
        seed=seed,
        cpu_threads=int(config["cpu_threads"]),
        memory_fraction=float(config["memory_fraction"]),
    )
    if checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _validate_checkpoint(payload, seed=seed, integrity=integrity)
        summary = dict(payload["summary"])
        summary["resumed"] = True
        return payload, summary

    from torch import nn

    from scripts.gaze_diversity.model import EyePoseTinyCNN, count_parameters

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    model = EyePoseTinyCNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loss_function = nn.SmoothL1Loss(beta=float(config["loss_beta_radians"]))
    generator = torch.Generator().manual_seed(seed)
    loader = _loader(
        torch,
        train_images,
        train_poses,
        train_targets,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    history: list[dict[str, float | int]] = []
    started = time.monotonic()
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for images, poses, targets in loader:
            images = _normalize_images(images.to(device, non_blocking=True))
            poses = poses.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            prediction = model(images, poses)
            loss = loss_function(prediction, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_norm_clip"])
            )
            optimizer.step()
            rows = int(images.shape[0])
            total_loss += float(loss.detach().cpu()) * rows
            total_rows += rows
        epoch_loss = total_loss / max(total_rows, 1)
        history.append({"epoch": epoch, "train_loss": epoch_loss})
        peak = int(torch.cuda.max_memory_reserved(device))
        resource_checkpoint(peak, f"candidate-seed-{seed}-epoch-{epoch}")
        print(
            f"[candidate-seed-{seed}] epoch={epoch:02d} loss={epoch_loss:.6f}",
            flush=True,
        )

    state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    peak = int(torch.cuda.max_memory_reserved(device))
    summary = {
        "seed": int(seed),
        "parameter_count": count_parameters(model),
        "epochs_completed": len(history),
        "history": history,
        "duration_seconds": time.monotonic() - started,
        "peak_process_memory_bytes": peak,
        "state_dict_sha256": state_dict_sha256(state),
        "resumed": False,
    }
    payload = {
        "schema_version": 1,
        "kind": "EyePoseTinyCNN-v1-final-fit",
        "seed": int(seed),
        "integrity": dict(integrity),
        "summary": summary,
        "state_dict": state,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint_path)
    del model, optimizer, loader
    torch.cuda.empty_cache()
    return payload, summary


def predict_candidate(
    *,
    checkpoint_payloads: list[dict[str, Any]],
    eye_images: np.ndarray,
    eye_poses: np.ndarray,
    batch_size: int,
    resource_checkpoint: Callable[[int, str], None],
) -> np.ndarray:
    """Predict both eyes for every seed and undo the frozen right-eye yaw flip."""
    import torch

    from scripts.gaze_diversity.model import EyePoseTinyCNN

    images = np.asarray(eye_images)
    poses = np.asarray(eye_poses, dtype=np.float32)
    if images.ndim != 5 or images.shape[1:] != (2, 1, 36, 60):
        raise ValueError("eye_images must have shape (N,2,1,36,60)")
    if poses.shape != (len(images), 2, 2):
        raise ValueError("eye_poses must have shape (N,2,2)")
    flat_images = np.ascontiguousarray(images.reshape(-1, 1, 36, 60))
    flat_poses = np.ascontiguousarray(poses.reshape(-1, 2))
    loader = _loader(
        torch,
        flat_images,
        flat_poses,
        None,
        batch_size=batch_size,
        shuffle=False,
        generator=None,
    )
    device = torch.device("cuda:0")
    members: list[np.ndarray] = []
    for payload in checkpoint_payloads:
        model = EyePoseTinyCNN().to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        prediction = _predict(model, loader, device, torch).reshape(len(images), 2, 2)
        prediction[:, 1, 1] *= -1.0
        members.extend((prediction[:, 0, :], prediction[:, 1, :]))
        peak = int(torch.cuda.max_memory_reserved(device))
        resource_checkpoint(peak, f"candidate-seed-{payload['seed']}-inference")
        del model
        torch.cuda.empty_cache()
    return np.stack(members, axis=0)


def state_dict_sha256(state: dict[str, Any]) -> str:
    """Hash names, dtypes, shapes, and exact contiguous tensor bytes."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_checkpoint(
    payload: dict[str, Any],
    *,
    seed: int,
    integrity: dict[str, str],
) -> None:
    if payload.get("kind") != "EyePoseTinyCNN-v1-final-fit":
        raise ValueError("unexpected candidate checkpoint kind")
    if int(payload.get("seed", -1)) != seed:
        raise ValueError("candidate checkpoint seed mismatch")
    if payload.get("integrity") != integrity:
        raise ValueError("candidate checkpoint integrity mismatch")
    summary = payload.get("summary", {})
    state = payload.get("state_dict", {})
    if int(summary.get("epochs_completed", -1)) != 11:
        raise ValueError("candidate checkpoint epoch mismatch")
    if state_dict_sha256(state) != summary.get("state_dict_sha256"):
        raise ValueError("candidate checkpoint state hash mismatch")
