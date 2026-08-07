"""Frozen training loop for EyePoseTinyCNN-v1."""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from typing import Any

import numpy as np


def configure_torch_runtime(
    *,
    seed: int,
    cpu_threads: int,
    memory_fraction: float,
) -> Any:
    """Configure deterministic FP32 CUDA execution and return torch."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen experiment protocol")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
    return torch


def train_one_fold(
    *,
    train_images: np.ndarray,
    train_poses: np.ndarray,
    train_targets: np.ndarray,
    validation_images: np.ndarray,
    validation_poses: np.ndarray,
    validation_targets: np.ndarray,
    test_images: np.ndarray,
    test_poses: np.ndarray,
    seed: int,
    config: dict[str, Any],
    resource_checkpoint: Callable[[int, str], None],
) -> dict[str, Any]:
    """Train on 13 subjects, select on one, and test exactly once on one."""
    torch = configure_torch_runtime(
        seed=seed,
        cpu_threads=int(config["cpu_threads"]),
        memory_fraction=float(config["memory_fraction"]),
    )
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

    train_loader = _loader(
        torch,
        train_images,
        train_poses,
        train_targets,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    validation_loader = _loader(
        torch,
        validation_images,
        validation_poses,
        validation_targets,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        generator=None,
    )
    test_loader = _loader(
        torch,
        test_images,
        test_poses,
        None,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        generator=None,
    )

    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for images, poses, targets in train_loader:
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

        validation_predictions = _predict(model, validation_loader, device, torch)
        validation_error = _mean_angular_error_degrees(
            validation_predictions,
            validation_targets,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_rows, 1),
                "validation_mean_degrees": validation_error,
            }
        )
        print(
            f"[{config['job_label']}] epoch={epoch:02d} "
            f"loss={history[-1]['train_loss']:.6f} "
            f"val_deg={validation_error:.4f}",
            flush=True,
        )
        process_peak = int(torch.cuda.max_memory_reserved(device))
        resource_checkpoint(process_peak, f"{config['job_label']}-epoch-{epoch}")

        minimum_delta = float(config["early_stopping_min_delta_degrees"])
        if validation_error < best_validation - minimum_delta:
            best_validation = validation_error
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config["early_stopping_patience"]):
                break

    if best_state is None:
        raise RuntimeError("training never produced a validation-selected checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    test_predictions = _predict(model, test_loader, device, torch)
    process_peak = int(torch.cuda.max_memory_reserved(device))
    resource_checkpoint(process_peak, f"{config['job_label']}-test-complete")
    result = {
        "seed": int(seed),
        "parameter_count": count_parameters(model),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_mean_degrees": best_validation,
        "history": history,
        "test_predictions": test_predictions,
        "peak_process_memory_bytes": process_peak,
    }
    del model, optimizer, train_loader, validation_loader, test_loader
    torch.cuda.empty_cache()
    return result


def _loader(
    torch: Any,
    images: np.ndarray,
    poses: np.ndarray,
    targets: np.ndarray | None,
    *,
    batch_size: int,
    shuffle: bool,
    generator: Any,
) -> Any:
    from torch.utils.data import DataLoader, TensorDataset

    image_tensor = torch.from_numpy(np.ascontiguousarray(images))
    pose_tensor = torch.from_numpy(np.ascontiguousarray(poses, dtype=np.float32))
    tensors = [image_tensor, pose_tensor]
    if targets is not None:
        tensors.append(
            torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32))
        )
    dataset = TensorDataset(*tensors)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        generator=generator,
    )


def _normalize_images(images: Any) -> Any:
    return images.float().div_(255.0).sub_(0.5).div_(0.25)


def _predict(model: Any, loader: Any, device: Any, torch: Any) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            images, poses = batch[:2]
            images = _normalize_images(images.to(device, non_blocking=True))
            poses = poses.to(device, non_blocking=True)
            predictions.append(model(images, poses).cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float64, copy=False)


def _mean_angular_error_degrees(
    prediction_angles: np.ndarray, target_angles: np.ndarray
) -> float:
    from scripts.gaze_diversity.metrics import angular_errors_degrees

    return float(np.mean(angular_errors_degrees(prediction_angles, target_angles)))
