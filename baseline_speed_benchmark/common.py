from __future__ import annotations

import csv
import datetime as _datetime
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_ROOT = BENCH_ROOT / "logs"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def infer_batch_size(batch: Any) -> int:
    if torch.is_tensor(batch):
        return int(batch.shape[0])
    if isinstance(batch, dict):
        for value in batch.values():
            try:
                return infer_batch_size(value)
            except ValueError:
                continue
    if isinstance(batch, (tuple, list)):
        for value in batch:
            try:
                return infer_batch_size(value)
            except ValueError:
                continue
    raise ValueError("Cannot infer batch size from batch.")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total), int(trainable)


def bytes_to_mib(value: int | None) -> float | None:
    return None if value is None else value / (1024.0 ** 2)


def make_log_dir(model_name: str, root: str | os.PathLike[str] | None = None) -> Path:
    log_root = Path(root) if root is not None else DEFAULT_LOG_ROOT
    now = _datetime.datetime.now()
    timestamp = "_".join("%02d" % value for value in (now.month, now.day, now.hour, now.minute))
    log_dir = log_root / f"{timestamp}_{model_name}_speed"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Log] -> {log_dir}")
    return log_dir


def measure_forward_speed(
    model: torch.nn.Module,
    loader: Any,
    forward_fn: Callable[[Any], Any],
    *,
    device: torch.device,
    warmup_batches: int,
    measure_batches: int,
) -> dict[str, Any]:
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model.eval().to(device)
    times: list[float] = []
    samples = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch = move_to_device(batch, device)
            if batch_index < warmup_batches:
                synchronize(device)
                forward_fn(batch)
                synchronize(device)
                continue

            if not times and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            batch_size = infer_batch_size(batch)
            synchronize(device)
            start = time.perf_counter()
            forward_fn(batch)
            synchronize(device)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            samples += batch_size

            if len(times) >= measure_batches:
                break

    if not times:
        raise RuntimeError("No timed batches were collected. Reduce --warmup-batches or check the dataset.")

    total_time = float(sum(times))
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    total_params, trainable_params = count_parameters(model)
    return {
        "timestamp": _datetime.datetime.now().isoformat(timespec="seconds"),
        "warmup_batches": int(warmup_batches),
        "measured_batches": len(times),
        "measured_samples": samples,
        "avg_per_batch_ms": total_time / len(times) * 1000.0,
        "avg_per_sample_ms": total_time / samples * 1000.0,
        "fps": samples / total_time,
        "param_count": total_params,
        "param_million": total_params / 1e6,
        "trainable_param_count": trainable_params,
        "trainable_param_million": trainable_params / 1e6,
        "peak_memory_allocated_mb": bytes_to_mib(peak_allocated),
        "peak_memory_reserved_mb": bytes_to_mib(peak_reserved),
    }


def profile_flops(
    model: torch.nn.Module,
    loader: Any,
    forward_fn: Callable[[Any], Any],
    *,
    device: torch.device,
) -> tuple[int | None, int | None, str]:
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception as exc:
        return None, None, f"torch.profiler unavailable: {exc}"

    try:
        batch = next(iter(loader))
    except StopIteration:
        return None, None, "dataloader is empty"

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    batch = move_to_device(batch, device)
    batch_size = infer_batch_size(batch)
    try:
        synchronize(device)
        with torch.inference_mode():
            with profile(activities=activities, record_shapes=True, with_flops=True) as prof:
                forward_fn(batch)
        synchronize(device)
        flops = sum(int(getattr(event, "flops", 0) or 0) for event in prof.key_averages())
        return flops, flops // max(batch_size, 1), "torch.profiler with_flops=True; unsupported ops are not counted"
    except Exception as exc:
        return None, None, f"FLOPs profiling failed: {exc}"


def write_metrics(log_dir: Path, name: str, metrics: dict[str, Any], extra: dict[str, Any]) -> tuple[Path, Path]:
    row = {**metrics, **extra}
    csv_path = log_dir / f"{name}_speed_metrics.csv"
    json_path = log_dir / f"{name}_speed_metrics.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow({key: "" if value is None else value for key, value in row.items()})
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2)
    return csv_path, json_path


def print_summary(name: str, metrics: dict[str, Any], csv_path: Path, flops_note: str | None = None) -> None:
    print(
        f"[Speed:{name}] avg per-sample: {metrics['avg_per_sample_ms']:.2f} ms | "
        f"FPS: {metrics['fps']:.2f} | samples={metrics['measured_samples']} | "
        f"batches={metrics['measured_batches']}"
    )
    print(f"[Speed:{name}] avg per-batch: {metrics['avg_per_batch_ms']:.2f} ms")
    print(
        f"[Params:{name}] total={metrics['param_count'] / 1e6:.2f}M | "
        f"trainable={metrics['trainable_param_count'] / 1e6:.2f}M"
    )
    if metrics.get("peak_memory_allocated_mb") is not None:
        print(
            f"[Memory:{name}] peak allocated: {metrics['peak_memory_allocated_mb']:.2f} MiB | "
            f"reserved/cache: {metrics['peak_memory_reserved_mb']:.2f} MiB"
        )
    if metrics.get("gflops_per_sample") is not None:
        print(
            f"[FLOPs:{name}] {metrics['gflops_per_sample']:.2f} GFLOPs per sample "
            f"({metrics['gflops']:.2f} GFLOPs per batch)"
        )
    elif flops_note:
        print(f"[FLOPs:{name}] unavailable ({flops_note})")
    print(f"[Speed:{name}] metrics csv -> {csv_path}")
