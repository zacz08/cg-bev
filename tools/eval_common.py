"""Shared utilities for BEV mask dumping and evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from skimage.transform import resize


CLASS_NAMES: Tuple[str, ...] = (
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "lane_divider",
)

BEV_SPAN_METERS = 100.0
TOKEN_RE = re.compile(r"(?:bev_(?:pred|gt|feat)_)([0-9a-fA-F]{32})")


@dataclass(frozen=True)
class MaskSample:
    """A prediction/ground-truth pair for one nuScenes sample token."""

    token: str
    pred: np.ndarray
    gt: np.ndarray
    source: str

    @property
    def resolution(self) -> int:
        """Return the square BEV mask resolution in pixels."""
        return int(self.gt.shape[-1])


@dataclass(frozen=True)
class ModelSource:
    """A named collection of samples from cached predictions or CG-BEV dumps."""

    name: str
    samples: Tuple[MaskSample, ...]
    failures: Tuple[str, ...] = ()

    @property
    def skipped(self) -> int:
        """Return the number of skipped samples recorded while loading."""
        return len(self.failures)


def baseline_key(name: str) -> str:
    """Return a filesystem-friendly lower-case key for a baseline name."""
    key = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    return key or "baseline"


def baseline_display_name(name: str) -> str:
    """Return a compact display name for common baseline keys."""
    key = baseline_key(name)
    common = {
        "lss": "LSS",
        "bevfusion": "BEVFusion",
        "bev_fusion": "BEVFusion",
        "bevformer": "BEVFormer",
        "hdmapnet": "HDMapNet",
        "vectormapnet": "VectorMapNet",
    }
    return common.get(key, name.strip() or "Baseline")


def default_prompt_path(data_root: Path, split: str, baseline: str) -> Path:
    """Return the default JSONL prompt path for a cached baseline source."""
    return data_root / f"prompt_{baseline_key(baseline)}_{split}.json"


def default_pred_root(data_root: Path, split: str, baseline: str) -> Path:
    """Return the default cached prediction directory for a baseline source."""
    return data_root / f"bev_pred_{baseline_key(baseline)}" / split


def default_gt_root(data_root: Path, split: str, resolution: int) -> Path:
    """Return the native-resolution GT cache directory."""
    return data_root / f"bev_seg_gt_mask_{resolution}" / split


def px_per_meter(resolution: int, bev_span_m: float = BEV_SPAN_METERS) -> float:
    """Return pixels per meter for a square BEV grid covering the configured span."""
    if resolution <= 0:
        raise ValueError(f"Resolution must be positive, got {resolution}")
    return float(resolution) / float(bev_span_m)


def source_resolution(source: ModelSource) -> int:
    """Return the common resolution for a source, or raise if it is mixed."""
    resolutions = {sample.resolution for sample in source.samples}
    if not resolutions:
        raise ValueError(f"Source {source.name} has no samples")
    if len(resolutions) != 1:
        raise ValueError(f"Source {source.name} has mixed resolutions: {sorted(resolutions)}")
    return resolutions.pop()


def extract_sample_token(path_or_name: str) -> str:
    """Extract the 32-character sample token from a cached BEV filename."""
    path = Path(path_or_name)
    match = TOKEN_RE.search(path.name)
    if match is None and re.fullmatch(r"[0-9a-fA-F]{32}", path.stem):
        return path.stem
    if match is None:
        raise ValueError(f"Could not extract sample token from: {path_or_name}")
    return match.group(1)


def load_jsonl(path: Path) -> List[Mapping[str, str]]:
    """Load a JSON-lines prompt file into a list of dictionaries."""
    rows: List[Mapping[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _binarize_chw_auto(chw: np.ndarray) -> np.ndarray:
    """Binarize a CHW mask/logit/probability tensor robustly.

    Cached BEV predictions are not stored in a single universal numeric format:
    - binary masks are often uint8/bool in {0, 1} or {0, 255};
    - diffusion-style tensors may use {-1, 1}, where zero is the natural threshold;
    - some baseline caches, such as BEVFusion exports, store probabilities in
      [0, 1], where using ``> 0`` would incorrectly mark almost every pixel as
      foreground.

    This helper therefore uses a 0.5 threshold only for floating-point arrays
    whose finite values are within [0, 1]. Other formats keep the previous
    sign/positive-mask convention. Integer {0, 1}/{0, 255} masks are unchanged
    by the positive-mask convention.
    """
    array = np.asarray(chw)
    if array.dtype == np.bool_:
        return array.astype(np.uint8, copy=False)

    finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.floating) else array
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.uint8)

    min_value = float(finite.min())
    max_value = float(finite.max())
    if np.issubdtype(array.dtype, np.floating) and min_value >= 0.0 and max_value <= 1.0:
        return (array >= 0.5).astype(np.uint8, copy=False)

    return (array > 0).astype(np.uint8, copy=False)


def ensure_chw_binary(mask: np.ndarray, class_count: Optional[int] = None) -> np.ndarray:
    """Convert a BEV mask/logit/probability array to uint8 CHW binary layout."""
    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {array.shape}")

    if array.shape[0] <= 16:
        chw = array
    elif array.shape[-1] <= 16:
        chw = np.moveaxis(array, -1, 0)
    else:
        raise ValueError(f"Cannot infer channel axis for mask shape {array.shape}")

    if class_count is not None:
        if chw.shape[0] < class_count:
            raise ValueError(f"Mask has {chw.shape[0]} channels, expected at least {class_count}")
        chw = chw[:class_count]

    return _binarize_chw_auto(chw)


def resize_chw_nearest(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize a CHW binary mask with nearest-neighbor interpolation."""
    out_h, out_w = size
    resized = resize(
        mask.astype(np.uint8, copy=False),
        output_shape=(mask.shape[0], out_h, out_w),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return (resized > 0).astype(np.uint8, copy=False)


def load_np_mask(path: Path, class_count: Optional[int] = None) -> np.ndarray:
    """Load a `.npy` mask file as CHW uint8 binary data."""
    return ensure_chw_binary(np.load(path), class_count=class_count)


def load_dump_mask(path: Path, class_count: Optional[int] = None) -> Tuple[str, np.ndarray, np.ndarray]:
    """Load a CG-BEV dump `.npz` file and return token, prediction, and GT masks."""
    with np.load(path) as data:
        pred = ensure_chw_binary(data["pred"], class_count=class_count)
        gt = ensure_chw_binary(data["gt"], class_count=class_count)
        if "sample_token" in data:
            token = str(np.asarray(data["sample_token"]).item())
        else:
            token = path.stem
    return token, pred, gt


def _join_root(root: Path, relative_or_absolute: str) -> Path:
    """Join root with a relative path while preserving absolute paths."""
    path = Path(relative_or_absolute)
    return path if path.is_absolute() else root / path


def _token_from_row(row: Mapping[str, str], pred_name: str, gt_name: str) -> str:
    """Infer a stable sample token from prompt metadata or cached filenames."""
    for key in ("sample_token", "token"):
        value = row.get(key)
        if value:
            return str(value)
    for candidate in (pred_name, gt_name):
        try:
            return extract_sample_token(candidate)
        except ValueError:
            continue
    raise ValueError(f"Could not infer sample token from row keys={sorted(row.keys())}")


def load_cached_samples(
    prompt_path: Path,
    data_root: Path,
    split: str,
    name: str,
    baseline: str,
    pred_root: Optional[Path] = None,
    gt_root: Optional[Path] = None,
    gt_resolution: int = 200,
    max_samples: Optional[int] = None,
    class_count: int = len(CLASS_NAMES),
    resize_pred_to_gt: bool = False,
) -> Tuple[Tuple[MaskSample, ...], Tuple[str, ...]]:
    """Load cached baseline predictions and matching native-resolution GT masks.

    The prediction directory defaults to `data_root/bev_pred_<baseline>/<split>`.
    This makes the loader usable for LSS, BEVFusion, or any other cached
    baseline whose prompt rows contain `pred_map` and `bev_map_gt` entries.
    """
    rows = load_jsonl(prompt_path)
    if max_samples is not None:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError(f"No rows found in {prompt_path}")

    pred_dir = pred_root or default_pred_root(data_root, split, baseline)
    gt_dir = gt_root or default_gt_root(data_root, split, gt_resolution)
    samples: List[MaskSample] = []
    failures: List[str] = []

    for row_index, row in enumerate(rows):
        try:
            pred_name = row["pred_map"]
            gt_name = row["bev_map_gt"]
            pred_path = _join_root(pred_dir, pred_name)
            gt_path = _join_root(gt_dir, gt_name)
            if not pred_path.exists():
                raise FileNotFoundError(f"missing prediction: {pred_path}")
            if not gt_path.exists():
                raise FileNotFoundError(f"missing GT: {gt_path}")

            pred = load_np_mask(pred_path, class_count=class_count)
            gt = load_np_mask(gt_path, class_count=class_count)
            if pred.shape[-2:] != gt.shape[-2:]:
                if not resize_pred_to_gt:
                    raise ValueError(f"shape mismatch pred={pred.shape}, gt={gt.shape}")
                pred = resize_chw_nearest(pred, size=gt.shape[-2:])

            token = _token_from_row(row, pred_name, gt_name)
            samples.append(MaskSample(token=token, pred=pred, gt=gt, source=name))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"row={row_index}: {exc}")

    if not samples:
        preview = "; ".join(failures[:3])
        raise ValueError(f"All {name} samples failed to load from {prompt_path}: {preview}")
    return tuple(samples), tuple(failures)


def load_dump_samples(
    dump_dir: Path,
    name: str,
    max_samples: Optional[int] = None,
    class_count: int = len(CLASS_NAMES),
) -> Tuple[Tuple[MaskSample, ...], Tuple[str, ...]]:
    """Load `.npz` files produced by `tools/dump_masks.py`."""
    files = sorted(dump_dir.glob("*.npz"))
    if max_samples is not None:
        files = files[:max_samples]
    if not files:
        raise ValueError(f"No `.npz` dump files found in {dump_dir}")

    samples: List[MaskSample] = []
    failures: List[str] = []
    for path in files:
        try:
            token, pred, gt = load_dump_mask(path, class_count=class_count)
            if pred.shape != gt.shape:
                raise ValueError(f"shape mismatch pred={pred.shape}, gt={gt.shape}")
            samples.append(MaskSample(token=token, pred=pred, gt=gt, source=name))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{path.name}: {exc}")

    if not samples:
        preview = "; ".join(failures[:3])
        raise ValueError(f"All dump files failed to load from {dump_dir}: {preview}")
    return tuple(samples), tuple(failures)


def parse_named_path(value: str) -> Tuple[str, Path]:
    """Parse a CLI `NAME=PATH` argument."""
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got: {value}")
    name, raw_path = value.split("=", 1)
    if not name:
        raise ValueError(f"Missing source name in: {value}")
    return name, Path(raw_path)


def compute_iou_counts(pred: np.ndarray, gt: np.ndarray, region: Optional[np.ndarray] = None) -> Tuple[int, int, int]:
    """Return TP, FP, and FN counts for binary masks, optionally restricted to a region."""
    pred_bool = pred.astype(bool, copy=False)
    gt_bool = gt.astype(bool, copy=False)
    if region is not None:
        pred_bool = np.logical_and(pred_bool, region)
        gt_bool = np.logical_and(gt_bool, region)
    true_positive = int(np.logical_and(pred_bool, gt_bool).sum())
    false_positive = int(np.logical_and(pred_bool, np.logical_not(gt_bool)).sum())
    false_negative = int(np.logical_and(np.logical_not(pred_bool), gt_bool).sum())
    return true_positive, false_positive, false_negative


def counts_to_iou(true_positive: int, false_positive: int, false_negative: int) -> float:
    """Convert TP/FP/FN counts into IoU, returning NaN for an empty union."""
    denom = true_positive + false_positive + false_negative
    if denom == 0:
        return float("nan")
    return true_positive / denom


def class_names(class_count: int) -> Tuple[str, ...]:
    """Return configured class names, extending with generic names when needed."""
    if class_count <= len(CLASS_NAMES):
        return CLASS_NAMES[:class_count]
    extra = tuple(f"class_{index}" for index in range(len(CLASS_NAMES), class_count))
    return CLASS_NAMES + extra


def mean_ignore_nan(values: Iterable[float]) -> float:
    """Return the arithmetic mean after dropping NaN values."""
    array = np.asarray(list(values), dtype=np.float64)
    valid = array[~np.isnan(array)]
    if valid.size == 0:
        return float("nan")
    return float(valid.mean())


def iter_sources(
    baseline_prompt: Optional[Path],
    data_root: Path,
    split: str,
    dumps: Sequence[str],
    max_samples: Optional[int],
    class_count: int,
    baseline: str,
    baseline_name: Optional[str] = None,
    baseline_pred_root: Optional[Path] = None,
    baseline_gt_root: Optional[Path] = None,
    baseline_gt_resolution: int = 200,
    include_baseline: bool = True,
) -> Iterator[ModelSource]:
    """Yield sources configured from one cached baseline and model dump dirs.

    The cached baseline defaults to:
      - prompt: ``data_root/prompt_<baseline>_<split>.json`` when the caller
        passes that path as ``baseline_prompt``;
      - predictions: ``data_root/bev_pred_<baseline>/<split>`` unless
        ``baseline_pred_root`` is provided;
      - GT: ``data_root/bev_seg_gt_mask_<resolution>/<split>`` unless
        ``baseline_gt_root`` is provided.
    """
    display_name = baseline_display_name(baseline) if baseline_name is None else baseline_name

    if include_baseline and baseline_prompt is not None:
        samples, failures = load_cached_samples(
            prompt_path=baseline_prompt,
            data_root=data_root,
            split=split,
            name=display_name,
            baseline=baseline,
            pred_root=baseline_pred_root,
            gt_root=baseline_gt_root,
            gt_resolution=baseline_gt_resolution,
            max_samples=max_samples,
            class_count=class_count,
        )
        yield ModelSource(name=display_name, samples=samples, failures=failures)

    for dump in dumps:
        name, dump_dir = parse_named_path(dump)
        samples, failures = load_dump_samples(
            dump_dir,
            name=name,
            max_samples=max_samples,
            class_count=class_count,
        )
        yield ModelSource(name=name, samples=samples, failures=failures)
