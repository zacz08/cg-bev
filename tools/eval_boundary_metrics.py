"""Boundary-aware BEV segmentation metrics for CG-BEV supplemental experiments."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import binary_erosion, distance_transform_edt  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402
from tqdm import tqdm  # noqa: E402

from tools.eval_common import (  # noqa: E402
    CLASS_NAMES,
    MaskSample,
    ModelSource,
    class_names,
    compute_iou_counts,
    counts_to_iou,
    default_pred_root,
    ensure_chw_binary,
    extract_sample_token,
    iter_sources,
    load_jsonl,
    mean_ignore_nan,
    px_per_meter,
    source_resolution,
)

np.random.seed(42)

BOUNDARY_TOLERANCES_M: Tuple[float, ...] = (0.5, 1.0, 2.0)
LANE_CURVE_TOLERANCES_M: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
CHAMFER_CLASSES = set(CLASS_NAMES)
CHAMFER_TRUNCATE_M: float = 5.0
# The CD implementation sums two truncated one-way distances: mean(P->G) + mean(G->P).
# With a 5m truncation per direction, the natural maximum penalty for a
# one-sided empty skeleton is therefore 10m.
CHAMFER_EMPTY_PENALTY_M: float = 2.0 * CHAMFER_TRUNCATE_M
BEVFUSION_NATIVE_CLASSES: Tuple[str, ...] = (
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "divider",
)


@dataclass
class BoundaryTotals:
    """Accumulated boundary precision/recall counts."""

    pred_match: int = 0
    pred_count: int = 0
    gt_match: int = 0
    gt_count: int = 0
    empty_pairs: int = 0

    def update(self, pred_match: int, pred_count: int, gt_match: int, gt_count: int) -> None:
        if pred_count == 0 and gt_count == 0:
            self.empty_pairs += 1
            return
        self.pred_match += pred_match
        self.pred_count += pred_count
        self.gt_match += gt_match
        self.gt_count += gt_count

    def precision(self) -> float:
        if self.pred_count == 0:
            return float("nan") if self.gt_count == 0 else 0.0
        return self.pred_match / self.pred_count

    def recall(self) -> float:
        if self.gt_count == 0:
            return float("nan") if self.pred_count == 0 else 0.0
        return self.gt_match / self.gt_count

    def f1(self) -> float:
        precision = self.precision()
        recall = self.recall()
        if np.isnan(precision) and np.isnan(recall):
            return float("nan")
        if np.isnan(precision):
            precision = 0.0
        if np.isnan(recall):
            recall = 0.0
        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compute boundary-aware BEV segmentation metrics.")
    parser.add_argument("--data-root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--baseline",
        default="lss",
        help="Cached baseline source to load through eval_common (e.g. lss or bevfusion).",
    )
    parser.add_argument(
        "--baseline-name",
        default=None,
        help="Display name for the cached baseline. Defaults to a normalized form of --baseline.",
    )
    parser.add_argument(
        "--baseline-prompt",
        type=Path,
        default=None,
        help=(
            "Prompt file for the cached baseline. If omitted, defaults to "
            "data/nuscenes/prompt_<baseline>_<split>.json."
        ),
    )
    parser.add_argument(
        "--baseline-pred-root",
        type=Path,
        default=None,
        help="Optional cached baseline prediction root.",
    )
    parser.add_argument(
        "--baseline-gt-root",
        type=Path,
        default=None,
        help="Optional cached baseline GT root.",
    )
    parser.add_argument(
        "--baseline-gt-resolution",
        type=int,
        default=200,
        help="Native GT resolution for the cached baseline.",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Do not include the cached baseline source.",
    )
    parser.add_argument("--dump", action="append", default=[], metavar="NAME=DIR", help="Model dump directory.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--class-count", type=int, default=len(CLASS_NAMES))
    parser.add_argument("--out-dir", type=Path, default=Path("logs/exp1"))
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--skip-csv", type=Path, default=None)
    parser.add_argument("--curve-png", type=Path, default=None)
    return parser.parse_args()


def baseline_key(baseline: str) -> str:
    """Return a filesystem-friendly baseline key."""
    return baseline.strip().lower().replace("-", "_").replace(" ", "_")


def baseline_display_name(baseline: str, explicit_name: Optional[str] = None) -> str:
    """Return a readable baseline label used in tables, plots, and CSV rows."""
    if explicit_name:
        return explicit_name
    key = baseline_key(baseline)
    common_names = {
        "lss": "LSS",
        "bevfusion": "BEVFusion",
        "bev_fusion": "BEVFusion",
        "bevformer": "BEVFormer",
        "hdmapnet": "HDMapNet",
        "vectormapnet": "VectorMapNet",
    }
    return common_names.get(key, baseline.strip())


def resolve_baseline_prompt(args: argparse.Namespace) -> Path:
    """Resolve the prompt path for the cached baseline source."""
    if args.baseline_prompt is not None:
        return args.baseline_prompt
    return args.data_root / f"prompt_{baseline_key(args.baseline)}_{args.split}.json"


def is_bevfusion_baseline(baseline: str) -> bool:
    """Return whether a baseline key should use the BEVFusion-native loader."""
    return baseline_key(baseline) in {"bevfusion", "bev_fusion"}


def nuscenes_version_for_split(split: str) -> str:
    """Return the nuScenes version needed to regenerate map GT for a split."""
    return "v1.0-mini" if split.startswith("mini_") else "v1.0-trainval"


def patch_nuscenes_mask_for_lines() -> None:
    """Patch nuscenes-devkit line rasterization for Shapely 2.x."""
    import cv2  # noqa: PLC0415
    from nuscenes.map_expansion.map_api import NuScenesMapExplorer  # noqa: PLC0415

    def _mask_for_lines_shapely2(lines: object, mask: np.ndarray) -> np.ndarray:
        line_iter = lines.geoms if getattr(lines, "geom_type", None) == "MultiLineString" else [lines]
        for line in line_iter:
            coords = np.asarray(list(line.coords), np.int32).reshape((-1, 2))
            cv2.polylines(mask, [coords], False, 1, 2)
        return mask

    NuScenesMapExplorer.mask_for_lines = staticmethod(_mask_for_lines_shapely2)


def transform_matrix(translation: Sequence[float], rotation: Sequence[float]) -> np.ndarray:
    """Return a 4x4 transform matrix from a nuScenes translation/quaternion pair."""
    from pyquaternion import Quaternion  # noqa: PLC0415

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Quaternion(rotation).rotation_matrix
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def bevfusion_layer_mappings(class_count: int) -> Dict[str, List[str]]:
    """Return BEVFusion map-layer mappings for the requested class count."""
    if class_count > len(BEVFUSION_NATIVE_CLASSES):
        raise ValueError(
            f"BEVFusion-native baseline supports at most {len(BEVFUSION_NATIVE_CLASSES)} classes, "
            f"got class_count={class_count}"
        )
    mappings: Dict[str, List[str]] = {}
    for name in BEVFUSION_NATIVE_CLASSES[:class_count]:
        if name == "divider":
            mappings[name] = ["road_divider", "lane_divider"]
        else:
            mappings[name] = [name]
    return mappings


def bevfusion_native_gt_saved_coordinates(
    token: str,
    resolution: int,
    nusc: object,
    scene_to_location: Mapping[str, str],
    maps: Mapping[str, object],
    class_count: int,
) -> np.ndarray:
    """Generate BEVFusion-native GT in the coordinate frame of saved predictions.

    BEVFusion evaluates ``masks_bev`` against ``gt_masks_bev`` in LiDAR-frame
    coordinates. The cache created by ``third_party/bevfusion/mmdet3d/apis/test.py``
    then saves predictions after ``np.rot90(preds, k=1, axes=(1, 2))``. This
    function regenerates the BEVFusion GT and applies the same rotation so the
    cached predictions can be evaluated without modifying the cache.
    """
    sample = nusc.get("sample", token)
    sample_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    calibrated_sensor = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])

    lidar2ego = transform_matrix(calibrated_sensor["translation"], calibrated_sensor["rotation"])
    ego2global = transform_matrix(ego_pose["translation"], ego_pose["rotation"])
    lidar2global = ego2global @ lidar2ego

    map_pose = lidar2global[:2, 3]
    heading = lidar2global[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    patch_angle = float(np.degrees(np.arctan2(heading[1], heading[0])))
    patch_box = (float(map_pose[0]), float(map_pose[1]), 100.0, 100.0)

    mappings = bevfusion_layer_mappings(class_count)
    layer_names: List[str] = []
    for mapped_layers in mappings.values():
        layer_names.extend(mapped_layers)
    layer_names = list(dict.fromkeys(layer_names))

    location = scene_to_location[sample["scene_token"]]
    map_masks = maps[location].get_map_mask(
        patch_box=patch_box,
        patch_angle=patch_angle,
        layer_names=layer_names,
        canvas_size=(resolution, resolution),
    )
    map_masks = map_masks.transpose(0, 2, 1).astype(bool, copy=False)

    labels = np.zeros((class_count, resolution, resolution), dtype=np.uint8)
    for class_index, (_name, mapped_layers) in enumerate(mappings.items()):
        for layer_name in mapped_layers:
            labels[class_index, map_masks[layer_names.index(layer_name)]] = 1

    return np.rot90(labels, k=1, axes=(1, 2)).astype(np.uint8, copy=False)


def load_bevfusion_native_samples(
    prompt_path: Path,
    data_root: Path,
    split: str,
    name: str,
    pred_root: Optional[Path],
    gt_resolution: int,
    max_samples: Optional[int],
    class_count: int,
) -> Tuple[Tuple[MaskSample, ...], Tuple[str, ...]]:
    """Load BEVFusion cached predictions with GT regenerated by its native protocol."""
    patch_nuscenes_mask_for_lines()

    from nuscenes.nuscenes import NuScenes  # noqa: PLC0415
    from nuscenes.map_expansion.map_api import NuScenesMap, locations as LOCATIONS  # noqa: PLC0415

    rows = load_jsonl(prompt_path)
    if max_samples is not None:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError(f"No rows found in {prompt_path}")

    pred_dir = pred_root or default_pred_root(data_root, split, "bevfusion")
    nusc = NuScenes(version=nuscenes_version_for_split(split), dataroot=str(data_root), verbose=False)
    scene_to_location = {
        scene["token"]: nusc.get("log", scene["log_token"])["location"]
        for scene in nusc.scene
    }
    maps = {location: NuScenesMap(str(data_root), location) for location in LOCATIONS}

    samples: List[MaskSample] = []
    failures: List[str] = []
    for row_index, row in enumerate(tqdm(rows, desc=f"BEVFusion-native GT: {name}")):
        try:
            pred_name = row["pred_map"]
            pred_path = pred_dir / pred_name if not Path(pred_name).is_absolute() else Path(pred_name)
            if not pred_path.exists():
                raise FileNotFoundError(f"missing prediction: {pred_path}")

            token = str(row.get("sample_token") or row.get("token") or extract_sample_token(pred_name))
            pred = ensure_chw_binary(np.load(pred_path), class_count=class_count)
            gt = bevfusion_native_gt_saved_coordinates(
                token=token,
                resolution=gt_resolution,
                nusc=nusc,
                scene_to_location=scene_to_location,
                maps=maps,
                class_count=class_count,
            )
            if pred.shape != gt.shape:
                raise ValueError(f"shape mismatch pred={pred.shape}, gt={gt.shape}")
            samples.append(MaskSample(token=token, pred=pred, gt=gt, source=name))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"row={row_index}: {exc}")

    if not samples:
        preview = "; ".join(failures[:3])
        raise ValueError(f"All BEVFusion-native samples failed to load from {prompt_path}: {preview}")
    return tuple(samples), tuple(failures)


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Extract a one-pixel binary boundary from a mask."""
    mask_bool = mask.astype(bool, copy=False)
    if not mask_bool.any():
        return np.zeros_like(mask_bool, dtype=bool)
    eroded = binary_erosion(mask_bool, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return np.logical_and(mask_bool, np.logical_not(eroded))


def boundary_distance_maps(
    pred_boundary: np.ndarray,
    gt_boundary: np.ndarray,
) -> Tuple[int, int, Optional[np.ndarray], Optional[np.ndarray]]:
    """Precompute boundary counts and distance maps shared by all tolerances.

    dist_to_gt[y, x] is the Euclidean pixel distance from (y, x) to the
    nearest GT boundary pixel. dist_to_pred is defined symmetrically. The
    distance maps are needed only when both prediction and GT boundaries
    are non-empty; otherwise all possible matches are zero for every
    tolerance, so computing distance transforms is unnecessary.
    """
    pred_count = int(pred_boundary.sum())
    gt_count = int(gt_boundary.sum())

    if pred_count == 0 or gt_count == 0:
        return pred_count, gt_count, None, None

    dist_to_gt = distance_transform_edt(np.logical_not(gt_boundary))
    dist_to_pred = distance_transform_edt(np.logical_not(pred_boundary))
    return pred_count, gt_count, dist_to_gt, dist_to_pred


def boundary_match_counts_from_distances(
    pred_boundary: np.ndarray,
    gt_boundary: np.ndarray,
    pred_count: int,
    gt_count: int,
    dist_to_gt: Optional[np.ndarray],
    dist_to_pred: Optional[np.ndarray],
    tolerance_px: float,
) -> Tuple[int, int, int, int]:
    """Return boundary match/count pixels using precomputed distance maps."""
    if pred_count == 0 and gt_count == 0:
        return 0, 0, 0, 0

    pred_match = 0
    gt_match = 0
    if pred_count > 0 and gt_count > 0:
        if dist_to_gt is None or dist_to_pred is None:
            raise ValueError("Distance maps must be provided when both boundaries are non-empty.")
        pred_match = int((dist_to_gt[pred_boundary] <= tolerance_px).sum())
        gt_match = int((dist_to_pred[gt_boundary] <= tolerance_px).sum())

    return pred_match, pred_count, gt_match, gt_count


def boundary_match_counts(pred_boundary: np.ndarray, gt_boundary: np.ndarray, tolerance_px: float) -> Tuple[int, int, int, int]:
    """Return matched/count pixels for boundary precision and recall.

    This compatibility wrapper computes the distance maps once and evaluates
    a single tolerance. In evaluate_source, distance maps are reused across
    all tolerances through boundary_match_counts_from_distances().
    """
    pred_count, gt_count, dist_to_gt, dist_to_pred = boundary_distance_maps(pred_boundary, gt_boundary)
    return boundary_match_counts_from_distances(
        pred_boundary,
        gt_boundary,
        pred_count,
        gt_count,
        dist_to_gt,
        dist_to_pred,
        tolerance_px,
    )


def chamfer_distance_m_with_status(
    pred: np.ndarray,
    gt: np.ndarray,
    resolution: int,
    truncate_m: float = CHAMFER_TRUNCATE_M,
    empty_penalty_m: float = CHAMFER_EMPTY_PENALTY_M,
) -> Tuple[float, str]:
    """Compute skeleton CD and report how empty skeletons were handled.

    The returned status is one of:
      - "valid": both skeletons are non-empty and CD was computed normally.
      - "both_empty": both skeletons are empty; there is no geometry to compare, so CD is NaN.
      - "single_empty_penalty": exactly one skeleton is empty; this is treated as a
        complete miss/false positive and assigned empty_penalty_m.

    CD follows the convention used in this script: mean(P->G) + mean(G->P), with
    each one-way distance truncated at truncate_m before averaging.
    """
    pred_skeleton = skeletonize(pred.astype(bool, copy=False))
    gt_skeleton = skeletonize(gt.astype(bool, copy=False))
    pred_points = np.column_stack(np.nonzero(pred_skeleton))
    gt_points = np.column_stack(np.nonzero(gt_skeleton))

    pred_empty = pred_points.size == 0
    gt_empty = gt_points.size == 0
    if pred_empty and gt_empty:
        return float("nan"), "both_empty"
    if pred_empty or gt_empty:
        return float(empty_penalty_m), "single_empty_penalty"

    pixels_per_meter = px_per_meter(resolution)
    truncate_px = truncate_m * pixels_per_meter
    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)
    pred_to_gt = gt_tree.query(pred_points, k=1)[0]
    gt_to_pred = pred_tree.query(gt_points, k=1)[0]
    cd_px = np.minimum(pred_to_gt, truncate_px).mean() + np.minimum(gt_to_pred, truncate_px).mean()
    return float(cd_px / pixels_per_meter), "valid"


def chamfer_distance_m(
    pred: np.ndarray,
    gt: np.ndarray,
    resolution: int,
    truncate_m: float = CHAMFER_TRUNCATE_M,
    empty_penalty_m: float = CHAMFER_EMPTY_PENALTY_M,
) -> float:
    """Compute skeleton CD = mean(P->G) + mean(G->P) in meters with truncation.

    Both-empty skeleton pairs return NaN. One-sided empty skeleton pairs return
    empty_penalty_m, whose default is 10m for the 5m-per-direction truncation.
    """
    cd_m, _ = chamfer_distance_m_with_status(
        pred,
        gt,
        resolution,
        truncate_m=truncate_m,
        empty_penalty_m=empty_penalty_m,
    )
    return cd_m


def evaluate_source(source: ModelSource, class_count: int) -> Tuple[List[Mapping[str, object]], Dict[float, float]]:
    """Evaluate all boundary metrics for one source."""
    names = class_names(class_count)
    resolution = source_resolution(source)
    iou_counts = [[0, 0, 0] for _ in range(class_count)]
    boundary_totals = {
        tolerance: [BoundaryTotals() for _ in range(class_count)]
        for tolerance in set(BOUNDARY_TOLERANCES_M + LANE_CURVE_TOLERANCES_M)
    }
    chamfer_values: Dict[int, List[float]] = {index: [] for index, name in enumerate(names) if name in CHAMFER_CLASSES}
    chamfer_skipped: Dict[int, int] = {index: 0 for index in chamfer_values}
    chamfer_penalized: Dict[int, int] = {index: 0 for index in chamfer_values}

    for sample in tqdm(source.samples, desc=f"Boundary metrics: {source.name}"):
        sample_resolution = sample.resolution
        for class_index, class_name in enumerate(names):
            pred = sample.pred[class_index].astype(bool, copy=False)
            gt = sample.gt[class_index].astype(bool, copy=False)
            tp, fp, fn = compute_iou_counts(pred, gt)
            iou_counts[class_index][0] += tp
            iou_counts[class_index][1] += fp
            iou_counts[class_index][2] += fn

            pred_boundary = boundary_mask(pred)
            gt_boundary = boundary_mask(gt)
            pred_count, gt_count, dist_to_gt, dist_to_pred = boundary_distance_maps(pred_boundary, gt_boundary)
            for tolerance_m, totals_by_class in boundary_totals.items():
                tolerance_px = tolerance_m * px_per_meter(sample_resolution)
                totals_by_class[class_index].update(
                    *boundary_match_counts_from_distances(
                        pred_boundary,
                        gt_boundary,
                        pred_count,
                        gt_count,
                        dist_to_gt,
                        dist_to_pred,
                        tolerance_px,
                    )
                )

            if class_name in CHAMFER_CLASSES:
                cd_m, cd_status = chamfer_distance_m_with_status(pred, gt, sample_resolution)
                if cd_status == "both_empty":
                    chamfer_skipped[class_index] += 1
                else:
                    if cd_status == "single_empty_penalty":
                        chamfer_penalized[class_index] += 1
                    chamfer_values[class_index].append(cd_m)

    rows: List[Mapping[str, object]] = []
    mean_iou: List[float] = []
    mean_bf: Dict[float, List[float]] = {tolerance: [] for tolerance in BOUNDARY_TOLERANCES_M}
    mean_cd: List[float] = []
    for class_index, class_name in enumerate(names):
        iou = counts_to_iou(*iou_counts[class_index])
        row = {
            "model": source.name,
            "resolution": resolution,
            "class": class_name,
            "IoU": iou,
            "n_samples": len(source.samples),
            "load_skipped": source.skipped,
        }
        mean_iou.append(iou)
        for tolerance_m in BOUNDARY_TOLERANCES_M:
            key = f"BF_{tolerance_m:.1f}m"
            row[key] = boundary_totals[tolerance_m][class_index].f1()
            mean_bf[tolerance_m].append(row[key])
            row[f"skipped_empty_boundary_{tolerance_m:.1f}m"] = boundary_totals[tolerance_m][class_index].empty_pairs
        if class_index in chamfer_values:
            row["CD_m"] = mean_ignore_nan(chamfer_values[class_index])
            mean_cd.append(row["CD_m"])
            row["CD_skipped"] = chamfer_skipped[class_index]
            row["CD_penalized_empty"] = chamfer_penalized[class_index]
        else:
            row["CD_m"] = float("nan")
            row["CD_skipped"] = 0
            row["CD_penalized_empty"] = 0
        rows.append(row)

    mean_row = {
        "model": source.name,
        "resolution": resolution,
        "class": "mean",
        "IoU": mean_ignore_nan(mean_iou),
        "BF_0.5m": mean_ignore_nan(mean_bf[0.5]),
        "BF_1.0m": mean_ignore_nan(mean_bf[1.0]),
        "BF_2.0m": mean_ignore_nan(mean_bf[2.0]),
        "CD_m": mean_ignore_nan(mean_cd),
        "n_samples": len(source.samples),
        "load_skipped": source.skipped,
        "skipped_empty_boundary_0.5m": sum(boundary_totals[0.5][idx].empty_pairs for idx in range(class_count)),
        "skipped_empty_boundary_1.0m": sum(boundary_totals[1.0][idx].empty_pairs for idx in range(class_count)),
        "skipped_empty_boundary_2.0m": sum(boundary_totals[2.0][idx].empty_pairs for idx in range(class_count)),
        "CD_skipped": sum(chamfer_skipped.values()),
        "CD_penalized_empty": sum(chamfer_penalized.values()),
    }
    rows.append(mean_row)

    lane_index = names.index("lane_divider") if "lane_divider" in names else class_count - 1
    curve = {
        tolerance_m: boundary_totals[tolerance_m][lane_index].f1()
        for tolerance_m in LANE_CURVE_TOLERANCES_M
    }
    return rows, curve


def plot_lane_curve(curves: Mapping[str, Dict[float, float]], png_path: Path) -> None:
    """Save the lane divider BF tolerance sweep."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=300)
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    for index, (label, curve) in enumerate(curves.items()):
        xs = list(LANE_CURVE_TOLERANCES_M)
        ys = [curve.get(x, float("nan")) * 100.0 for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=2.0, color=colors[index % len(colors)], label=label)
    ax.set_xlabel("Boundary tolerance (m)")
    ax.set_ylabel("Lane divider BF (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def _curve_resolution(label: str) -> int:
    """Best-effort parser for labels like Name@512."""
    try:
        return int(label.rsplit("@", 1)[1])
    except (IndexError, ValueError):
        return -1


def select_lane_curve_sources(
    curves: Mapping[str, Dict[float, float]],
    baseline_name: str,
) -> Dict[str, Dict[float, float]]:
    """Select a compact baseline-vs-model subset for the lane-divider BF plot."""
    if len(curves) <= 2:
        return dict(curves)

    selected: Dict[str, Dict[float, float]] = {}
    baseline_prefix = f"{baseline_name}@"
    for label, curve in curves.items():
        if label.startswith(baseline_prefix):
            selected[label] = curve
            break

    non_baseline = [(label, curve) for label, curve in curves.items() if label not in selected]
    if non_baseline:
        # Prefer the highest-resolution non-baseline curve; this keeps the previous
        # CG-BEV@512 behavior without hard-coding a method name.
        label, curve = max(non_baseline, key=lambda item: _curve_resolution(item[0]))
        selected[label] = curve

    return selected if len(selected) >= 2 else dict(curves)

def _format_metric(value: object, scale: float = 1.0) -> str:
    """Format a scalar metric for compact terminal output."""
    try:
        numeric = float(value) * scale
    except (TypeError, ValueError):
        return "--"
    if np.isnan(numeric):
        return "--"
    return f"{numeric:.2f}"


def _method_resolution_label(row: Mapping[str, object]) -> str:
    """Return labels like Method@200 for terminal tables."""
    model = str(row["model"])
    resolution = int(row["resolution"])
    suffix = f"-{resolution}"
    display_name = model.removesuffix(suffix)
    return f"{display_name}@{resolution}"


def print_metrics_table(rows: Sequence[Mapping[str, object]], class_count: int) -> None:
    """Print a paper-style compact metric table to the terminal.

    IoU and BF are printed as percentages, while CD is printed in meters. The
    mean row corresponds to Avg. (all class_count). CD uses the one-sided empty
    skeleton penalty configured by CHAMFER_EMPTY_PENALTY_M; both-empty pairs are
    still skipped because no geometry exists in either prediction or GT.
    """
    if not rows:
        return

    class_order = list(class_names(class_count)) + ["mean"]
    row_by_class: Dict[str, List[Mapping[str, object]]] = {name: [] for name in class_order}
    for row in rows:
        row_class = str(row["class"])
        row_by_class.setdefault(row_class, []).append(row)

    display_rows: List[List[str]] = []
    for class_name in class_order:
        class_rows = row_by_class.get(class_name, [])
        if not class_rows:
            continue
        layer_label = f"Avg. (all {class_count})" if class_name == "mean" else class_name
        for index, row in enumerate(class_rows):
            display_rows.append(
                [
                    layer_label if index == 0 else "",
                    _method_resolution_label(row),
                    _format_metric(row.get("IoU"), scale=100.0),
                    _format_metric(row.get("BF_1.0m"), scale=100.0),
                    _format_metric(row.get("BF_2.0m"), scale=100.0),
                    _format_metric(row.get("CD_m")),
                ]
            )

    if not display_rows:
        return

    headers = ["Layer", "Method & Resolution", "mIoU ↑", "BF@1m ↑", "BF@2m ↑", "CD (m) ↓"]
    widths = [len(header) for header in headers]
    for row in display_rows:
        for col_idx, cell in enumerate(row):
            widths[col_idx] = max(widths[col_idx], len(cell))

    def fmt_row(row: Sequence[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    print("\n[exp1] Boundary metrics summary")
    print(
        f"[exp1] CD policy: both-empty skeleton pairs are skipped; "
        f"one-sided empty pairs use a {CHAMFER_EMPTY_PENALTY_M:.2f}m penalty."
    )
    print(fmt_row(headers))
    print(separator)
    last_nonempty_layer = None
    for row in display_rows:
        if row[0] and last_nonempty_layer is not None:
            print(separator)
        if row[0]:
            last_nonempty_layer = row[0]
        print(fmt_row(row))


def warn_failures(sources: Sequence[ModelSource]) -> None:
    """Print compact skip information for loaded sources."""
    for source in sources:
        if not source.failures:
            continue
        print(f"[load-warning] {source.name}: skipped {len(source.failures)} samples")
        for failure in source.failures[:3]:
            print(f"  - {failure}")


def build_eval_sources(args: argparse.Namespace, baseline_prompt: Path, baseline_name: str) -> List[ModelSource]:
    """Build evaluation sources, using BEVFusion-native GT for that baseline."""
    use_bevfusion_native = (
        not args.no_baseline
        and is_bevfusion_baseline(args.baseline)
        and args.baseline_gt_root is None
    )

    sources: List[ModelSource] = []
    if use_bevfusion_native:
        samples, failures = load_bevfusion_native_samples(
            prompt_path=baseline_prompt,
            data_root=args.data_root,
            split=args.split,
            name=baseline_name,
            pred_root=args.baseline_pred_root,
            gt_resolution=args.baseline_gt_resolution,
            max_samples=args.max_samples,
            class_count=args.class_count,
        )
        sources.append(ModelSource(name=baseline_name, samples=samples, failures=failures))
        sources.extend(
            iter_sources(
                baseline_prompt=None,
                data_root=args.data_root,
                split=args.split,
                dumps=args.dump,
                max_samples=args.max_samples,
                class_count=args.class_count,
                baseline=args.baseline,
                baseline_name=baseline_name,
                baseline_pred_root=args.baseline_pred_root,
                baseline_gt_root=args.baseline_gt_root,
                baseline_gt_resolution=args.baseline_gt_resolution,
                include_baseline=False,
            )
        )
        return sources

    return list(
        iter_sources(
            baseline_prompt=baseline_prompt,
            data_root=args.data_root,
            split=args.split,
            dumps=args.dump,
            max_samples=args.max_samples,
            class_count=args.class_count,
            baseline=args.baseline,
            baseline_name=baseline_name,
            baseline_pred_root=args.baseline_pred_root,
            baseline_gt_root=args.baseline_gt_root,
            baseline_gt_resolution=args.baseline_gt_resolution,
            include_baseline=not args.no_baseline,
        )
    )


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    out_dir = args.out_dir
    metrics_csv = args.metrics_csv or out_dir / "metrics_table.csv"
    skip_csv = args.skip_csv or out_dir / "skip_stats.csv"
    curve_png = args.curve_png or out_dir / "lane_divider_bf_curve.png"

    baseline_name = baseline_display_name(args.baseline, args.baseline_name)
    baseline_prompt = resolve_baseline_prompt(args)

    sources = build_eval_sources(args, baseline_prompt=baseline_prompt, baseline_name=baseline_name)
    if not sources:
        raise SystemExit("No evaluation sources configured. Use cached baseline defaults and/or --dump NAME=DIR.")
    if not args.no_baseline:
        if is_bevfusion_baseline(args.baseline) and args.baseline_gt_root is None:
            print(
                f"[exp1] baseline={baseline_name} prompt={baseline_prompt} "
                f"gt_resolution={args.baseline_gt_resolution} protocol=bevfusion-native-lidar"
            )
            print(
                "[exp1] BEVFusion baseline uses LoadBEVSegmentation-style GT in the "
                "saved-prediction coordinate frame; class index 5 is BEVFusion divider "
                "(road_divider + lane_divider)."
            )
        else:
            print(
                f"[exp1] baseline={baseline_name} prompt={baseline_prompt} "
                f"gt_resolution={args.baseline_gt_resolution}"
            )
    warn_failures(sources)

    rows: List[Mapping[str, object]] = []
    curves: Dict[str, Dict[float, float]] = {}
    for source in sources:
        source_rows, curve = evaluate_source(source, class_count=args.class_count)
        rows.extend(source_rows)
        res = source_resolution(source)
        res_str = str(res)
        display_name = source.name.removesuffix(f"-{res_str}")
        curves[f"{display_name}@{res_str}"] = curve

    columns = [
        "model",
        "resolution",
        "class",
        "IoU",
        "BF_0.5m",
        "BF_1.0m",
        "BF_2.0m",
        "CD_m",
    ]
    skip_columns = [
        "model",
        "resolution",
        "class",
        "n_samples",
        "load_skipped",
        "skipped_empty_boundary_0.5m",
        "skipped_empty_boundary_1.0m",
        "skipped_empty_boundary_2.0m",
        "CD_skipped",
        "CD_penalized_empty",
    ]
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(metrics_csv, index=False)
    pd.DataFrame(rows, columns=skip_columns).to_csv(skip_csv, index=False)
    plot_lane_curve(select_lane_curve_sources(curves, baseline_name), png_path=curve_png)
    print_metrics_table(rows, class_count=args.class_count)
    print(f"[exp1] wrote {metrics_csv}")
    print(f"[exp1] wrote {skip_csv}")
    print(f"[exp1] wrote {curve_png}")


if __name__ == "__main__":
    main()
