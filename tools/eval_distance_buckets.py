"""Distance-bucket BEV mIoU evaluation for CG-BEV supplemental experiments."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from tools.eval_common import (  # noqa: E402
    BEV_SPAN_METERS,
    CLASS_NAMES,
    ModelSource,
    class_names,
    compute_iou_counts,
    counts_to_iou,
    iter_sources,
    mean_ignore_nan,
    px_per_meter,
    source_resolution,
)

np.random.seed(42)


@dataclass(frozen=True)
class DistanceBucket:
    """A radial ego-distance interval in meters."""

    name: str
    start_m: float
    end_m: float

    @property
    def range_label(self) -> str:
        return f"{self.start_m:.0f}-{self.end_m:.0f}"


BUCKETS: Tuple[DistanceBucket, ...] = (
    DistanceBucket("0~15", 0.0, 15.0),
    DistanceBucket("15~30", 15.0, 30.0),
    DistanceBucket("30~40", 30.0, 40.0),
    DistanceBucket("40~50", 40.0, 50.0),
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compute ego-distance bucket mIoU for a baseline and a model dump.")
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
        "--method-dump",
        required=True,
        metavar="NAME=DIR",
        help="Model dump directory to compare against the cached baseline.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--class-count", type=int, default=len(CLASS_NAMES))
    parser.add_argument("--out-dir", type=Path, default=Path("logs/exp2"))
    parser.add_argument("--table-csv", type=Path, default=None)
    parser.add_argument("--skip-csv", type=Path, default=None)
    parser.add_argument("--bars-png", type=Path, default=None)
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


def distance_grid(resolution: int) -> np.ndarray:
    """Return per-pixel radial distance from ego in meters."""
    pixels_per_meter = px_per_meter(resolution)
    half_span = BEV_SPAN_METERS / 2.0
    axis_m = (np.arange(resolution, dtype=np.float64) + 0.5) / pixels_per_meter - half_span
    x_m, y_m = np.meshgrid(axis_m, axis_m)
    return np.sqrt(x_m * x_m + y_m * y_m)


def bucket_mask(distances_m: np.ndarray, bucket: DistanceBucket) -> np.ndarray:
    """Return a boolean mask for one distance bucket."""
    lower = distances_m >= bucket.start_m
    upper = distances_m <= bucket.end_m if bucket.end_m == 50.0 else distances_m < bucket.end_m
    return np.logical_and(lower, upper)


def sanity_check_bucket_pixels(resolutions: Sequence[int], buckets: Sequence[DistanceBucket]) -> None:
    """Print bucket pixel counts for each native resolution."""
    for resolution in sorted(set(resolutions)):
        distances = distance_grid(resolution)
        counts = [int(bucket_mask(distances, bucket).sum()) for bucket in buckets]
        pieces = ", ".join(f"{bucket.name}={count}" for bucket, count in zip(buckets, counts))
        print(f"[distance sanity] resolution={resolution} px_per_meter={px_per_meter(resolution):.2f}: {pieces}")


def evaluate_source_distance(
    source: ModelSource,
    buckets: Sequence[DistanceBucket],
    class_count: int,
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, int]]:
    """Evaluate bucket mIoU for one source."""
    names = class_names(class_count)
    resolution = source_resolution(source)
    distances = distance_grid(resolution)
    regions = {bucket.name: bucket_mask(distances, bucket) for bucket in buckets}
    counts = {
        bucket.name: [[0, 0, 0] for _ in range(class_count)]
        for bucket in buckets
    }
    valid_cells = {bucket.name: [0 for _ in range(class_count)] for bucket in buckets}
    skipped_cells = {bucket.name: 0 for bucket in buckets}

    for sample in tqdm(source.samples, desc=f"Distance buckets: {source.name}"):
        for bucket in buckets:
            region = regions[bucket.name]
            for class_index, _class_name in enumerate(names):
                tp, fp, fn = compute_iou_counts(sample.pred[class_index], sample.gt[class_index], region=region)
                if tp + fp + fn == 0:
                    skipped_cells[bucket.name] += 1
                    continue
                counts[bucket.name][class_index][0] += tp
                counts[bucket.name][class_index][1] += fp
                counts[bucket.name][class_index][2] += fn
                valid_cells[bucket.name][class_index] += 1

    miou_by_bucket: Dict[str, float] = {}
    valid_by_bucket: Dict[str, int] = {}
    for bucket in buckets:
        class_ious = [counts_to_iou(*counts[bucket.name][class_index]) for class_index in range(class_count)]
        miou_by_bucket[bucket.name] = mean_ignore_nan(class_ious)
        valid_by_bucket[bucket.name] = int(sum(valid_cells[bucket.name]))
    return miou_by_bucket, valid_by_bucket, skipped_cells


def build_comparison_rows(
    buckets: Sequence[DistanceBucket],
    baseline_metrics: Mapping[str, float],
    method_metrics: Mapping[str, float],
    baseline_samples: int,
    method_samples: int,
    baseline_valid: Mapping[str, int],
    method_valid: Mapping[str, int],
    baseline_skipped: Mapping[str, int],
    method_skipped: Mapping[str, int],
    baseline_label: str,
    method_label: str,
) -> List[Mapping[str, object]]:
    """Build a baseline-agnostic distance-bucket comparison table."""
    rows: List[Mapping[str, object]] = []
    for bucket in buckets:
        baseline_pct = baseline_metrics[bucket.name] * 100.0 if not np.isnan(baseline_metrics[bucket.name]) else float("nan")
        method_pct = method_metrics[bucket.name] * 100.0 if not np.isnan(method_metrics[bucket.name]) else float("nan")
        abs_improvement = method_pct - baseline_pct if not (np.isnan(baseline_pct) or np.isnan(method_pct)) else float("nan")
        if np.isnan(baseline_pct) or baseline_pct == 0.0 or np.isnan(method_pct):
            rel_improvement_pct = float("nan")
        else:
            rel_improvement_pct = (method_pct - baseline_pct) / baseline_pct * 100.0
        rows.append(
            {
                "bucket": bucket.name,
                "range_m": bucket.range_label,
                "baseline": baseline_label,
                "method": method_label,
                "baseline_mIoU": baseline_pct,
                "method_mIoU": method_pct,
                "abs_improvement": abs_improvement,
                "rel_improvement_pct": rel_improvement_pct,
                "n_samples_baseline": baseline_samples,
                "n_samples_method": method_samples,
                "valid_class_samples_baseline": baseline_valid[bucket.name],
                "valid_class_samples_method": method_valid[bucket.name],
                "skipped_empty_class_samples_baseline": baseline_skipped[bucket.name],
                "skipped_empty_class_samples_method": method_skipped[bucket.name],
            }
        )
    return rows

def plot_distance_bars(
    rows: Sequence[Mapping[str, object]],
    png_path: Path,
    baseline_label: str,
    method_label: str,
) -> None:
    """Save grouped baseline/method mIoU bars."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["bucket"]) for row in rows]
    baseline = np.asarray([float(row["baseline_mIoU"]) for row in rows], dtype=np.float64)
    method = np.asarray([float(row["method_mIoU"]) for row in rows], dtype=np.float64)
    improvement = method - baseline
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=300)
    ax.bar(x - width / 2, baseline, width, label=baseline_label, color="tab:blue")
    bars = ax.bar(x + width / 2, method, width, label=method_label, color="tab:orange")
    for bar, delta in zip(bars, improvement):
        if np.isnan(delta):
            continue
        ax.annotate(
            f"{delta:+.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mIoU (%)")
    ax.set_title(f"Per-distance mIoU: {method_label} vs {baseline_label}")
    finite_values = np.concatenate([baseline[np.isfinite(baseline)], method[np.isfinite(method)]])
    max_value = float(finite_values.max()) if finite_values.size else 0.0
    ax.set_ylim(0, max(5.0, max_value * 1.18))
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_xlabel("Distance to the ego vehicle (m)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)

def select_sources(sources: Sequence[ModelSource], baseline_name: str) -> Tuple[ModelSource, ModelSource]:
    """Return the cached baseline source and the first non-baseline dump source."""
    baseline_source: Optional[ModelSource] = None
    method_source: Optional[ModelSource] = None
    for source in sources:
        if source.name == baseline_name and baseline_source is None:
            baseline_source = source
        elif method_source is None:
            method_source = source
    if baseline_source is None or method_source is None:
        raise SystemExit(
            "Distance-bucket evaluation requires one cached baseline plus one --method-dump NAME=DIR source."
        )
    return baseline_source, method_source

def align_sources_by_token(
    baseline_source: ModelSource,
    method_source: ModelSource,
) -> Tuple[ModelSource, ModelSource]:
    """Return sources restricted to their shared sample tokens."""
    baseline_by_token = {sample.token: sample for sample in baseline_source.samples}
    method_by_token = {sample.token: sample for sample in method_source.samples}
    shared_tokens = tuple(token for token in baseline_by_token if token in method_by_token)
    if not shared_tokens:
        raise SystemExit(f"No shared sample tokens between {baseline_source.name} and {method_source.name} sources.")
    if len(shared_tokens) != len(baseline_source.samples) or len(shared_tokens) != len(method_source.samples):
        print(
            f"[align] using {len(shared_tokens)} shared tokens "
            f"({baseline_source.name}={len(baseline_source.samples)}, "
            f"{method_source.name}={len(method_source.samples)})"
        )
    return (
        rename_source(
            ModelSource(
                baseline_source.name,
                tuple(baseline_by_token[token] for token in shared_tokens),
                baseline_source.failures,
            ),
            baseline_source.name,
        ),
        rename_source(
            ModelSource(
                method_source.name,
                tuple(method_by_token[token] for token in shared_tokens),
                method_source.failures,
            ),
            method_source.name,
        ),
    )


def _format_float(value: object) -> str:
    """Format a metric for terminal tables."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "--"
    if np.isnan(numeric):
        return "--"
    return f"{numeric:.2f}"


def print_distance_table(rows: Sequence[Mapping[str, object]], baseline_label: str, method_label: str) -> None:
    """Print the distance-bucket comparison table to the terminal."""
    if not rows:
        return
    headers = ["Bucket", "Range (m)", f"{baseline_label} mIoU ↑", f"{method_label} mIoU ↑", "Abs. Δ", "Rel. Δ (%)"]
    display_rows = [
        [
            str(row["bucket"]),
            str(row["range_m"]),
            _format_float(row.get("baseline_mIoU")),
            _format_float(row.get("method_mIoU")),
            _format_float(row.get("abs_improvement")),
            _format_float(row.get("rel_improvement_pct")),
        ]
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for row in display_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_row(row: Sequence[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    print("\n[exp2] Distance-bucket mIoU summary")
    print(fmt_row(headers))
    print(separator)
    for row in display_rows:
        print(fmt_row(row))

def warn_failures(sources: Sequence[ModelSource]) -> None:
    """Print compact skip information for loaded sources."""
    for source in sources:
        if not source.failures:
            continue
        print(f"[load-warning] {source.name}: skipped {len(source.failures)} samples")
        for failure in source.failures[:3]:
            print(f"  - {failure}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    out_dir = args.out_dir
    table_csv = args.table_csv or out_dir / "distance_mIoU_table.csv"
    skip_csv = args.skip_csv or out_dir / "distance_skip_stats.csv"
    bars_png = args.bars_png or out_dir / "distance_mIoU_bars.png"

    baseline_name = baseline_display_name(args.baseline, args.baseline_name)
    baseline_prompt = resolve_baseline_prompt(args)

    sources = list(
        iter_sources(
            baseline_prompt=baseline_prompt,
            data_root=args.data_root,
            split=args.split,
            dumps=[args.method_dump],
            max_samples=args.max_samples,
            class_count=args.class_count,
            baseline=args.baseline,
            baseline_name=baseline_name,
            baseline_pred_root=args.baseline_pred_root,
            baseline_gt_root=args.baseline_gt_root,
            baseline_gt_resolution=args.baseline_gt_resolution,
            include_baseline=True,
        )
    )
    warn_failures(sources)
    print(
        f"[exp2] baseline={baseline_name} prompt={baseline_prompt} "
        f"gt_resolution={args.baseline_gt_resolution}"
    )

    baseline_source, method_source = select_sources(sources, baseline_name=baseline_name)
    baseline_source, method_source = align_sources_by_token(baseline_source, method_source)
    sanity_check_bucket_pixels([source_resolution(baseline_source), source_resolution(method_source)], BUCKETS)

    baseline_metrics, baseline_valid, baseline_skipped = evaluate_source_distance(baseline_source, BUCKETS, args.class_count)
    method_metrics, method_valid, method_skipped = evaluate_source_distance(method_source, BUCKETS, args.class_count)
    rows = build_comparison_rows(
        BUCKETS,
        baseline_metrics,
        method_metrics,
        baseline_samples=len(baseline_source.samples),
        method_samples=len(method_source.samples),
        baseline_valid=baseline_valid,
        method_valid=method_valid,
        baseline_skipped=baseline_skipped,
        method_skipped=method_skipped,
        baseline_label=baseline_source.name,
        method_label=method_source.name,
    )

    columns = [
        "bucket",
        "range_m",
        "baseline",
        "method",
        "baseline_mIoU",
        "method_mIoU",
        "abs_improvement",
        "rel_improvement_pct",
        "n_samples_baseline",
        "n_samples_method",
    ]
    skip_columns = [
        "bucket",
        "range_m",
        "baseline",
        "method",
        "valid_class_samples_baseline",
        "valid_class_samples_method",
        "skipped_empty_class_samples_baseline",
        "skipped_empty_class_samples_method",
    ]
    table_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(table_csv, index=False)
    pd.DataFrame(rows, columns=skip_columns).to_csv(skip_csv, index=False)
    plot_distance_bars(rows, png_path=bars_png, baseline_label=baseline_source.name, method_label=method_source.name)
    print_distance_table(rows, baseline_label=baseline_source.name, method_label=method_source.name)
    print(f"[exp2] wrote {table_csv}")
    print(f"[exp2] wrote {skip_csv}")
    print(f"[exp2] wrote {bars_png}")


if __name__ == "__main__":
    main()
