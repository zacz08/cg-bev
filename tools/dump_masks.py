"""Dump CG-BEV prediction and GT masks to per-sample `.npz` files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test import build_dataloader, build_model  # noqa: E402
from tools.eval_common import CLASS_NAMES  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for CG-BEV mask dumping."""
    parser = argparse.ArgumentParser(
        description="Run CG-BEV inference and dump binary pred/gt masks as .npz files."
    )
    parser.add_argument("--config", required=True, help="Path to a CG-BEV config YAML.")
    parser.add_argument("--ckpt", required=True, help="Path to a CG-BEV checkpoint.")
    parser.add_argument("--out-dir", required=True, help="Directory for per-sample .npz dumps.")
    parser.add_argument(
        "--baseline",
        default="lss",
        help="Control baseline feature source used by the CG-BEV model/dataloader, e.g. lss or bevfusion.",
    )
    parser.add_argument("--split", default="val", help="Dataset split to evaluate.")
    parser.add_argument("--resolution", type=int, default=None, help="Override BEV resolution from config.")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sample dumps.")
    parser.add_argument("--seed", type=int, default=42, help="Torch random seed for deterministic sampling setup.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """Resolve a CLI device string into a torch device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def move_batch_to_device(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    """Move tensor values in a dataloader batch to the selected device."""
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def batch_tokens(batch: Mapping[str, object]) -> List[str]:
    """Return sample tokens from a CLDM dataloader batch."""
    raw_tokens = batch.get("sample_token")
    if raw_tokens is None:
        raise KeyError("Batch does not contain `sample_token`.")
    if isinstance(raw_tokens, str):
        return [raw_tokens]
    if isinstance(raw_tokens, (list, tuple)):
        return [str(token) for token in raw_tokens]
    raise TypeError(f"Unsupported sample_token type: {type(raw_tokens)!r}")


def infer_masks(model: torch.nn.Module, batch: Mapping[str, object], device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Run one CG-BEV batch and return pred/GT masks as CHW uint8 arrays."""
    batch_on_device = move_batch_to_device(batch, device)
    gt = batch_on_device["bev_map_gt"]
    if not torch.is_tensor(gt):
        raise TypeError("Expected `bev_map_gt` to be a tensor.")

    gt_mask = ((gt + 1.0) / 2.0 > 0.5).permute(0, 3, 1, 2).to(torch.uint8)
    _, cond, _ = model.get_input(batch_on_device, model.first_stage_key)
    recon = model.sample(batch_size=gt.shape[0], cond=cond)
    pred_mask = (recon > 0).to(torch.uint8)
    return pred_mask.cpu().numpy(), gt_mask.cpu().numpy()


def save_mask_dump(
    path: Path,
    token: str,
    pred: np.ndarray,
    gt: np.ndarray,
    baseline: str,
    config: str,
    ckpt: str,
    split: str,
    resolution: Optional[int],
) -> None:
    """Save one sample's prediction and GT masks to a compressed `.npz` file.

    The metadata is not required by the evaluation scripts, but it makes future
    cache inspection safer when multiple baselines share the same dump format.
    """
    np.savez_compressed(
        path,
        sample_token=np.asarray(token),
        pred=pred.astype(np.uint8, copy=False),
        gt=gt.astype(np.uint8, copy=False),
        class_names=np.asarray(CLASS_NAMES),
        baseline=np.asarray(baseline),
        config=np.asarray(config),
        ckpt=np.asarray(ckpt),
        split=np.asarray(split),
        resolution=np.asarray(-1 if resolution is None else resolution),
    )


def make_test_args(args: argparse.Namespace) -> SimpleNamespace:
    """Create the minimal argument object expected by `test.py` helpers."""
    return SimpleNamespace(
        task="cldm",
        config=args.config,
        ckpt=args.ckpt,
        baseline=args.baseline,
        split=args.split,
        resolution=args.resolution,
        batch_size=args.batch_size,
    )


def dump_masks(args: argparse.Namespace) -> Tuple[int, int]:
    """Run CG-BEV inference and write per-token mask dumps."""
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    cfg = OmegaConf.load(args.config)
    test_args = make_test_args(args)
    model = build_model(test_args, cfg).to(device)
    loader = build_dataloader(test_args, cfg)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    seen = 0
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Dumping masks"):
            tokens = batch_tokens(batch)
            remaining = None if args.max_samples is None else args.max_samples - seen
            if remaining is not None and remaining <= 0:
                break
            if remaining is not None and len(tokens) > remaining:
                # Keep the dataloader simple and truncate after inference output below.
                tokens = tokens[:remaining]

            pred_masks, gt_masks = infer_masks(model, batch, device)
            for index, token in enumerate(tokens):
                out_path = output_dir / f"{token}.npz"
                if out_path.exists() and not args.overwrite:
                    skipped += 1
                else:
                    save_mask_dump(
                        out_path,
                        token,
                        pred_masks[index],
                        gt_masks[index],
                        baseline=args.baseline,
                        config=args.config,
                        ckpt=args.ckpt,
                        split=args.split,
                        resolution=args.resolution,
                    )
                    written += 1
                seen += 1
                if args.max_samples is not None and seen >= args.max_samples:
                    break
            if args.max_samples is not None and seen >= args.max_samples:
                break
    return written, skipped


def main() -> None:
    """CLI entry point for CG-BEV mask dumping."""
    args = parse_args()
    written, skipped = dump_masks(args)
    print(f"[dump_masks] baseline={args.baseline} wrote={written} skipped={skipped} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()