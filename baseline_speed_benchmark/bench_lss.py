from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch
from omegaconf import OmegaConf

from common import (
    make_log_dir,
    measure_forward_speed,
    print_summary,
    profile_flops,
    resolve_device,
    seed_everything,
    write_metrics,
)


DEFAULT_PROJECT = Path("third_party/lift-splat-shoot")
DEFAULT_CONFIG = DEFAULT_PROJECT / "configs/lss.yaml"
DEFAULT_CKPT = DEFAULT_PROJECT / "ckpts/6_layer_epoch=30-step=13299.ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Lift-Splat-Shoot inference speed on nuScenes val.")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT), help="Lift-Splat-Shoot project path.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="LSS config YAML path.")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="LSS checkpoint path.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for timing; default matches online bs=1.")
    parser.add_argument("--num-workers", type=int, default=2, help="Validation dataloader workers.")
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--measure-batches", type=int, default=100)
    parser.add_argument("--skip-flops", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-root", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_lss(args: argparse.Namespace):
    project = Path(args.project).resolve()
    sys.path.insert(0, str(project))
    os.chdir(project)

    from src.data import compile_data
    from src.models_pl import LiftSplatShoot

    cfg = OmegaConf.load(args.config)
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if "frustum" in state_dict:
        _depth_bins, frustum_h, frustum_w, _xyz = state_dict["frustum"].shape
        cfg.data_aug.final_dim = [int(frustum_h * cfg.model.downsample), int(frustum_w * cfg.model.downsample)]
        print(f"[Config:LSS] matched final_dim to checkpoint frustum: {cfg.data_aug.final_dim}")
    if "trainer_config" not in cfg:
        cfg.trainer_config = {"iou_loss": "lovasz", "lr": 1e-3, "min_lr": 1e-4, "weight_decay": 1e-4, "warmup_percent": 0.1}
    cfg.loader.batch_size = int(args.batch_size)
    cfg.loader.nworkers = int(args.num_workers)
    cfg.dataset.dataroot = str((project / cfg.dataset.dataroot).resolve()) if not Path(cfg.dataset.dataroot).is_absolute() else cfg.dataset.dataroot

    model = LiftSplatShoot(cfg).cpu()
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"[Model:LSS] config={args.config}")
    print(f"[Ckpt:LSS] {args.ckpt}")
    print(f"[Ckpt:LSS] missing={len(missing_keys)} unexpected={len(unexpected_keys)}")

    _, val_loader = compile_data(cfg=cfg, parser_name="segmentationdata")
    print(f"[Data:LSS] split=val | bs={cfg.loader.batch_size} | workers={cfg.loader.nworkers} | n={len(val_loader.dataset)}")
    return model, val_loader, cfg


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    log_dir = make_log_dir("lss", args.log_root)
    model, loader, cfg = load_lss(args)

    def forward_fn(batch):
        imgs, rots, trans, intrins, post_rots, post_trans, _bev_seg_gt, _sample_token = batch
        return model(imgs, rots, trans, intrins, post_rots, post_trans)

    metrics = measure_forward_speed(
        model,
        loader,
        forward_fn,
        device=device,
        warmup_batches=args.warmup_batches,
        measure_batches=args.measure_batches,
    )
    if args.skip_flops:
        flops, flops_per_sample, flops_note = None, None, "skipped by --skip-flops"
    else:
        flops, flops_per_sample, flops_note = profile_flops(model, loader, forward_fn, device=device)
    metrics.update(
        {
            "model": "LSS",
            "batch_size": int(args.batch_size),
            "flops": flops,
            "gflops": None if flops is None else flops / 1e9,
            "flops_per_sample": flops_per_sample,
            "gflops_per_sample": None if flops_per_sample is None else flops_per_sample / 1e9,
            "flops_note": flops_note,
        }
    )
    csv_path, _json_path = write_metrics(
        log_dir,
        "lss",
        metrics,
        {
            "project": str(Path(args.project).resolve()),
            "config": str(Path(args.config).resolve()),
            "ckpt": str(Path(args.ckpt).resolve()),
            "dataset_root": str(Path(cfg.dataset.dataroot).resolve()),
            "device": str(device),
        },
    )
    print_summary("LSS", metrics, csv_path, flops_note)


if __name__ == "__main__":
    main()
