from __future__ import annotations

import argparse
import importlib
import os
import sys
import types
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import torch
from mmcv import Config

from common import (
    make_log_dir,
    measure_forward_speed,
    print_summary,
    profile_flops,
    resolve_device,
    seed_everything,
    write_metrics,
)
from openmmlab_compat import install_mmcv_ops_shim


DEFAULT_PROJECT = Path("third_party/bevformer")
DEFAULT_CONFIG = DEFAULT_PROJECT / "projects/configs/bevformer/bevformer_base_seg_det_150x150.py"
DEFAULT_CKPT = DEFAULT_PROJECT / "work_dirs/bevformer_base_seg_det_150x150/epoch_18_best.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark BEVFormer forward speed with val-shaped dummy inputs.")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--measure-batches", type=int, default=100)
    parser.add_argument("--skip-flops", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-root", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-height", type=int, default=900)
    parser.add_argument("--input-width", type=int, default=1600)
    return parser.parse_args()


class RepeatLoader:
    def __init__(self, batch, length: int):
        self.batch = batch
        self.length = length

    def __iter__(self):
        for _ in range(self.length):
            yield self.batch

    def __len__(self):
        return self.length


def import_plugin(cfg: Config, project: Path) -> None:
    projects_dir = project / "projects"
    plugin_dir = projects_dir / "mmdet3d_plugin"
    package_paths = {
        "projects": projects_dir,
        "projects.mmdet3d_plugin": plugin_dir,
        "projects.mmdet3d_plugin.core": plugin_dir / "core",
        "projects.mmdet3d_plugin.core.bbox": plugin_dir / "core/bbox",
        "projects.mmdet3d_plugin.core.bbox.coders": plugin_dir / "core/bbox/coders",
        "projects.mmdet3d_plugin.models": plugin_dir / "models",
        "projects.mmdet3d_plugin.models.utils": plugin_dir / "models/utils",
        "projects.mmdet3d_plugin.bevformer": plugin_dir / "bevformer",
    }
    for package_name, package_path in package_paths.items():
        module = sys.modules.get(package_name)
        if module is None:
            module = types.ModuleType(package_name)
            module.__path__ = [str(package_path)]
            sys.modules[package_name] = module

    for module_name in [
        "projects.mmdet3d_plugin.core.bbox.coders.nms_free_coder",
        "projects.mmdet3d_plugin.bevformer.modules",
        "projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head",
        "projects.mmdet3d_plugin.bevformer.detectors.bevformer",
    ]:
        importlib.import_module(module_name)


def patch_config_for_sm120(cfg: Config) -> None:
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False


def load_checkpoint_compat(model: torch.nn.Module, ckpt_path: str) -> tuple[list[str], list[str]]:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    result = model.load_state_dict(cleaned, strict=False)
    return list(result.missing_keys), list(result.unexpected_keys)


def build_bevformer(args: argparse.Namespace):
    install_mmcv_ops_shim()
    project = Path(args.project).resolve()
    sys.path.insert(0, str(project))
    os.chdir(project)

    cfg = Config.fromfile(args.config)
    patch_config_for_sm120(cfg)
    import_plugin(cfg, project)

    from mmdet3d.models import build_model

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    missing_keys, unexpected_keys = load_checkpoint_compat(model, args.ckpt)
    model.eval()
    print(f"[Model:BEVFormer] config={args.config}")
    print(f"[Ckpt:BEVFormer] {args.ckpt}")
    print(f"[Ckpt:BEVFormer] missing={len(missing_keys)} unexpected={len(unexpected_keys)}")
    return model, cfg


def make_dummy_batch(args: argparse.Namespace, cfg: Config):
    batch_size = int(args.batch_size)
    num_cams = 6
    image = torch.randn(batch_size, num_cams, 3, args.input_height, args.input_width)
    lidar2img = [np.eye(4, dtype=np.float32) for _ in range(num_cams)]
    metas = []
    for batch_index in range(batch_size):
        metas.append(
            {
                "scene_token": f"dummy_scene_{batch_index}",
                "can_bus": np.zeros(18, dtype=np.float32),
                "lidar2img": lidar2img,
                "img_shape": [(args.input_height, args.input_width, 3) for _ in range(num_cams)],
                "pcd_horizontal_flip": False,
                "pcd_vertical_flip": False,
                "pcd_rotation": np.eye(3, dtype=np.float32),
                "pcd_scale_factor": 1.0,
            }
        )
    return {"img": [image], "img_metas": [metas]}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    log_dir = make_log_dir("bevformer", args.log_root)
    model, cfg = build_bevformer(args)
    batch = make_dummy_batch(args, cfg)
    loader = RepeatLoader(batch, args.warmup_batches + args.measure_batches + 1)

    def forward_fn(batch):
        return model(return_loss=False, rescale=True, **batch)

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
            "model": "BEVFormer",
            "batch_size": int(args.batch_size),
            "input_height": int(args.input_height),
            "input_width": int(args.input_width),
            "flops": flops,
            "gflops": None if flops is None else flops / 1e9,
            "flops_per_sample": flops_per_sample,
            "gflops_per_sample": None if flops_per_sample is None else flops_per_sample / 1e9,
            "flops_note": flops_note,
            "compat_note": "DCNv2 stages kept; missing DCNv2 CUDA op is emulated with a PyTorch module containing the main convolution and conv_offset branch. Missing mmcv CUDA attention ops use PyTorch fallback in benchmark scope only.",
        }
    )
    csv_path, _json_path = write_metrics(
        log_dir,
        "bevformer",
        metrics,
        {
            "project": str(Path(args.project).resolve()),
            "config": str(Path(args.config).resolve()),
            "ckpt": str(Path(args.ckpt).resolve()),
            "device": str(device),
        },
    )
    print_summary("BEVFormer", metrics, csv_path, flops_note)


if __name__ == "__main__":
    main()
