from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path
from typing import Any

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import torch
from mmcv import Config
from torch import nn

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


DEFAULT_PROJECT = Path("third_party/bevfusion")
DEFAULT_CONFIG = DEFAULT_PROJECT / "configs/nuscenes/seg/camera-bev256d2.yaml"
DEFAULT_CKPT = DEFAULT_PROJECT / "pretrained/camera-only-seg.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark BEVFusion forward speed with val-shaped dummy inputs.")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--measure-batches", type=int, default=100)
    parser.add_argument("--num-points", type=int, default=0)
    parser.add_argument("--skip-flops", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-root", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dummy-lidar", action="store_true", help="Use a dummy sparse LiDAR encoder if a fusion config with lidar is explicitly selected.")
    return parser.parse_args()


class RepeatLoader:
    def __init__(self, batch: dict[str, Any], length: int):
        self.batch = batch
        self.length = length

    def __iter__(self):
        for _ in range(self.length):
            yield self.batch

    def __len__(self):
        return self.length


def _bev_pool_torch(feats: torch.Tensor, coords: torch.Tensor, batch_size: int, depth: int, height: int, width: int) -> torch.Tensor:
    if coords.numel() == 0:
        return feats.new_zeros((batch_size, feats.shape[-1], depth, height, width))
    coords = coords.long()
    kept = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < width)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < height)
        & (coords[:, 2] >= 0)
        & (coords[:, 2] < depth)
        & (coords[:, 3] >= 0)
        & (coords[:, 3] < batch_size)
    )
    feats = feats[kept]
    coords = coords[kept]
    output = feats.new_zeros((batch_size, depth, height, width, feats.shape[-1]))
    if feats.numel() > 0:
        output.index_put_((coords[:, 3], coords[:, 2], coords[:, 1], coords[:, 0]), feats, accumulate=True)
    return output.permute(0, 4, 1, 2, 3).contiguous()


class _Voxelization(nn.Module):
    def __init__(self, max_num_points: int = 10, max_voxels=None, **kwargs):
        super().__init__()
        self.max_num_points = max(1, int(max_num_points if max_num_points and max_num_points > 0 else 1))
        self.max_voxels = max_voxels[1] if isinstance(max_voxels, (list, tuple)) else (max_voxels or 20000)

    def forward(self, points: torch.Tensor):
        num_voxels = max(1, min(int(points.shape[0]), int(self.max_voxels)))
        feature_dim = int(points.shape[1]) if points.ndim == 2 else 5
        voxels = points.new_zeros((num_voxels, self.max_num_points, feature_dim))
        count = min(num_voxels, int(points.shape[0]))
        voxels[:count, 0, :feature_dim] = points[:count, :feature_dim]
        coords = torch.zeros((num_voxels, 3), device=points.device, dtype=torch.int32)
        coords[:, 0] = torch.arange(num_voxels, device=points.device, dtype=torch.int32) % 32
        coords[:, 1] = torch.arange(num_voxels, device=points.device, dtype=torch.int32) % 128
        coords[:, 2] = torch.arange(num_voxels, device=points.device, dtype=torch.int32) % 128
        sizes = torch.ones((num_voxels,), device=points.device, dtype=torch.int32)
        return voxels, coords, sizes


class _SparseConvTensor:
    def __init__(self, features, indices=None, spatial_shape=None, batch_size=1):
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape or [1, 1, 1]
        self.batch_size = int(batch_size) if not torch.is_tensor(batch_size) else int(batch_size.item())

    def dense(self):
        channels = int(self.features.shape[-1]) if self.features.ndim > 1 else 1
        return self.features.new_zeros((self.batch_size, channels, 1, 1, 1))


class _SparseIdentity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class _SparseSequential(nn.Sequential):
    pass


class DummySparseEncoder(nn.Module):
    def __init__(self, out_channels: int = 256, height: int = 128, width: int = 128, **kwargs):
        super().__init__()
        self.out_channels = int(out_channels)
        self.height = int(height)
        self.width = int(width)

    def forward(self, feats, coords, batch_size, **kwargs):
        batch = int(batch_size.item()) if torch.is_tensor(batch_size) else int(batch_size)
        return feats.new_zeros((batch, self.out_channels, self.height, self.width))


def _make_sparse_convmodule(*args, **kwargs):
    return _SparseIdentity()


def _zero_first(*args, **kwargs):
    if args and torch.is_tensor(args[0]):
        return torch.zeros_like(args[0])
    return None


def install_bevfusion_ops_shim() -> None:
    ops_module = types.ModuleType("mmdet3d.ops")
    ops_module.__path__ = []
    ops_module.bev_pool = _bev_pool_torch
    ops_module.Voxelization = _Voxelization
    ops_module.DynamicScatter = _Voxelization
    ops_module.voxelization = _zero_first
    ops_module.dynamic_scatter = _zero_first
    ops_module.SparseBasicBlock = _SparseIdentity
    ops_module.SparseBottleneck = _SparseIdentity
    ops_module.make_sparse_convmodule = _make_sparse_convmodule
    ops_module.points_in_boxes_batch = _zero_first
    ops_module.points_in_boxes_cpu = _zero_first
    ops_module.points_in_boxes_gpu = _zero_first
    ops_module.feature_decorator = _zero_first

    spconv_module = types.ModuleType("mmdet3d.ops.spconv")
    spconv_module.SparseConvTensor = _SparseConvTensor
    spconv_module.SparseSequential = _SparseSequential
    spconv_module.SubMConv3d = _SparseIdentity
    spconv_module.SparseConv3d = _SparseIdentity
    spconv_module.SparseInverseConv3d = _SparseIdentity
    ops_module.spconv = spconv_module

    iou3d_module = types.ModuleType("mmdet3d.ops.iou3d")
    iou3d_module.iou3d_cuda = types.SimpleNamespace()
    iou3d_utils_module = types.ModuleType("mmdet3d.ops.iou3d.iou3d_utils")
    iou3d_utils_module.nms_gpu = lambda boxes, scores, thresh, *args, **kwargs: torch.arange(boxes.shape[0], device=boxes.device)
    iou3d_utils_module.nms_normal_gpu = iou3d_utils_module.nms_gpu
    iou3d_module.iou3d_utils = iou3d_utils_module

    roiaware_module = types.ModuleType("mmdet3d.ops.roiaware_pool3d")
    roiaware_module.points_in_boxes_gpu = _zero_first
    roiaware_module.points_in_boxes_cpu = _zero_first
    roiaware_module.points_in_boxes_batch = _zero_first

    sys.modules["mmdet3d.ops"] = ops_module
    sys.modules["mmdet3d.ops.spconv"] = spconv_module
    sys.modules["mmdet3d.ops.iou3d"] = iou3d_module
    sys.modules["mmdet3d.ops.iou3d.iou3d_utils"] = iou3d_utils_module
    sys.modules["mmdet3d.ops.roiaware_pool3d"] = roiaware_module


def load_bevfusion_config(config_path: str) -> Config:
    from torchpack.utils.config import configs
    from mmdet3d.utils import recursive_eval

    configs.load(config_path, recursive=True)
    return Config(recursive_eval(configs), filename=config_path)


def patch_config(cfg: Config, args: argparse.Namespace) -> None:
    cfg.model.pretrained = None
    camera_backbone = cfg.model.encoders.camera.backbone
    camera_backbone.pretrained = None
    camera_backbone.init_cfg = None
    if args.dummy_lidar and cfg.model.encoders.get("lidar") is not None:
        cfg.model.encoders.lidar.backbone = dict(type="DummySparseEncoder", out_channels=256, height=128, width=128)


def patch_lss_signature() -> None:
    from mmdet3d.models.vtransforms.lss import LSSTransform

    original_get_cam_feats = LSSTransform.get_cam_feats
    if getattr(original_get_cam_feats, "_baseline_patched", False):
        return

    def wrapped_get_cam_feats(self, x, *args, **kwargs):
        return original_get_cam_feats(self, x)

    original_forward = LSSTransform.forward

    def wrapped_forward(
        self,
        img,
        points,
        radar,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        *args,
        **kwargs,
    ):
        return original_forward(
            self,
            img,
            points,
            radar,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            **kwargs,
        )

    wrapped_get_cam_feats._baseline_patched = True
    LSSTransform.get_cam_feats = wrapped_get_cam_feats
    LSSTransform.forward = wrapped_forward


def register_dummy_lidar() -> None:
    from mmdet.models import BACKBONES

    if "DummySparseEncoder" not in BACKBONES.module_dict:
        BACKBONES.register_module(module=DummySparseEncoder)


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


def build_bevfusion(args: argparse.Namespace):
    install_mmcv_ops_shim()
    install_bevfusion_ops_shim()
    project = Path(args.project).resolve()
    sys.path.insert(0, str(project))
    os.chdir(project)

    cfg = load_bevfusion_config(args.config)
    patch_config(cfg, args)

    import mmdet3d.models  # noqa: F401
    patch_lss_signature()
    register_dummy_lidar()
    from mmdet3d.models import build_model

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    missing_keys, unexpected_keys = load_checkpoint_compat(model, args.ckpt)
    model.eval()
    print(f"[Model:BEVFusion] config={args.config}")
    print(f"[Ckpt:BEVFusion] {args.ckpt}")
    print(f"[Ckpt:BEVFusion] missing={len(missing_keys)} unexpected={len(unexpected_keys)}")
    if args.dummy_lidar:
        print("[Compat:BEVFusion] lidar sparse encoder uses DummySparseEncoder because spconv CUDA ops are unavailable for this env")
    else:
        print("[Mode:BEVFusion] camera-only config; lidar encoder and fuser are not constructed")
    return model, cfg


def _identity_batch_matrix(batch_size: int, num_cams: int) -> torch.Tensor:
    return torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(batch_size, num_cams, 1, 1)


def make_dummy_batch(args: argparse.Namespace, cfg: Config):
    batch_size = int(args.batch_size)
    num_cams = 6
    image_height, image_width = [int(value) for value in cfg.image_size]
    img = torch.randn(batch_size, num_cams, 3, image_height, image_width)
    point_count = max(0, int(args.num_points))
    points = [torch.randn(point_count, 5) for _ in range(batch_size)]

    camera2ego = _identity_batch_matrix(batch_size, num_cams)
    lidar2camera = _identity_batch_matrix(batch_size, num_cams)
    camera2lidar = _identity_batch_matrix(batch_size, num_cams)
    img_aug_matrix = _identity_batch_matrix(batch_size, num_cams)
    camera_intrinsics = _identity_batch_matrix(batch_size, num_cams)
    camera_intrinsics[:, :, 0, 0] = 1000.0
    camera_intrinsics[:, :, 1, 1] = 1000.0
    camera_intrinsics[:, :, 0, 2] = image_width / 2.0
    camera_intrinsics[:, :, 1, 2] = image_height / 2.0
    lidar2image = camera_intrinsics.clone()

    lidar2ego = torch.eye(4, dtype=torch.float32).view(1, 4, 4).repeat(batch_size, 1, 1)
    lidar_aug_matrix = torch.eye(4, dtype=torch.float32).view(1, 4, 4).repeat(batch_size, 1, 1)
    gt_masks_bev = torch.zeros(batch_size, len(cfg.map_classes), 200, 200, dtype=torch.float32)
    metas = [dict(token=f"dummy_{batch_index}") for batch_index in range(batch_size)]

    return {
        "img": img,
        "points": points,
        "camera2ego": camera2ego,
        "lidar2ego": lidar2ego,
        "lidar2camera": lidar2camera,
        "lidar2image": lidar2image,
        "camera_intrinsics": camera_intrinsics,
        "camera2lidar": camera2lidar,
        "img_aug_matrix": img_aug_matrix,
        "lidar_aug_matrix": lidar_aug_matrix,
        "metas": metas,
        "depths": None,
        "gt_masks_bev": gt_masks_bev,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    log_dir = make_log_dir("bevfusion", args.log_root)
    model, cfg = build_bevfusion(args)
    batch = make_dummy_batch(args, cfg)
    loader = RepeatLoader(batch, args.warmup_batches + args.measure_batches + 1)

    def forward_fn(batch):
        return model(**batch)

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
            "model": "BEVFusion",
            "batch_size": int(args.batch_size),
            "input_height": int(cfg.image_size[0]),
            "input_width": int(cfg.image_size[1]),
            "num_points": int(args.num_points),
            "dummy_lidar": bool(args.dummy_lidar),
            "uses_lidar_encoder": cfg.model.encoders.get("lidar") is not None,
            "flops": flops,
            "gflops": None if flops is None else flops / 1e9,
            "flops_per_sample": flops_per_sample,
            "gflops_per_sample": None if flops_per_sample is None else flops_per_sample / 1e9,
            "flops_note": flops_note,
        }
    )
    csv_path, _json_path = write_metrics(
        log_dir,
        "bevfusion",
        metrics,
        {
            "project": str(Path(args.project).resolve()),
            "config": str(Path(args.config).resolve()),
            "ckpt": str(Path(args.ckpt).resolve()),
            "device": str(device),
            "compat_note": "camera-only benchmark by default; PyTorch bev_pool replaces the unavailable BEVFusion CUDA bev_pool op in benchmark scope. DummySparseEncoder is only used when --dummy-lidar is active with an explicit lidar config.",
        },
    )
    print_summary("BEVFusion", metrics, csv_path, flops_note)


if __name__ == "__main__":
    main()
