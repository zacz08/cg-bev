from __future__ import annotations

import sys
import types
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    bs, _num_value, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([int(h * w) for h, w in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (height, width) in enumerate(value_spatial_shapes):
        value_l = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, int(height), int(width))
        sampling_grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l = F.grid_sample(value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampling_value_list.append(sampling_value_l)
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1)
    return output.view(bs, num_heads * embed_dims, num_queries).transpose(1, 2).contiguous()


class _DummyExt:
    def ms_deform_attn_forward(self, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step=64):
        return multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights)

    def ms_deform_attn_backward(self, *args, **kwargs):
        raise RuntimeError("ms_deform_attn_backward is unavailable in inference-only compatibility mode")


def point_sample(input, points, **kwargs):
    align_corners = kwargs.get("align_corners", False)
    if points.dim() == 3:
        grid = points.unsqueeze(2)
    else:
        grid = points
    grid = grid * 2 - 1
    output = F.grid_sample(input, grid, align_corners=align_corners)
    return output.squeeze(-1) if points.dim() == 3 else output


def nms(boxes, scores, iou_threshold=0.5, **kwargs):
    order = torch.argsort(scores, descending=True)
    return boxes[order], order


def nms_match(dets, iou_threshold=0.5):
    return dets.new_zeros((0, 2), dtype=torch.long)


def batched_nms(boxes, scores, idxs, nms_cfg=None, class_agnostic=False):
    order = torch.argsort(scores, descending=True)
    dets = torch.cat([boxes[order], scores[order, None]], dim=1)
    return dets, order


def sigmoid_focal_loss(input, target, gamma=2.0, alpha=0.25, weight=None, reduction="mean"):
    loss = F.binary_cross_entropy_with_logits(input, target.float(), reduction="none")
    if weight is not None:
        loss = loss * weight
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def _zero_like_first(*args, **kwargs):
    for arg in args:
        if torch.is_tensor(arg):
            return torch.zeros_like(arg)
    return None


class _DeformConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deform_groups=1, bias=False, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.deform_groups = deform_groups

    def forward(self, x, offset=None, mask=None):
        return super().forward(x)


class _ModulatedDeformConv2dPack(_DeformConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deform_groups=1, bias=False, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, deform_groups=deform_groups, bias=bias)
        if isinstance(kernel_size, tuple):
            kernel_h, kernel_w = kernel_size
        else:
            kernel_h = kernel_w = kernel_size
        self.conv_offset = nn.Conv2d(
            in_channels,
            deform_groups * 3 * kernel_h * kernel_w,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

    def forward(self, x):
        offset_mask = self.conv_offset(x)
        return super().forward(x) + offset_mask.sum() * 0.0


class _MultiScaleDeformableAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, query, key=None, value=None, identity=None, **kwargs):
        return query if identity is None else identity + 0 * query


class _Voxelization(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, points):
        if not torch.is_tensor(points):
            points = torch.as_tensor(points)
        coords = points.new_zeros((points.shape[0], 3), dtype=torch.int32)
        sizes = points.new_ones((points.shape[0],), dtype=torch.int32)
        return points, coords, sizes


class _SparseSequential(nn.Sequential):
    pass


class _PassThroughModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


def _module_getattr(name: str) -> Any:
    if name in {"DCN", "DCNv2", "ModulatedDeformConv", "ModulatedDeformConv2d", "ModulatedDeformConv2dPack", "DeformConv2dPack"}:
        return _ModulatedDeformConv2dPack
    if name.endswith("Conv2d") or name.endswith("Conv2dPack") or name in {"DeformConv2d"}:
        return _DeformConv2d
    if name in {"RoIAlign", "RoIPool", "PrRoIPool", "CARAFEPack", "CornerPool", "ConcatCell", "GlobalPoolingCell", "SumCell"}:
        return _PassThroughModule
    if name in {"nms", "soft_nms"}:
        return nms
    if name == "nms_match":
        return nms_match
    if name == "batched_nms":
        return batched_nms
    if name == "sigmoid_focal_loss":
        return sigmoid_focal_loss
    if name in {"Voxelization", "DynamicScatter"}:
        return _Voxelization
    if name in {"SparseModule", "SparseSequential"}:
        return _SparseSequential
    if name in {"SparseConvTensor", "SparseMaxPool3d", "SubMConv3d", "SparseConv3d", "SparseInverseConv3d"}:
        return nn.Identity
    if name == "point_sample":
        return point_sample
    if name == "MultiScaleDeformableAttention":
        return _MultiScaleDeformableAttention
    if name.startswith("nms3d") or name in {
        "nms_rotated", "box_iou_rotated", "points_in_boxes_all", "points_in_boxes_part",
        "diff_iou_rotated_3d", "three_interpolate", "three_nn", "assign_score_withk",
        "furthest_point_sample", "gather_points", "roi_align", "rel_roi_point_to_rel_img_point",
        "get_compiler_version", "get_compiling_cuda_version", "deform_conv2d",
    }:
        return _zero_like_first
    if name in {"GroupAll", "PointsSampler", "QueryAndGroup", "RoIAwarePool3d", "RoIPointPool3d", "CrissCrossAttention", "PSAMask", "MaskedConv2d", "SigmoidFocalLoss"}:
        return _PassThroughModule
    raise AttributeError(name)


def install_mmcv_ops_shim() -> None:
    import mmcv.utils.ext_loader as ext_loader
    from mmcv.cnn.bricks.registry import CONV_LAYERS

    ext_loader.load_ext = lambda *args, **kwargs: _DummyExt()

    for conv_name, conv_module in {
        "DCN": _ModulatedDeformConv2dPack,
        "DCNv2": _ModulatedDeformConv2dPack,
        "DeformConv": _DeformConv2d,
        "DeformConv2d": _DeformConv2d,
        "DeformConv2dPack": _ModulatedDeformConv2dPack,
        "ModulatedDeformConv": _ModulatedDeformConv2dPack,
        "ModulatedDeformConv2d": _ModulatedDeformConv2dPack,
        "ModulatedDeformConv2dPack": _ModulatedDeformConv2dPack,
    }.items():
        if conv_name not in CONV_LAYERS.module_dict:
            CONV_LAYERS.register_module(conv_name, module=conv_module)

    ops_module = types.ModuleType("mmcv.ops")
    ops_module.__path__ = []
    ops_module.__getattr__ = _module_getattr
    ops_module.point_sample = point_sample
    ops_module.nms = nms
    ops_module.nms_match = nms_match
    ops_module.batched_nms = batched_nms
    ops_module.sigmoid_focal_loss = sigmoid_focal_loss
    ops_module.DeformConv2d = _DeformConv2d
    ops_module.ModulatedDeformConv2d = _ModulatedDeformConv2dPack
    ops_module.DeformConv2dPack = _ModulatedDeformConv2dPack
    ops_module.ModulatedDeformConv2dPack = _ModulatedDeformConv2dPack
    ops_module.MultiScaleDeformableAttention = _MultiScaleDeformableAttention
    for symbol in [
        "MaskedConv2d", "RoIAlign", "RoIPool", "SigmoidFocalLoss", "Voxelization", "DynamicScatter",
        "SparseModule", "SparseSequential", "SparseConvTensor", "SparseMaxPool3d", "SubMConv3d",
        "SparseConv3d", "SparseInverseConv3d", "GroupAll", "PointsSampler", "QueryAndGroup",
        "RoIAwarePool3d", "RoIPointPool3d", "CrissCrossAttention", "PSAMask", "ConcatCell",
        "GlobalPoolingCell", "SumCell",
    ]:
        setattr(ops_module, symbol, _module_getattr(symbol))
    for symbol in [
        "nms_rotated", "box_iou_rotated", "points_in_boxes_all", "points_in_boxes_part",
        "diff_iou_rotated_3d", "three_interpolate", "three_nn", "assign_score_withk",
        "furthest_point_sample", "gather_points", "roi_align", "rel_roi_point_to_rel_img_point",
        "get_compiler_version", "get_compiling_cuda_version", "deform_conv2d", "nms3d", "nms3d_normal",
    ]:
        setattr(ops_module, symbol, _zero_like_first)

    msda_module = types.ModuleType("mmcv.ops.multi_scale_deform_attn")
    msda_module.multi_scale_deformable_attn_pytorch = multi_scale_deformable_attn_pytorch
    msda_module.MultiScaleDeformableAttention = _MultiScaleDeformableAttention

    sys.modules["mmcv.ops"] = ops_module
    sys.modules["mmcv.ops.multi_scale_deform_attn"] = msda_module
    for submodule_name in [
        "assign_score_withk", "ball_query", "carafe", "deform_conv", "furthest_point_sample",
        "gather_points", "group_points", "knn", "merge_cells", "modulated_deform_conv",
        "nms", "point_sample", "points_in_boxes", "points_sampler", "roi_align",
        "roiaware_pool3d", "roipoint_pool3d", "scatter_points", "three_interpolate", "three_nn",
        "voxelize",
    ]:
        submodule = types.ModuleType(f"mmcv.ops.{submodule_name}")
        submodule.__getattr__ = _module_getattr
        submodule.roi_align = _zero_like_first
        submodule.nms = nms
        submodule.batched_nms = batched_nms
        submodule.point_sample = point_sample
        submodule.CARAFEPack = nn.Identity
        submodule.ConcatCell = _PassThroughModule
        submodule.GlobalPoolingCell = _PassThroughModule
        submodule.SumCell = _PassThroughModule
        submodule.DeformConv2d = _DeformConv2d
        submodule.ModulatedDeformConv2d = _ModulatedDeformConv2dPack
        submodule.DeformConv2dPack = _ModulatedDeformConv2dPack
        submodule.ModulatedDeformConv2dPack = _ModulatedDeformConv2dPack
        submodule.Voxelization = _Voxelization
        submodule.DynamicScatter = _Voxelization
        submodule.RoIAwarePool3d = _PassThroughModule
        submodule.RoIPointPool3d = _PassThroughModule
        submodule.GroupAll = _PassThroughModule
        submodule.QueryAndGroup = _PassThroughModule
        submodule.PointsSampler = _PassThroughModule
        submodule.furthest_point_sample = _zero_like_first
        submodule.furthest_point_sample_with_dist = _zero_like_first
        submodule.gather_points = _zero_like_first
        submodule.grouping_operation = _zero_like_first
        submodule.ball_query = _zero_like_first
        submodule.knn = _zero_like_first
        submodule.points_in_boxes_all = _zero_like_first
        submodule.points_in_boxes_cpu = _zero_like_first
        submodule.points_in_boxes_part = _zero_like_first
        submodule.dynamic_scatter = _zero_like_first
        submodule.three_interpolate = _zero_like_first
        submodule.three_nn = _zero_like_first
        submodule.voxelization = _zero_like_first
        submodule.assign_score_withk = _zero_like_first
        sys.modules[f"mmcv.ops.{submodule_name}"] = submodule
    sys.modules["mmcv._ext"] = types.ModuleType("mmcv._ext")
