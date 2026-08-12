"""BEVFusion-style backbone, neck, and segmentation head for CG-BEV conditions.

The module exposes a compact encoder interface for conditional BEV features.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "GeneralizedResNet", 
    "LSSFPN", 
    "BEVGridTransform", 
    "BEVSegmentationHead",
    "BEVFusionSegModule",
    "sigmoid_focal_loss"
]


# ============================================================================
# Loss Functions
# ============================================================================

def sigmoid_xent_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Sigmoid cross-entropy loss."""
    inputs = inputs.float()
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Sigmoid focal loss for handling class imbalance.
    
    Args:
        inputs: Predicted logits tensor of shape [B, C, H, W] or [B, C]
        targets: Ground truth tensor of same shape as inputs
        alpha: Weighting factor for positive/negative samples. -1 means no weighting.
        gamma: Focusing parameter for hard examples
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Computed focal loss
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


# ============================================================================
# Building Blocks
# ============================================================================

class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def make_res_layer(
    block: type,
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    stride: int = 1,
) -> nn.Sequential:
    """Create a residual layer with given number of blocks."""
    downsample = None
    if stride != 1 or in_channels != out_channels * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels * block.expansion,
                kernel_size=1, stride=stride, bias=False
            ),
            nn.BatchNorm2d(out_channels * block.expansion),
        )

    layers = []
    layers.append(block(in_channels, out_channels, stride, downsample))
    in_channels = out_channels * block.expansion
    for _ in range(1, num_blocks):
        layers.append(block(in_channels, out_channels))

    return nn.Sequential(*layers)


# ============================================================================
# BEV Decoder Backbone (from bevfusion/mmdet3d/models/backbones/resnet.py)
# ============================================================================

class GeneralizedResNet(nn.ModuleList):
    """
    Generalized ResNet backbone for BEV feature processing.
    
    This is the BEV decoder backbone from BEVFusion that processes the 
    BEV feature map through multiple residual stages.
    
    Default config from camera-bev256d2.yaml:
        in_channels: 80
        blocks:
            - [2, 160, 2]  # (num_blocks, out_channels, stride)
            - [2, 320, 2]
            - [2, 640, 1]
    """
    def __init__(
        self,
        in_channels: int,
        blocks: List[Tuple[int, int, int]],
    ) -> None:
        """
        Args:
            in_channels: Number of input channels
            blocks: List of (num_blocks, out_channels, stride) tuples
        """
        super().__init__()
        self.in_channels = in_channels
        self.blocks = blocks

        for num_blocks, out_channels, stride in self.blocks:
            res_blocks = make_res_layer(
                BasicBlock,
                in_channels,
                out_channels,
                num_blocks,
                stride=stride,
            )
            in_channels = out_channels
            self.append(res_blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Input BEV features [B, C, H, W]
            
        Returns:
            List of feature maps from each stage
        """
        outputs = []
        for module in self:
            x = module(x)
            outputs.append(x)
        return outputs


# ============================================================================
# BEV Decoder Neck (from bevfusion/mmdet3d/models/necks/lss.py)
# ============================================================================

class LSSFPN(nn.Module):
    """
    LSS-style Feature Pyramid Network for BEV features.
    
    This is the BEV decoder neck from BEVFusion that fuses multi-scale 
    features from the backbone.
    
    Default config from camera-bev256d2.yaml:
        in_indices: [-1, 0]
        in_channels: [640, 160]
        out_channels: 256
        scale_factor: 2
    """
    def __init__(
        self,
        in_indices: Tuple[int, int],
        in_channels: Tuple[int, int],
        out_channels: int,
        scale_factor: int = 1,
    ) -> None:
        """
        Args:
            in_indices: Indices of features to fuse (e.g., [-1, 0])
            in_channels: Channels of features to fuse
            out_channels: Output channel dimension
            scale_factor: Upsampling factor
        """
        super().__init__()
        self.in_indices = in_indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels[0] + in_channels[1], out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        if scale_factor > 1:
            self.upsample = nn.Sequential(
                nn.Upsample(
                    scale_factor=scale_factor,
                    mode="bilinear",
                    align_corners=True,
                ),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: List of feature maps from backbone
            
        Returns:
            Fused feature map
        """
        x1 = x[self.in_indices[0]]
        assert x1.shape[1] == self.in_channels[0], \
            f"Expected {self.in_channels[0]} channels, got {x1.shape[1]}"

        x2 = x[self.in_indices[1]]
        assert x2.shape[1] == self.in_channels[1], \
            f"Expected {self.in_channels[1]} channels, got {x2.shape[1]}"

        x1 = F.interpolate(
            x1,
            size=x2.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        x = torch.cat([x1, x2], dim=1)

        x = self.fuse(x)
        if self.scale_factor > 1:
            x = self.upsample(x)
        return x


# ============================================================================
# BEV Grid Transform (from bevfusion/mmdet3d/models/heads/segm/vanilla.py)
# ============================================================================

class BEVGridTransform(nn.Module):
    """
    Transform BEV features from one grid scope to another.
    
    This module handles the spatial transformation between different 
    BEV grid resolutions/ranges.
    
    Default config from default.yaml:
        input_scope: [[-51.2, 51.2, 0.8], [-51.2, 51.2, 0.8]]
        output_scope: [[-50, 50, 0.5], [-50, 50, 0.5]]
    """
    def __init__(
        self,
        *,
        input_scope: List[Tuple[float, float, float]],
        output_scope: List[Tuple[float, float, float]],
        prescale_factor: float = 1,
    ) -> None:
        """
        Args:
            input_scope: List of (min, max, step) for input BEV grid
            output_scope: List of (min, max, step) for output BEV grid  
            prescale_factor: Factor to scale input before transform
        """
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.prescale_factor = prescale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prescale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.prescale_factor,
                mode="bilinear",
                align_corners=False,
            )

        coords = []
        for (imin, imax, _), (omin, omax, ostep) in zip(
            self.input_scope, self.output_scope
        ):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 - 1
            coords.append(v.to(x.device))

        u, v = torch.meshgrid(coords, indexing="ij")
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)

        x = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            align_corners=False,
        )
        return x


# ============================================================================
# BEV Segmentation Head (from bevfusion/mmdet3d/models/heads/segm/vanilla.py)
# ============================================================================

class BEVSegmentationHead(nn.Module):
    """
    BEV Segmentation Head from BEVFusion.
    
    This is the final classifier that outputs segmentation predictions.
    
    Default config from default.yaml:
        in_channels: 256
        classes: [drivable_area, ped_crossing, walkway, stop_line, carpark_area, divider]
        loss: focal
    """
    def __init__(
        self,
        in_channels: int,
        grid_transform: Optional[Dict[str, Any]] = None,
        classes: List[str] = None,
        num_classes: int = 6,
        loss: str = "focal",
    ) -> None:
        """
        Args:
            in_channels: Input channel dimension
            grid_transform: Config for BEVGridTransform (optional)
            classes: List of class names
            num_classes: Number of output classes (used if classes is None)
            loss: Loss type ('xent' or 'focal')
        """
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes if classes is not None else [f"class_{i}" for i in range(num_classes)]
        self.num_classes = len(self.classes)
        self.loss = loss

        if grid_transform is not None:
            self.transform = BEVGridTransform(**grid_transform)
        else:
            self.transform = None
            
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.num_classes, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x: Input BEV features [B, C, H, W]
            target: Ground truth segmentation (only used in training with internal loss)
            
        Returns:
            Logits [B, num_classes, H, W]
        """
        if isinstance(x, (list, tuple)):
            x = x[0]

        if self.transform is not None:
            x = self.transform(x)
        x = self.classifier(x)

        return x  # Return logits directly


# ============================================================================
# Combined BEVFusion Segmentation Module
# ============================================================================

class BEVFusionSegModule(nn.Module):
    """
    Complete BEVFusion segmentation module combining:
    - GeneralizedResNet (BEV decoder backbone)
    - LSSFPN (BEV decoder neck)
    - BEVSegmentationHead (classifier)
    
    This module is designed to replace the simple BevEncode for improved 
    segmentation performance.
    
    Default config from bevfusion camera-bev256d2.yaml:
        decoder:
            backbone:
                in_channels: 80
                blocks: [[2, 160, 2], [2, 320, 2], [2, 640, 1]]
            neck:
                in_indices: [-1, 0]
                in_channels: [640, 160]
                out_channels: 256
                scale_factor: 2
        head:
            in_channels: 256
            classes: 6
            loss: focal
    """
    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 6,
        backbone_blocks: List[Tuple[int, int, int]] = None,
        neck_in_indices: Tuple[int, int] = (-1, 0),
        neck_in_channels: Tuple[int, int] = None,
        neck_out_channels: int = 256,
        neck_scale_factor: int = 2,
        grid_transform: Optional[Dict[str, Any]] = None,
        classes: List[str] = None,
        loss: str = "focal",
    ) -> None:
        """
        Args:
            in_channels: Input BEV feature channels
            num_classes: Number of segmentation classes
            backbone_blocks: ResNet block config [(num_blocks, out_channels, stride), ...]
            neck_in_indices: Indices of backbone features to fuse
            neck_in_channels: Channels of features to fuse
            neck_out_channels: Output channels from neck
            neck_scale_factor: Upsampling factor in neck
            grid_transform: Config for BEVGridTransform
            classes: List of class names
            loss: Loss type for segmentation head
        """
        super().__init__()
        
        # Default backbone config from BEVFusion camera-bev256d2.yaml
        if backbone_blocks is None:
            backbone_blocks = [
                (2, 160, 2),
                (2, 320, 2),
                (2, 640, 1),
            ]
        
        # Default neck channels matching backbone
        if neck_in_channels is None:
            neck_in_channels = (640, 160)  # (last stage, first stage)
            
        self.backbone = GeneralizedResNet(
            in_channels=in_channels,
            blocks=backbone_blocks,
        )
        
        self.neck = LSSFPN(
            in_indices=neck_in_indices,
            in_channels=neck_in_channels,
            out_channels=neck_out_channels,
            scale_factor=neck_scale_factor,
        )
        
        self.head = BEVSegmentationHead(
            in_channels=neck_out_channels,
            grid_transform=grid_transform,
            classes=classes,
            num_classes=num_classes,
            loss=loss,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input BEV features [B, in_channels, H, W]
            target: Ground truth segmentation (optional, for training loss)
            
        Returns:
            Logits [B, num_classes, H, W]
        """
        # Backbone: extract multi-scale features
        features = self.backbone(x)
        
        # Neck: fuse features
        x = self.neck(features)
        
        # Head: predict segmentation
        logits = self.head(x)
        
        return logits


class BEVFusionEncoder(nn.Module):
    """Adapt BEVFusionSegModule to the CG-BEV encoder interface."""
    def __init__(
        self,
        inC: int,
        outC: int,
        backbone_blocks: List[Tuple[int, int, int]] = None,
        neck_out_channels: int = 256,
        neck_scale_factor: int = 2,
    ) -> None:
        """
        Args:
            inC: Input channels (matches BevEncode's inC)
            outC: Output channels (matches BevEncode's outC, i.e., num_classes)
            backbone_blocks: Custom backbone config (optional)
            neck_out_channels: Neck output channels
            neck_scale_factor: Upsampling factor in neck
        """
        super().__init__()
        
        # Use default BEVFusion config if not specified
        if backbone_blocks is None:
            backbone_blocks = [
                (2, 160, 2),
                (2, 320, 2),
                (2, 640, 1),
            ]
        
        # Compute neck input channels from backbone config
        neck_in_channels = (backbone_blocks[-1][1], backbone_blocks[0][1])  # (last, first)
        
        self.seg_module = BEVFusionSegModule(
            in_channels=inC,
            num_classes=outC,
            backbone_blocks=backbone_blocks,
            neck_in_indices=(-1, 0),
            neck_in_channels=neck_in_channels,
            neck_out_channels=neck_out_channels,
            neck_scale_factor=neck_scale_factor,
            grid_transform=None,
            loss="focal",
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input BEV features [B, inC, H, W]
            
        Returns:
            Segmentation logits [B, outC, H, W]
        """
        return self.seg_module(x)
