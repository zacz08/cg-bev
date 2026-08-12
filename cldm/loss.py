import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

# ===== 1. Centerness Head =====
class CenternessHead(nn.Module):
    def __init__(self, in_channels, out_channels=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1),  # 4 classes
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(x)  # [B, 4, H, W]


# ===== 2. Offset Head =====
class OffsetHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 8, kernel_size=1),  # 4 classes * 2 directions
        )

    def forward(self, x):
        B, _, H, W = x.shape
        out = self.head(x)
        return out.view(B, 4, 2, H, W)  # [B, 4, 2, H, W]
    
    
class SturctureNeck(nn.Module):
    def __init__(self, in_channels=4, out_channels=64):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(out_channels, out_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(out_channels // 2, out_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        return self.upsample(x)  # output shape: [B, ~16, 512, 512]



# ===== 3. Centerness Loss =====
class CenternessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target_mask, weights):
        """
        pred: [B, 4, H, W] after sigmoid
        target_mask: [B, 4, H, W] binary mask in [-1, 1]
        weights: [B, 4]
        """
        B, C, H, W = target_mask.shape

        # Normalize target_mask to [0,1]
        if target_mask.min() < 0:
            target_mask = (target_mask + 1) / 2

        target_centerness = torch.zeros_like(target_mask, dtype=torch.float32)

        for b in range(B):
            for c in range(C):
                mask_np = target_mask[b, c].cpu().numpy().astype(np.uint8)
                if mask_np.sum() == 0:
                    continue
                dist = distance_transform_edt(mask_np)
                dist = dist / (dist.max() + 1e-6)
                dist = dist.astype(np.float32)
                target_centerness[b, c] = torch.from_numpy(dist).to(target_mask.device)


        # BCE loss with weighting [B, 4, H, W] → [B, 4]
        bce = F.binary_cross_entropy(pred, target_centerness, reduction='none')  # [B, 4, H, W]
        bce = bce.mean(dim=(2, 3))  # → [B, 4]
        loss = (bce * weights).sum(dim=1).mean()  # → scalar
        return loss


# ===== 4. Offset Loss =====
class OffsetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target_mask, weights):
        """
        pred: [B, 4, 2, H, W]
        target_mask: [B, 4, H, W] binary mask in [-1, 1]
        weights: [4] (global per-layer weights)
        """
        B, C, _, H, W = pred.shape

        # Normalize target_mask to [0,1]
        if target_mask.min() < 0:
            target_mask = (target_mask + 1) / 2

        gt_offset = torch.zeros_like(pred)
        valid_mask = torch.zeros((B, C, 1, H, W), dtype=torch.float32, device=pred.device)

        # Compute per-object center and create offset and valid mask
        for b in range(B):
            for c in range(C):
                mask_np = target_mask[b, c].cpu().numpy().astype(np.uint8)
                if mask_np.sum() == 0:
                    continue  # Skip if no positive pixels in this class
                ys, xs = np.nonzero(mask_np)
                cx, cy = xs.mean(), ys.mean()

                if not (0 <= cx < W and 0 <= cy < H):
                    print(f"[WARN] cx={cx:.2f}, cy={cy:.2f} out of bounds at b={b}, c={c}")
                    continue  # Skip if the center is out of bounds

                grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                grid_x = grid_x.astype(np.float32)
                grid_y = grid_y.astype(np.float32)

                offset_x = (cx - grid_x) * mask_np
                offset_y = (cy - grid_y) * mask_np
                offset = np.stack([offset_x, offset_y]).astype(np.float32)

                gt_offset[b, c] = torch.from_numpy(offset).to(dtype=torch.float32, device=pred.device)
                valid_mask[b, c] = torch.from_numpy(mask_np).to(pred.device).unsqueeze(0)


        # -------- Vectorized loss calculation --------
        # pred: [B, 4, 2, H, W], gt_offset: [B, 4, 2, H, W], valid_mask: [B, 4, 1, H, W]
        mask = valid_mask
        norm = mask.sum(dim=(2, 3, 4))  # [B, 4], number of valid pixels per sample and class

        # Only keep entries with positive pixels
        valid = norm > 0  # [B, 4], bool

        pred_masked = pred * mask
        gt_masked = gt_offset * mask

        # Compute per-pixel L1 loss (Smooth L1)
        diff = F.smooth_l1_loss(pred_masked, gt_masked, reduction='none')  # [B, 4, 2, H, W]
        diff = diff.sum(dim=2)  # sum offset directions, [B, 4, H, W]
        diff = diff.sum(dim=(2, 3))  # sum spatial dims, [B, 4]

        # Normalize loss per (batch, class)
        layer_loss = torch.zeros_like(diff)
        layer_loss[valid] = diff[valid] / (norm[valid] + 1e-6)

        # Apply per-layer global weights: weights shape [4] -> broadcast to [B, 4]
        weighted_loss = layer_loss * weights[None, :]  # [B, 4]

        if valid.sum() > 0:
            return weighted_loss[valid].mean()
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits, targets, weights):
        targets = (targets + 1) / 2  # Normalized to [0,1]
        bce_loss = self.bce(logits, targets).mean(dim=(2, 3))
        
        # apply per-layer weights
        # weighted_loss = weights[:, :, None, None] * bce_loss
        weighted_loss = (bce_loss * weights).sum(dim=1)
        return weighted_loss.mean()


class WeightedMSELoss(nn.Module):
    """Per-layer-weighted MSE on binary segmentation masks.

    The CLDM VAE in this repo is configured with ``tanh_out=False`` (see
    ``configs/cldm_res_192.yaml``), so the decoder output ``pred`` is an
    UNBOUNDED logit tensor (typical magnitude ±5..±20). Computing MSE on the
    raw logits — as the previous ``(pred + 1) / 2`` path effectively did —
    yields huge non-saturating errors that swamp ``loss_simple`` and collapse
    the diffusion prior (the only gradient path is through the frozen VAE
    decoder, which forces the latents off-distribution).

    Two activations are supported to bound ``pred`` first:
      - ``activation='sigmoid'`` (default): regress in [0, 1] probability
        space (target is mapped from [-1, 1] to [0, 1]). Same space as the
        Dice / BCEWithLogits branches, so the two branches don't pull
        against each other.
      - ``activation='tanh'``: regress in [-1, 1] (target stays in [-1, 1]).
        Matches the appendix VAE setup (tanh_out=True + MSE) literally.
        Note this makes the per-pixel error ~4× larger than the sigmoid
        path (`[-1,1]` vs `[0,1]` × squared), so you typically want to
        halve ``alpha`` (or the outer ``loss_seg *= 2.0`` in cldm.py) when
        switching.

    ``activation='none'`` reproduces the legacy behaviour for a tanh-output
    VAE (``tanh_out=True``), where ``pred`` is already in [-1, 1].
    """

    def __init__(self, activation='tanh'):
        super().__init__()
        if activation not in ('sigmoid', 'tanh', 'none'):
            raise ValueError(f"unknown activation '{activation}'")
        self.activation = activation

    def forward(self, pred, targets, weights):
        if self.activation == 'sigmoid':
            pred_b = torch.sigmoid(pred)              # logits -> [0, 1]
            targets_b = (targets + 1) / 2              # [-1, 1] -> [0, 1]
        elif self.activation == 'tanh':
            pred_b = torch.tanh(pred)                  # logits -> [-1, 1]
            targets_b = targets                        # already [-1, 1]
        else:  # 'none' — predictor already bounded (legacy tanh_out=True VAE)
            pred_b = pred
            targets_b = targets

        mse = F.mse_loss(pred_b, targets_b, reduction='none').mean(dim=(2, 3))  # [B, C]
        weighted_loss = (mse * weights).sum(dim=1)
        return weighted_loss.mean()

class WeightedDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, weights):
        # logits = (logits + 1) / 2
        logits = torch.sigmoid(logits)  # Apply sigmoid to logits, make sure tanh_out=False in VAE config
        targets = (targets + 1) / 2 # Normalized to [0,1]
        
        intersection = (logits * targets).sum(dim=(2, 3))
        union = logits.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth) # shape: [B, C]
        
        # apply per-layer weights
        # weighted_dice_loss = weights[:, :, None, None] * (1 - dice_score)
        dice_loss = (1 - dice_score) * weights
        weighted_dice_loss = dice_loss.sum(dim=1)  # Sum over channels

        return weighted_dice_loss.mean()

class SegmentationLoss(nn.Module):
    """alpha * weighted-MSE + beta * weighted-Dice.

    The original BCE-with-logits branch was sketchy here: the "logits" being
    passed in are actually post-tanh outputs of the VAE decoder (range
    [-1, 1]), which BCEWithLogits then silently sigmoids. Replacing it with
    weighted MSE on the [0, 1]-mapped tensors removes that mismatch and gives
    the Dice branch a complementary, well-conditioned regression signal.
    Set ``mask_loss='bce'`` to fall back to the legacy behaviour.
    """

    def __init__(self, alpha=1, beta=0.2, mask_loss='bce', mse_activation='sigmoid'):
        super(SegmentationLoss, self).__init__()
        if mask_loss == 'mse':
            self.mask_loss = WeightedMSELoss(activation=mse_activation)
        elif mask_loss == 'bce':
            self.mask_loss = WeightedBCEWithLogitsLoss()
        else:
            raise ValueError(f"unknown mask_loss '{mask_loss}', expected 'mse' or 'bce'")
        self.dice = WeightedDiceLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets, weights):
        return self.alpha * self.mask_loss(logits, targets, weights) + \
                self.beta * self.dice(logits, targets, weights)
 

def compute_layer_weights(inputs: torch.Tensor, epsilon=1e-6, clamp_min=1e-3, clamp_max=10.0):
    """
    inputs: [bs, 4, H, W], mask in [-1, 1]
    returns: [4,] global rec_weight for each layer in batch level
    """
    bs, num_layers, h, w = inputs.shape
    total_pixel = bs * h * w


    mask_bin = (inputs > 0).float()  # [bs, 4, H, W]
    layer_pixel = mask_bin.sum(dim=(0, 2, 3))  # [4]
    layer_pixel = torch.clamp(layer_pixel, min=1.0)

    raw_weights = torch.log(total_pixel / layer_pixel + epsilon)
    raw_weights = raw_weights.clamp(max=clamp_max) 

    weights = raw_weights / (raw_weights.sum() + epsilon) * num_layers

    return weights  # shape: [4]


def compute_rec_weights(inputs: torch.Tensor, epsilon=1e-6, clamp_min=1e-3, clamp_max=10.0):
    """
    inputs: [bs, 4, H, W] binary mask in [-1, 1]
    returns: [bs, 4] rec_weight for each layer
    """
    bs, num_layers, h, w = inputs.shape
    total_pixel = h * w

    layer_pixel = (inputs > 0).sum(dim=(2, 3)).float()  # [bs, 4]
    layer_pixel = torch.clamp(layer_pixel, min=5.0)

    raw_weights = torch.log(total_pixel / layer_pixel + epsilon)
    raw_weights = raw_weights.clamp(max=10.0) ** 2

    weights = raw_weights / (raw_weights.sum(dim=1, keepdim=True) + epsilon)
    weights = weights.clamp(min=clamp_min, max=clamp_max)

    return weights

# Lovasz hinge loss for binary mask per channel, vectorized batch version
def lovasz_hinge_flat(logits, labels):
    # logits: [P] (float, logits), labels: [P] (float, 0/1)
    if logits.numel() == 0:
        return logits.sum() * 0.
    signs = 2. * labels - 1.
    errors = 1. - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    grad = grad.to(errors_sorted.dtype)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

def lovasz_grad(gt_sorted):
    p = gt_sorted.sum()
    if p == 0:
        return gt_sorted
    intersection = p - gt_sorted.float().cumsum(0)
    union = p + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if gt_sorted.numel() == 1:
        return gt_sorted * 0. + 1.
    grad = jaccard
    grad[1:] = grad[1:] - grad[:-1]
    return grad

class LovaszHingeLossMultiChannel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logits, targets, weights):
        """
        logits: [B, C, H, W] (raw, not sigmoid/tanh)
        targets: [B, C, H, W] in [-1,1]
        weights: [B, C] or [C]
        """
        targets = (targets + 1) / 2  # to [0,1]
        B, C, H, W = logits.shape
        losses = torch.zeros((B, C), dtype=logits.dtype, device=logits.device)
        # calculate per-channel loss
        for b in range(B):
            for c in range(C):
                logit_flat = logits[b, c].reshape(-1)
                label_flat = targets[b, c].reshape(-1)
                losses[b, c] = lovasz_hinge_flat(logit_flat, label_flat)
        # weights: [B, C] or [C]
        if weights.dim() == 1:
            weights = weights.unsqueeze(0).expand(B, C)  # [B, C]
        weighted_loss = losses * weights

        return weighted_loss.sum(dim=1).mean()

