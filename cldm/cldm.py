import einops
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ldm.modules.diffusionmodules.util import conv_nd, zero_module
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential
from ldm.diffusion.ddm_const import LatentDiffusion
from ldm.util import instantiate_from_config
from cldm.bevencoder import BevEncode, MultiScaleBevEncode
from cldm.bevfusion_seghead import BEVFusionEncoder, sigmoid_focal_loss  # BEVFusion-style encoder
from cldm.loss import (
    compute_layer_weights,
    LovaszHingeLossMultiChannel,
)

## Controlnet dependencies
from ldm.modules.diffusionmodules.adm_unet import (
    DhariwalUNet, 
    UNetBlock, 
    Linear, 
    PositionalEmbedding,
    SpatialAtt,
    Conv2d,
    )
from torch.nn.functional import silu

class ControledDhariwalUNet(DhariwalUNet):
    def forward(self, x, noise_labels, class_labels, augment_labels=None, 
                control=None, only_mid_control=False, **kwargs):
        # Mapping.
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)
        
        ## Following Controlnet source code
        # Encoder (input_blocks)
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)
        
        # middle block
        x = self.decouple(x) + x
        # if control is not None and only_mid_control:
        if control is not None:
            x = x + control.pop()
        
        # Decoder (output_blocks)
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                if control is not None and not only_mid_control:
                    x = torch.cat([x, skips.pop() + control.pop()], dim=1)
                else:
                    x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        
        return self.out(x)
    

class ControlledUnetModel(nn.Module):
    def __init__(self,
        img_resolution,                     # latent img resolution at input/output.
        img_channels,                       # channel of latent img.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.
        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
        dropout             = 0.10,         # List of resolutions with self-attention.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
        out_mul             = 1,            # output channel multiplier
        precondition = True,
    ):
        super().__init__()
        self.precondition = precondition
        self.label_dim = label_dim
        self.channels = img_channels
        self.unet = globals()['ControledDhariwalUNet'](
            img_resolution, 
            img_channels, 
            img_channels,
            label_dim,
            augment_dim,
            model_channels,
            channel_mult,
            channel_mult_emb,
            num_blocks,
            attn_resolutions,
            dropout,
            label_dropout,
            out_mul)

    def forward(self, x, sigma, control, only_mid_control=False, **kwargs):
        # here sigma = t;
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float32

        c_skip = (sigma - 1) / (sigma ** 2 - sigma + 1)
        c_out = torch.sqrt(sigma / (sigma ** 2 - sigma + 1))
        c_in = 1 / torch.sqrt((1 - sigma) ** 2 + sigma)
        c_noise = sigma.log()

        F_x = self.unet((c_in * x).to(dtype), 
                        c_noise.flatten(), 
                        class_labels,
                        control=control,
                        only_mid_control=only_mid_control)
        # assert F_x.dtype == dtype
        if self.precondition:
            D_x = c_skip * x + c_out * F_x
            D_y = (x - (sigma - 1) * D_x) / sigma.sqrt()
        else:
            D_x = F_x
            D_y = (x - (sigma - 1) * D_x) / sigma.sqrt()
        return D_x, D_y

class ControlNet(nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of channels at input.
        bev_encoder_in,
        bev_encoder_out,
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.
        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
        dropout             = 0.10,         # List of resolutions with self-attention.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
        out_mul             = 1,            # output channel multiplier
        dims                = 2,
        use_fp16            = False,
        semantic_layers     = 4,  # number of semantic layers
        **kwargs
    ):
        
        super().__init__()
        self.label_dim = label_dim
        self.label_dropout = label_dropout
        self.dtype = torch.float16 if use_fp16 else torch.float32
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero)


        self.dims = dims

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None

        ## Encoder (input_blocks)
        # self.enc = nn.ModuleList()
        # self.zero_convs = nn.ModuleList()
        self.enc = nn.ModuleDict()
        self.zero_convs = nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
                # self.enc.append(Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init))
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                # self.enc.append(UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs))
            # self.zero_convs.append(self.make_zero_conv(cout))
            self.zero_convs[f'{res}x{res}_zero'] = self.make_zero_conv(cout)

            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
                # self.enc.append(UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs))
                # self.zero_convs.append(self.make_zero_conv(cout))
                self.zero_convs[f'{res}x{res}_block{idx}_zero'] = self.make_zero_conv(cout)
        # skips = [block.out_channels for block in self.input_blocks.values()]
   
        
        ## input hint block
        gt_resolution = kwargs.get('gt_image_size', 512)
        if gt_resolution == 192:
            self.input_hint_block = nn.Sequential(      # [bs, 6, 192, 192] -> [bs, 128, 24, 24]
                conv_nd(dims, bev_encoder_out + semantic_layers, 48, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 48, 64, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 64, 64, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 64, 96, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 96, 96, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 96, model_channels, 3, padding=1, stride=2),
                nn.SiLU(),
                zero_module(conv_nd(dims, model_channels, model_channels, 3, padding=1))
                )
        elif gt_resolution == 512:
             self.input_hint_block = nn.Sequential(      # [bs, 6 + 8, 192, 192] -> [bs, 128, 64, 64]
                nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False),
                conv_nd(dims, bev_encoder_out + semantic_layers, 32, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 32, 32, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 32, 64, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 64, 64, 3, padding=1),
                nn.SiLU(),
                zero_module(conv_nd(dims, 64, model_channels, 3, padding=1, stride=2))
                )
        ## middle block (decouple)
        self.decouple = nn.Sequential(
            nn.Conv2d(cout, cout, 3, 1, 1),
            SpatialAtt(cout)
        )
        self.decouple_out = self.make_zero_conv(cout)

        # Select the configured BEV encoder.
        self.use_multiscale = kwargs.get('use_multiscale', False)
        self.use_bevfusion  = kwargs.get('use_bevfusion', False)
        if self.use_multiscale:
            self.bevencode = MultiScaleBevEncode(
                bev_encoder_in, bev_encoder_out,
                hidden_dim=kwargs.get('multiscale_hidden_dim', 32),
                dilations=tuple(kwargs.get('multiscale_dilations', [1, 3, 9])),
            )
        elif self.use_bevfusion:
            self.bevencode = BEVFusionEncoder(bev_encoder_in, bev_encoder_out)
        else:
            self.bevencode = BevEncode(bev_encoder_in, bev_encoder_out)

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def time_embed(self, x, noise_labels, class_labels):
        emb = self.map_noise(noise_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        return silu(emb)

    def forward(self, x, hint, timesteps, **kwargs):
        x = x.to(self.dtype)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        sigma = timesteps.to(self.dtype).reshape(-1, 1, 1, 1)
        c_noise = sigma.log()
        noise_labels = c_noise.flatten()
        emb = self.time_embed(x, noise_labels, class_labels)

        # if hint.shape[-2:] == (200, 200):
        #     hint = F.interpolate(hint, size=(192, 192), mode='nearest')
        guided_hint = self.input_hint_block(hint)

        outs = []
        h = x
        # for module, zero_conv in zip(self.enc, self.zero_convs):
        for module, zero_conv in zip(self.enc.values(), self.zero_convs.values()):
            h = module(h, emb) if isinstance(module, UNetBlock) else module(x)
            if guided_hint is not None:
                h += guided_hint
                guided_hint = None
            outs.append(zero_conv(h, emb))

        h = self.decouple(h)
        outs.append(self.decouple_out(h, emb))

        return outs


class CGBEV(LatentDiffusion):

    def __init__(self, control_stage_config, 
                 control_key, only_mid_control, reference_key, 
                 use_lovasz_loss=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.reference_key = reference_key
        self.only_mid_control = only_mid_control
        self.use_lovasz_loss = use_lovasz_loss
        self.loss_lovasz = LovaszHingeLossMultiChannel() if self.use_lovasz_loss else None
        
        # Use sigmoid_focal_loss instead of BCE+Dice loss (like vggt/BEVFusion)
        # Set to True to use focal loss, False to use original BCE+Dice loss
        self.use_focal_loss = False
        self.focal_alpha = -1   # BEVFusion default: -1 (no alpha weighting)
        self.focal_gamma = 2.0  # BEVFusion default: 2.0
        
        if self.use_focal_loss:
            print("[CG-BEV] Using sigmoid_focal_loss for segmentation loss.")
    
    def compute_seg_loss(self, img_rec, img_ori, rec_weight=None):
        """
        Compute segmentation loss using focal loss or original BCE+Dice loss.
        
        Args:
            img_rec: Reconstructed image (VAE decoder output), shape [B, C, H, W]
                     - When tanh_out=False: unbounded logits (good for focal loss)
                     - When tanh_out=True: [-1, 1] range (needs conversion)
            img_ori: Ground truth image, shape [B, C, H, W], range [-1, 1]
            rec_weight: Per-layer weights (used only for original loss)
            
        Returns:
            Computed loss
        """
        if self.use_focal_loss:
            # Normalize ground truth from [-1, 1] to [0, 1] for focal loss
            target = (img_ori + 1) / 2  # [-1, 1] -> [0, 1]
            
            # img_rec is logits (tanh_out=False in current config)
            # If tanh_out=True, need to convert from [-1, 1] to logits
            # Check if values are bounded (indicating tanh output)
            if img_rec.min() >= -1.1 and img_rec.max() <= 1.1:
                # Likely tanh output, convert to logits using inverse sigmoid
                # First map from [-1, 1] to [0, 1], then to logits
                img_rec_prob = (img_rec + 1) / 2
                img_rec_prob = img_rec_prob.clamp(1e-6, 1 - 1e-6)  # Avoid log(0)
                logits = torch.log(img_rec_prob / (1 - img_rec_prob))  # inverse sigmoid
            else:
                # Already logits (unbounded)
                logits = img_rec
            
            # Compute per-class focal loss and average
            total_focal_loss = 0.0
            num_classes = logits.shape[1]
            
            for i in range(num_classes):
                class_pred = logits[:, i]  # [B, H, W]
                class_target = target[:, i]  # [B, H, W]
                focal_loss = sigmoid_focal_loss(
                    class_pred, class_target,
                    alpha=self.focal_alpha, gamma=self.focal_gamma
                )
                total_focal_loss += focal_loss
            
            return total_focal_loss
        else:
            # Use original BCE+Dice loss from parent class
            return self.loss_seg_func(img_rec, img_ori, rec_weight)

    def train_dataloader(self):
        return super().train_dataloader(return_feature=True)
    
    def val_dataloader(self):
        return super().val_dataloader(return_feature=True)
        

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):

        # return encoded input (gt_jpg or bev_feat), dict(encoded_text, condition)
        z, c, ref = super().get_input(batch, self.first_stage_key, *args, **kwargs)

        if bs is not None:
            z = z[:bs]
            ref = ref[:bs]

        if self.control_key in batch:
            control = batch[self.control_key]
            if bs is not None:
                control = control[:bs]
            control = control.to(self.device)
            control = einops.rearrange(control, 'b h w c -> b c h w')
            control = control.to(memory_format=torch.contiguous_format).float()

            ## Specific setting for "bev_feat + pred_map" style condition
            control = control if control is not None else ref
        else:
            control = None
            
        return z, dict(c_crossattn=[c], c_concat=[control], c_predmap=[ref]), ref

    @torch.no_grad()
    def get_condition(self, batch, bs=None):
        if self.reference_key is not None:
            ref = super(LatentDiffusion, self).get_input(batch, self.reference_key)
            if bs is not None:
                ref = ref[:bs]
            ref = ref.to(self.device)
        else:
            ref = super(LatentDiffusion, self).get_input(batch, self.first_stage_key)
            if bs is not None:
                ref = ref[:bs]
            ref = ref.to(self.device)

        if self.control_key in batch:
            control = batch[self.control_key]
            if bs is not None:
                control = control[:bs]
            control = control.to(self.device)
            control = einops.rearrange(control, 'b h w c -> b c h w')
            control = control.to(memory_format=torch.contiguous_format).float()
            control = control if control is not None else ref
        else:
            control = None

        return dict(c_crossattn=[None], c_concat=[control], c_predmap=[ref]), ref

    def apply_model(self, x_noisy, t, cond, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model    # ControlledUnetModel
        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, sigma=t, control=None, only_mid_control=self.only_mid_control)
        else:
            # output from control model (list of each Controlnet block's output)
            bev_feat = torch.cat(cond['c_concat'], 1)
            pred_map = torch.cat(cond['c_predmap'], 1)
            # Align bev_feat spatial size to pred_map so the encoder output
            # can be concatenated with pred_map. Backbones produce different
            # native sizes (e.g. lss=200, bevfusion=128) while pred_map is
            # either backbone-native (res=512) or resized to 192 (res=192).
            if bev_feat.shape[-2:] != pred_map.shape[-2:]:
                bev_feat = F.interpolate(bev_feat, size=pred_map.shape[-2:],
                                         mode='bilinear', align_corners=False)
            # Use single BEVFusion encoder
            bev_feat_encoded = self.control_model.bevencode(bev_feat)
            concat_hint = torch.cat([pred_map, bev_feat_encoded], dim=1)
            control = self.control_model(x=x_noisy, hint=concat_hint, timesteps=t)
            # control = [c * scale for c, scale in zip(control, self.control_scales)]
            phi, eps = diffusion_model(x=x_noisy, sigma=t, control=control, only_mid_control=self.only_mid_control, **kwargs)
        return phi, eps # C_pred, noise_pred
    
    def p_losses(self, x_start, t, cond, *args, **kwargs):
        if self.start_dist == 'normal':
            noise = torch.randn_like(x_start)
        elif self.start_dist == 'uniform':
            noise = 2 * torch.rand_like(x_start) - 1.
        else:
            raise NotImplementedError(f'{self.start_dist} is not supported !')

        C = -1 * x_start             # U(t) = Ct, U(1) = -x0
        x_noisy = self.q_sample(x_start=x_start, noise=noise, t=t, C=C)  # (b, 2, c, h, w)
        C_pred, noise_pred = self.apply_model(x_noisy, t, cond, **kwargs)
        x_rec = self.pred_x0_from_xt(x_noisy, noise_pred, C_pred, t)
        loss_dict = {}
        prefix = kwargs.get('prefix', 'train')

        target1 = C
        target2 = noise
        loss = 0.

        loss_simple = F.mse_loss(C_pred, target1, reduction='mean') + \
                        F.mse_loss(noise_pred, target2, reduction='mean')
        loss += loss_simple

        ## Segmentation loss
        img_ori = kwargs['batch'][self.first_stage_key].permute(0, 3, 1, 2)
        # rec_weight = compute_rec_weights(img_ori)
        rec_weight = compute_layer_weights(img_ori)
        img_rec = self.decode_first_stage(x_rec)
        # Focal loss is ~10x smaller than BCE+Dice, so scale up from 0.8 to 8.0
        loss_seg = self.compute_seg_loss(img_rec, img_ori, rec_weight) * 0.8
        loss += loss_seg

        if self.use_lovasz_loss:
            loss_lovasz = self.loss_lovasz(img_rec, img_ori, rec_weight) * 0.05
            loss += loss_lovasz
        else:
            loss_lovasz = loss_simple.new_zeros(())

        loss_dict.update({f'{prefix}/loss_simple': loss_simple.detach()})
        loss_dict.update({f'{prefix}/loss_seg': loss_seg.detach()})
        loss_dict.update({f'{prefix}/loss_lovasz': loss_lovasz.detach()})
        loss_dict.update({f'{prefix}/loss': loss.detach()})

        return loss, loss_dict
    
    # Keys/kwargs are kept identical across the early-skip branch and the
    # active branch so that PyTorch Lightning's per-metric metadata check
    # (prog_bar / on_step / on_epoch / logger / sync_dist) does not trip
    # `MisconfigurationException: ... twice in validation_step with different
    # arguments` when crossing the `val_after_epoch` threshold.
    _VAL_LOG_KWARGS = dict(prog_bar=True, logger=True,
                           on_step=False, on_epoch=True, sync_dist=True)
    _VAL_LOSS_KEYS = ('val/loss', 'val/loss_simple',
                      'val/loss_seg', 'val/loss_lovasz')

    def validation_step(self, batch, **kwargs):
        val_epoch = self.opt_cfg.get('val_after_epoch', 0)

        if self.current_epoch < val_epoch:
            nan = torch.tensor(float("nan"), device=self.device)
            nan_loss_dict = {k: nan for k in self._VAL_LOSS_KEYS}
            self.log_dict(nan_loss_dict, **self._VAL_LOG_KWARGS)
            self.log("val/IoU", nan, **self._VAL_LOG_KWARGS)
            return

        loss, loss_dict = self.shared_step(batch, prefix='val', **kwargs)
        self.log_dict(loss_dict, **self._VAL_LOG_KWARGS)
        self.predict_step(batch)

        return loss

    def on_validation_epoch_end(self):
        val_epoch = self.opt_cfg.get('val_after_epoch', 0)

        if self.current_epoch < val_epoch:
            return

        score = self.first_stage_model.seg_metric.compute()
        iou = score.mean().item()
        self.log('val/IoU', iou, **self._VAL_LOG_KWARGS)
        self.first_stage_model.seg_metric.reset()
    
    
    @torch.no_grad()
    def predict_step(self, batch, dataloader_idx = 0):   
        gt = batch['bev_map_gt']
        gt_mask = (gt + 1) / 2  # [-1, 1] -> [0, 1]
        gt_mask = gt_mask.permute(0, 3, 1, 2)   # b h w c-> b c h w
        batch_size = gt.shape[0]

        # get reconstruction
        c, _ = self.get_condition(batch)
        recon = self.sample(batch_size=batch_size, cond=c)

        recon_mask = torch.where(recon > 0, 1, 0)

        # pred = batch[self.control_key]
        # pred = (pred + 1) / 2  # [-1, 1] -> [0, 1]
        # pred = pred.permute(0, 3, 1, 2)   # b h w c-> b c h w
        # pred = F.interpolate(pred, size=(192, 192), mode='bilinear', align_corners=False)

        self.first_stage_model.seg_metric.update(recon_mask, gt_mask)


    def on_predict_epoch_end(self):
        score = self.first_stage_model.seg_metric.compute()
        iou_sum = 0.0
        num_layers = len(self.first_stage_model.semantic_layers)

        for index, layer in enumerate(self.first_stage_model.semantic_layers):
            iou_layer = score[index].item()
            iou_sum += iou_layer
            print(f"IoU {layer}: {iou_layer:.5f}")
        
        mean_iou = iou_sum / num_layers
        print(f"Average IoU: {mean_iou:.5f}")
        self.first_stage_model.seg_metric.reset()


    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=True, **kwargs):

        log = dict()
        N = min(batch[self.first_stage_key].shape[0], N)
        z, c, _ = self.get_input(batch, self.first_stage_key, bs=N)

        # log["recon_sd_input"] = self.decode_first_stage(z)
        log["gt"] = batch[self.first_stage_key][:N].permute(0, 3, 1, 2)
        log["recon_ref"] = batch[self.reference_key][:N].permute(0, 3, 1, 2)
        original_shape = log["gt"].shape[-2:]
        if log["recon_ref"].shape[-2:] != original_shape:
            log["recon_ref"] = F.interpolate(log["recon_ref"], size=original_shape, mode='bilinear', align_corners=False)

        if sample:
            log["samples"] = self.sample(batch_size=N, cond=c)

        return log
    
    
    def get_optim_params(self):
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.unet.dec.parameters())
            params += list(self.model.unet.out.parameters())

        return params
