import os
import csv
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from tools.metrics import IntersectionOverUnion

from nuScenesSegDataset import nuScenesSegDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from cldm.loss import compute_layer_weights
from tools.training_log_analysis import parse_csv_and_plot


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key=['image'],
                 colorize_nlabels=None,
                 monitor=None,
                 learn_logvar=False,
                 kl_div_weight=0.1,
                 rec_weight=[1.0, 1.0, 1.0, 1.0],
                 semantic_layers=['vehicle'],
                 vae_training_cfg = None
                 ):
        super().__init__()
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.ddconfig = ddconfig
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.control_key = 'None'
        if vae_training_cfg is not None:
            self.opt_cfg = vae_training_cfg.get('trainer_config', None)
            self.data_cfg = vae_training_cfg.get('data_config', None)
            self.use_scheduler = self.opt_cfg is not None
        self.loss = VAELoss(kl_div_weight)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if self.eval:
            n_classes = len(semantic_layers)
            self.semantic_layers = semantic_layers
            self.seg_metric = IntersectionOverUnion(n_classes).to(self.device)

        self.latent_collection = []

    def image_keys(self):
        if isinstance(self.image_key, str):
            return [self.image_key]
        return list(self.image_key)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def train_dataloader(self):
        assert self.data_cfg is not None, "VAE training config is not provided"
        dataset = nuScenesSegDataset(data_split=self.data_cfg['data_split_train'],
                                     resolution=self.data_cfg['resolution'],
                                     augment=self.data_cfg['augment'])

        if self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=self.trainer.world_size, rank=self.trainer.global_rank, shuffle=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(dataset,
                          batch_size=self.data_cfg['batch_size'],
                          shuffle=shuffle,
                          sampler=sampler,
                          num_workers=self.data_cfg['num_workers'])

    def val_dataloader(self):
        assert self.data_cfg is not None, "VAE training config is not provided"
        dataset = nuScenesSegDataset(data_split=self.data_cfg['data_split_val'],
                                     resolution=self.data_cfg['resolution'],)

        if self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=self.trainer.world_size, rank=self.trainer.global_rank, shuffle=False)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(dataset,
                          batch_size=self.data_cfg['batch_size'],
                          sampler=sampler,
                          shuffle = shuffle)

    def on_predict_epoch_end(self):
        score = self.seg_metric.compute()
        for index, layer in enumerate(self.semantic_layers):
            print(f"IoU {layer}: {score[index].item():.5f}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch, **kwargs):
        # losses, recon_losses, kl_divs = [], [], []
        inputs_list = []
        rec_weights_list = []

        for k in self.image_keys():
            inputs = self.get_input(batch, k)
            # reconstructions, posterior = self(inputs)

            # bs, num_layers, h, w = inputs.shape
            # total_pixel = h * w
            # layer_pixel = ((inputs > 0.99) & (inputs < 1.01)).sum(dim=(2, 3))    # calculate the sum of pixel values of each layer
            # # pixel_ratio = layer_pixel / total_pixel  # [bs, 4]
            # # weights = 1 / (pixel_ratio + 1e-6)
            # # weights = weights / weights.sum(dim=1, keepdim=True)  # [bs, 4] normalize
            # weights = torch.log(total_pixel / (layer_pixel + 1e-3))  # [bs, 4]
            # weights = weights ** 2
            # normalized_weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
            # rec_weight = normalized_weights.view(bs, num_layers, 1, 1)     # # [bs, 4]->[bs, 4, 1, 1]
            # rec_weight = compute_rec_weights(inputs)  # [bs, 4, 1, 1]
            rec_weight = compute_layer_weights(inputs, clamp_max=50)

            inputs_list.append(inputs)
            rec_weights_list.append(rec_weight)
            # loss, recon_loss, kl_div = self.loss(inputs, reconstructions, posterior, rec_weight, self.global_step)

            # losses.append(loss)
            # recon_losses.append(recon_loss)
            # kl_divs.append(kl_div)

        # concatenate to a larger batch
        inputs_cat = torch.cat(inputs_list, dim=0)  # [B * N, C, H, W]
        rec_weight_cat = torch.cat(rec_weights_list, dim=0)

        reconstructions, posterior = self(inputs_cat)
        loss, loss_dic = self.loss(inputs_cat, reconstructions, posterior, rec_weight_cat, self.global_step, **kwargs)
        return loss, loss_dic


    def training_step(self, batch, batch_idx):

        loss, loss_dict = self.shared_step(batch, prefix='train')
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)

        # Emit step-level loss dict for MetricLogger callback.
        step_payload = {}
        for k, v in loss_dict.items():
            base = k.split('/', 1)[1] if k.startswith('train/') else k
            if hasattr(v, 'detach'):
                v = v.detach().float().cpu().item()
            step_payload[base] = v
        step_payload['_global_step'] = int(self.global_step)
        step_payload['_epoch'] = int(self.current_epoch)
        self._step_loss_dict = step_payload

        return loss

    @torch.no_grad()
    def on_train_epoch_end(self):
        # MetricLogger handles CSV and plot output.
        pass


    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, prefix='val')
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):

        # Calculate IoU
        inputs = torch.cat([self.get_input(batch, k) for k in self.image_keys()], dim=0)

        posterior = self.encode(inputs)
        z = posterior.sample().detach().cpu()
        self.latent_collection.append(z)

        recon = self.decode(z.to(self.device))

        # Calculate semantic segmentation IoU
        inputs_mask = torch.where(inputs > 0, 1, 0)  # [-1, 1] -> [0, 1]
        recon_mask = torch.where(recon > 0, 1, 0)
        self.seg_metric.update(recon_mask, inputs_mask)

    def on_predict_epoch_end(self):
        score = self.seg_metric.compute()
        for index, layer in enumerate(self.semantic_layers):
            print(f"IoU {layer}: {score[index].item():.5f}")

        # calculate latent std
        all_latents = torch.cat(self.latent_collection, dim=0)
        std = all_latents.std().item()
        print(f"==== Latent std for this epoch: {std:.5f} ====")


    # def configure_optimizers(self):
    #     lr = self.learning_rate
    #     ae_params_list = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(
    #         self.quant_conv.parameters()) + list(self.post_quant_conv.parameters())
    #     if self.learn_logvar:
    #         print(f"{self.__class__.__name__}: Learning logvar")
    #         ae_params_list.append(self.loss.logvar)
    #     opt_ae = torch.optim.Adam(ae_params_list,
    #                               lr=lr, betas=(0.5, 0.9))
    #     # opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
    #     #                             lr=lr, betas=(0.5, 0.9))
    #     # return [opt_ae, opt_disc], []
    #     return opt_ae

    def configure_optimizers(self):
        assert self.opt_cfg is not None, "VAE training config is not provided"

        base_lr = self.opt_cfg.get("lr", 5e-5)
        min_lr = self.opt_cfg.get("min_lr", 1e-7)
        weight_decay = self.opt_cfg.get("weight_decay", 1e-4)

        ae_params_list = list(self.encoder.parameters()) + \
                     list(self.decoder.parameters()) + \
                     list(self.quant_conv.parameters()) + \
                     list(self.post_quant_conv.parameters())
        if self.learn_logvar:
            print(f"{self.__class__.__name__}: Learning logvar")
            ae_params_list.append(self.loss.logvar)

        train_loader = self.train_dataloader()
        steps_per_epoch = len(train_loader)
        num_epochs = self.trainer.max_epochs
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = int(0.05 * total_steps)

        optimizer = torch.optim.AdamW(
            ae_params_list,
            lr=base_lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay
        )

        # scheduler (linear warmup + cosine decay)
        def lr_lambda(current_step):
            if current_step <= warmup_steps:
                return float(current_step + 1) / float(warmup_steps)
            else:
                decay_step = current_step - warmup_steps
                decay_total = max(total_steps - warmup_steps, 1)
                decay_ratio = (1.0 - decay_step / decay_total) ** 0.96
                min_ratio = min_lr / base_lr
                return max(decay_ratio, min_ratio)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        print(f"Training steps: {total_steps}, Warmup: {warmup_steps}, Steps per epoch: {steps_per_epoch}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "warmup_cosine"
            }
        }

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        inputs_list = []
        for k in self.image_keys():
            x = self.get_input(batch, k)
            x = x.to(self.device)
            inputs_list.append(x)
        x = torch.cat(inputs_list, dim=0)

        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                # x = self.to_rgb(x)
                # xrec = self.to_rgb(xrec)
            # log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            # xrec = 2.*(xrec-xrec.min())/(xrec.max()-xrec.min()) - 1.
            log["reconstruction"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        # assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class VAELoss(nn.Module):
    def __init__(self, kl_dev_weight):
        super().__init__()
        self.reconstruction_loss = nn.MSELoss(reduction='none')
        # self.reconstruction_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.kl_dev_weight = kl_dev_weight

    def forward(self, inputs, reconstructions, posterior, rec_weight, global_step, prefix="train"):

        # VAE Loss: Reconstruction Loss + KL Divergence
        # inputs = (inputs + 1) / 2     # activate when using BCEWithLogitsLoss
        recon_loss = self.reconstruction_loss(reconstructions, inputs).mean(dim=(2, 3))
        weighted_recon_loss = (recon_loss * rec_weight).sum(dim=1)
        weighted_recon_loss = weighted_recon_loss.mean()

        kl_div = -0.5 * torch.sum(1 + posterior.logvar - posterior.mean.pow(2) - posterior.logvar.exp())
        kl_div = kl_div / inputs.shape[0]  # Normalize by batch size
        weighted_kl_div = kl_div * self.kl_dev_weight

        loss = weighted_recon_loss + weighted_kl_div
        # log_dict = {"recon_loss": weighted_recon_loss, "kl_loss": kl_div}
        loss_dic = {
            f'{prefix}/recon_loss': weighted_recon_loss,
            f'{prefix}/kl_loss': weighted_kl_div,
            f'{prefix}/loss': loss,
        }
        return loss, loss_dic


class WeightedMSELoss(nn.Module):
    def __init__(self, reduction='none'):
        super(WeightedMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, predictions, targets, weights):
        # Compute layer-wise MSE
        layer_mse = self.mse(predictions, targets).mean(dim=(2, 3))  # [B, C]
        weighted_loss = layer_mse * weights  # [B, C]

        return weighted_loss.sum(dim=1).mean()  # Sum over channels, average over batch
