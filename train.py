"""
CG-BEV unified training entry point.

Supports three training stages:
    - vae    : Train the first-stage VAE on BEV segmentation maps.
    - ldm    : Train the unconditional latent diffusion model (LDM).
    - cldm   : Train the conditional LDM (ControlNet-based, with BEV condition).
    - cldm_e2e: Train the end-to-end variant (CLDM + frozen perception backbone).

Examples
--------
    python train.py --task cldm --config configs/cldm_res_192.yaml
    python train.py --task ldm  --config configs/ldm_res_192.yaml
    python train.py --task vae  --config configs/cldm_res_192.yaml

    # Warm-start from pretrained weights only (fresh optimizer/epoch/scheduler):
    python train.py --task cldm --config <yaml> --init-weights <ckpt>

    # Full resume from a Lightning checkpoint (restores optimizer, lr scheduler,
    # global_step, current_epoch, RNG, callbacks, etc.):
    python train.py --task cldm --config <yaml> --resume-state <ckpt>
"""

import os

# --- Environment Variable Setup for Performance and Debugging ---
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"

import argparse
import datetime
import multiprocessing
import random

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from cldm.logger import ImageLogger
from cldm.metric_logger import MetricLogger
from cldm.model import create_model, load_state_dict
from ldm.util import instantiate_from_config
from tool_add_cgbev_vae import init_cldm_from_ldm


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CLDM_LDM_CKPT = 'ckpts/ldm_ch=128_res=192_6_layer_epoch=45.ckpt'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Fix random seeds for reproducibility (numpy / torch / lightning)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    print(f"[Seed] All random seeds set to {seed}.")


class CustomProgressBar(TQDMProgressBar):
    """Cleaner tqdm bar: scientific LR, integer global step, no v_num."""

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        if 'lr' in items:
            items['lr'] = f"{items['lr']:.2e}"
        if 'global_step' in items:
            items['global_step'] = int(items['global_step'])
        items.pop("v_num", None)
        return items


def make_log_dir(task: str, baseline: str, condition: str) -> str:
    """Create a timestamped log folder of the form
    `logs/<MM_DD_HH_MM>_train_<task>_<baseline>_<condition>`.

    `condition` is a free-form short tag (e.g. ``map+enc`` for CLDM,
    or ``none`` for vae/ldm).
    """
    now = datetime.datetime.now()
    ts = '_'.join('%02d' % x for x in (now.month, now.day, now.hour, now.minute))
    parts = [ts, 'train', task]
    if baseline:
        parts.append(baseline)
    if condition and condition != 'none':
        parts.append(condition)
    log_dir = os.path.join('logs', '_'.join(parts))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[Log] -> {log_dir}")
    return log_dir


def resolve_project_path(path):
    if not path:
        return None
    path = str(path)
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def infer_condition_tag(cfg) -> str:
    """Best-effort condition tag from a CLDM config.

    Looks at ``model.params.control_key`` and ``model.params.reference_key``.
    """
    try:
        params = cfg.model.params
        has_feat = bool(params.get('control_key'))
        has_map = bool(params.get('reference_key'))
        if has_feat and has_map:
            return 'feat+map'
        if has_feat:
            return 'feat'
        if has_map:
            return 'map'
    except Exception:
        pass
    return 'none'


# ---------------------------------------------------------------------------
# Per-task setup
# ---------------------------------------------------------------------------

def build_vae(cfg, args, log_dir):
    """VAE training: model is the first_stage submodule of the master config."""
    vae_cfg = cfg.model.params.first_stage_config
    trainer_cfg = vae_cfg.params.vae_training_cfg.trainer_config

    model = instantiate_from_config(vae_cfg).cpu()
    init_w = args.init_weights or trainer_cfg.get('init_weights_path') or trainer_cfg.get('resume_path')
    if init_w:
        init_w = resolve_project_path(init_w)
        print(f"[Init] Warm-starting VAE weights from {init_w}")
        model.load_state_dict(load_state_dict(init_w, location='cpu'))

    model.image_key = ['bev_map_gt']
    model.log_dir = log_dir

    ckpt_cb = ModelCheckpoint(
        dirpath=log_dir,
        monitor='val/loss',
        filename='best-ckpt-{epoch}-{step}-{val/loss:.4f}',
        auto_insert_metric_name=False,
        mode='min',
        save_last=True,
    )
    return model, trainer_cfg, ckpt_cb


def build_ldm(cfg, args, log_dir):
    """Unconditional latent diffusion training."""
    trainer_cfg = cfg.model.params.trainer_config

    model = instantiate_from_config(cfg.model).cpu()
    print(f"[Model] LDM instantiated from {args.config}")
    model.log_dir = log_dir
    init_w = args.init_weights or trainer_cfg.get('init_weights_path') or trainer_cfg.get('resume_path')
    if init_w:
        init_w = resolve_project_path(init_w)
        print(f"[Init] Warm-starting LDM weights from {init_w}")
        model.load_state_dict(load_state_dict(init_w, location='cpu'))

    ckpt_cb = ModelCheckpoint(
        dirpath=log_dir,
        monitor='val/loss_vlb',
        filename='best-ckpt-{epoch}-{step}-{val/loss_vlb:.4f}',
        auto_insert_metric_name=False,
        mode='min',
        save_last=True,
    )
    return model, trainer_cfg, ckpt_cb


# Per-baseline raw BEV-feature channel count (matches the perception backbone).
BEV_FEAT_CHANNELS = {
    'lss': 64, 'bevformer': 128, 'stp3': 192, 'bevfusion': 80, 'vggt': 128,
}


def _apply_baseline(cfg, baseline: str) -> None:
    """Override ``data_config.model`` and the encoder input channel."""
    cfg.model.params.data_config.model = baseline
    if baseline in BEV_FEAT_CHANNELS:
        cfg.model.params.control_stage_config.params.bev_encoder_in = BEV_FEAT_CHANNELS[baseline]
    print(f"[CLDM] baseline = {baseline}  (bev_encoder_in = "
          f"{cfg.model.params.control_stage_config.params.get('bev_encoder_in', '?')})")


def build_cldm(cfg, args, log_dir):
    """Conditional LDM (ControlNet-style) training.

    Optionally overrides ``data_config.model`` (the baseline) from CLI.
    """
    trainer_cfg = cfg.model.params.trainer_config

    # Allow CLI to override baseline (lss / bevformer / stp3 / bevfusion / vggt)
    if args.baseline:
        _apply_baseline(cfg, args.baseline)

    # NB: instantiate from the (possibly-mutated) cfg, NOT via create_model() which
    # re-reads the YAML from disk and would discard our overrides.
    model = instantiate_from_config(cfg.model).cpu()
    print(f"[Model] CLDM instantiated from {args.config}")
    model.log_dir = log_dir
    init_w = args.init_weights or trainer_cfg.get('init_weights_path') or trainer_cfg.get('resume_path')
    # When doing a full PL resume, skip the warm-start path: trainer.fit(ckpt_path=)
    # will overwrite the weights anyway (and additionally restore optimizer state).
    if args.resume_state:
        print(f"[Resume-State] Will fully resume training from {args.resume_state} "
              "(weights/optimizer/epoch/scheduler/RNG via trainer.fit). "
              "Skipping warm-start weight load.")
    elif init_w:
        init_w = resolve_project_path(init_w)
        if not os.path.exists(init_w):
            raise FileNotFoundError(
                f"[Init] Pretrained CLDM weights not found: {init_w}. "
                "For fresh CLDM training, leave --init-weights/init_weights_path empty and provide --ldm-ckpt if needed."
            )
        print(f"[Init] Warm-starting full CLDM weights from {init_w} "
              "(optimizer/epoch will start fresh).")
        model.load_state_dict(load_state_dict(init_w, location='cpu'))
    else:
        ldm_ckpt = (args.ldm_ckpt
                    or trainer_cfg.get('ldm_init_path')
                    or trainer_cfg.get('pretrained_ldm_path')
                    or DEFAULT_CLDM_LDM_CKPT)
        ldm_ckpt = resolve_project_path(ldm_ckpt)
        if not os.path.exists(ldm_ckpt):
            raise FileNotFoundError(
                f"[Init] No CLDM resume checkpoint was provided, so training requires a pretrained LDM checkpoint. "
                f"Expected: {ldm_ckpt}. You can override it with --ldm-ckpt or trainer_config.ldm_init_path."
            )
        print("[Init] No full CLDM resume checkpoint provided.")
        print(f"[Init] Initializing CLDM from LDM checkpoint: {ldm_ckpt}")
        init_cldm_from_ldm(model, ldm_ckpt, location='cpu')
    model.sd_locked = cfg.model.params.sd_locked

    ckpt_cb = ModelCheckpoint(
        dirpath=log_dir,
        monitor='val/IoU',
        filename='best-ckpt-{epoch}-{step}-{val/IoU:.4f}',
        auto_insert_metric_name=False,
        mode='max',
        save_last=True,
    )
    return model, trainer_cfg, ckpt_cb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CG-BEV unified trainer")
    p.add_argument('--task', required=True,
                   choices=['vae', 'ldm', 'cldm', 'cldm_e2e'])
    p.add_argument('--config', required=True, help='Path to YAML config.')
    p.add_argument('--baseline', default=None,
                   choices=[None, 'lss', 'bevformer', 'stp3', 'bevfusion', 'vggt'],
                   help='Override data_config.model (CLDM only).')
    p.add_argument('--init-weights', '--resume', dest='init_weights', default=None,
                   help='Pretrained checkpoint to WARM-START from (weights only; '
                        'optimizer/epoch/scheduler are reset). For CLDM, this must be a '
                        'complete CLDM checkpoint. (Old --resume is kept as a deprecated alias.)')
    p.add_argument('--resume-state', dest='resume_state', default=None,
                   help='Lightning checkpoint to FULLY RESUME training from. Restores model '
                        'weights, optimizer state, lr scheduler, global_step, current_epoch, '
                        'RNG and callback state via trainer.fit(ckpt_path=...).')
    p.add_argument('--ldm-ckpt', default=None,
                   help='Pretrained LDM checkpoint used to initialize fresh CLDM training. Defaults to ckpts/ldm_ch=128_res=192_6_layer_epoch=45.ckpt.')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


BUILDERS = {
    'vae': build_vae,
    'ldm': build_ldm,
    'cldm': build_cldm,
    'cldm_e2e': build_cldm,  # same builder; different config target class
}


def main():
    multiprocessing.set_start_method('spawn', force=True)
    args = parse_args()

    cfg = OmegaConf.load(args.config)
    seed_everything(args.seed)

    baseline = (args.baseline
                or cfg.model.params.get('data_config', {}).get('model', '')
                or '')
    condition = infer_condition_tag(cfg) if args.task.startswith('cldm') else 'none'

    log_dir = make_log_dir(args.task, baseline, condition)
    OmegaConf.save(cfg, os.path.join(log_dir, 'config.yaml'))

    # Resolve full-resume checkpoint (CLI takes precedence over yaml).
    resume_state_path = args.resume_state

    model, trainer_cfg, ckpt_cb = BUILDERS[args.task](cfg, args, log_dir)

    # YAML fallback for full PL resume.
    if not resume_state_path:
        resume_state_path = trainer_cfg.get('resume_state_path') or trainer_cfg.get('pl_resume_path')
    resume_state_path = resolve_project_path(resume_state_path)
    if resume_state_path and not os.path.exists(resume_state_path):
        raise FileNotFoundError(f"[Resume-State] Lightning ckpt not found: {resume_state_path}")

    image_logger = ImageLogger(
        batch_frequency=trainer_cfg.logger_freq,
        rescale=False,
        log_folder=log_dir,
    )
    metric_logger = MetricLogger(
        log_dir=log_dir,
        log_step_freq=int(trainer_cfg.get('log_step_freq', max(trainer_cfg.logger_freq // 5, 10))),
    )
    trainer = pl.Trainer(
        strategy='auto',
        accelerator='gpu',
        devices=trainer_cfg.gpu_num,
        precision=trainer_cfg.precision,
        callbacks=[image_logger, metric_logger, ckpt_cb, CustomProgressBar()],
        logger=False,
        max_epochs=trainer_cfg.epochs,
    )
    if resume_state_path:
        print(f"[Resume-State] trainer.fit(ckpt_path={resume_state_path})")
    trainer.fit(model, ckpt_path=resume_state_path)


if __name__ == '__main__':
    main()
