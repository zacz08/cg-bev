"""
CG-BEV unified inference / evaluation entry point.

Usage
-----
    python test.py --task cldm --config configs/cldm_res_192.yaml \
                   --ckpt logs/.../best-ckpt-...ckpt
    python test.py --task ldm  --config configs/ldm_res_192.yaml --ckpt <ckpt>
    python test.py --task vae  --config configs/cldm_res_192.yaml --ckpt <ckpt>

Optional flags:
    --baseline {lss,bevformer,stp3,bevfusion,vggt}   override perception model
    --split    {val,mini_val,test,train,mini_train}  dataset split
    --batch-size N
    --save-vis              render visualisations to log dir
    --log-frame             render camera frames + predicted BEV when saving visualisations
    --measure-speed         report per-sample inference time / FPS
"""

import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["HYDRA_FULL_ERROR"] = "1"

import argparse
import csv
import datetime
import multiprocessing
import random
import time

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from ldm.util import instantiate_from_config
from nuScenesSegDataset import nuScenesSegDataset


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def make_log_dir(task: str, baseline: str) -> str:
    now = datetime.datetime.now()
    ts = '_'.join('%02d' % x for x in (now.month, now.day, now.hour, now.minute))
    parts = [ts, 'pred', task]
    if baseline:
        parts.append(baseline)
    log_dir = os.path.join('logs', '_'.join(parts))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[Log] -> {log_dir}")
    return log_dir


def parse_args():
    p = argparse.ArgumentParser(description="CG-BEV unified tester")
    p.add_argument('--task', required=True, choices=['vae', 'ldm', 'cldm', 'cldm_e2e'])
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--baseline', default=None,
                   choices=[None, 'lss', 'bevformer', 'stp3', 'bevfusion', 'vggt'])
    p.add_argument('--split', default='val',
                   choices=['train', 'val', 'test', 'mini_train', 'mini_val'])
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--save-vis', dest='save_vis', action='store_true', default=None,
                   help='Enable ImageLogger to save visualisations '
                        '(default: ON for ldm, OFF otherwise).')
    p.add_argument('--no-save-vis', dest='save_vis', action='store_false',
                   help='Disable saving visualisations.')
    p.add_argument('--log-frame', action='store_true',
                   help='When used with --save-vis, save nuScenes camera frames together with the predicted BEV mask.')
    p.add_argument('--measure-speed', action='store_true',
                   help='Report per-sample inference time and FPS.')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


BEV_FEAT_CHANNELS = {
    'lss': 64, 'bevformer': 128, 'stp3': 192, 'bevfusion': 80, 'vggt': 128,
}


def build_model(args, cfg):
    """Instantiate model and load checkpoint based on task."""
    if args.task == 'vae':
        model = instantiate_from_config(cfg.model.params.first_stage_config).cpu()
        model.image_key = 'bev_map_gt'
    else:
        if args.task.startswith('cldm') and args.baseline:
            cfg.model.params.data_config.model = args.baseline
            if args.baseline in BEV_FEAT_CHANNELS:
                cfg.model.params.control_stage_config.params.bev_encoder_in = \
                    BEV_FEAT_CHANNELS[args.baseline]
            print(f"[CLDM] baseline = {args.baseline} | "
                  f"bev_encoder_in = {cfg.model.params.control_stage_config.params.bev_encoder_in}")
        # NB: instantiate from the (possibly-mutated) cfg, NOT via create_model() which
        # re-reads the YAML from disk and would discard our overrides.
        model = instantiate_from_config(cfg.model).cpu()
        print(f"[Model] instantiated from {args.config}")
    print(f"[Ckpt] Loading {args.ckpt}")
    sd = load_state_dict(args.ckpt, location='cpu')

    model.load_state_dict(sd)
    model.eval()
    return model


def build_dataloader(args, cfg):
    """Build dataset / loader, honouring CLI overrides."""
    data_cfg = cfg.model.params.get('data_config', {})
    resolution = data_cfg.get('resolution', 192)
    batch_size = args.batch_size or data_cfg.get('batch_size', 8)
    baseline = args.baseline or data_cfg.get('model', 'lss')
    return_feature = args.task.startswith('cldm')

    dataset = nuScenesSegDataset(
        data_split=args.split,
        resolution=resolution,
        augment=False,                  # never augment at test time
        return_feature=return_feature,
        model=baseline,
    )
    loader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=False)
    print(f"[Data] split={args.split} | baseline={baseline} | "
          f"res={resolution} | bs={batch_size} | n={len(dataset)}")
    return loader


def move_batch_to_device(batch, device):
    """Move tensor values in a batch to the selected device."""
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def infer_batch_size(batch) -> int:
    """Infer batch size from the first tensor value in a batch."""
    for value in batch.values():
        if torch.is_tensor(value):
            return int(value.shape[0])
    raise ValueError("Cannot infer batch size from a batch with no tensor values.")


def count_parameters(model) -> tuple[int, int]:
    """Return total and trainable parameter counts for the instantiated model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def bytes_to_mb(value):
    """Convert bytes to MiB, preserving None for non-CUDA runs."""
    return None if value is None else value / (1024.0 ** 2)


def synchronize_device(device):
    """Synchronize CUDA work when running on GPU."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def reset_seg_metric(model):
    """Clear the segmentation metric updated by predict_step, when present."""
    first_stage = getattr(model, 'first_stage_model', None)
    metric = getattr(first_stage, 'seg_metric', None)
    if metric is not None and hasattr(metric, 'reset'):
        metric.reset()


def profile_flops(model, loader, device):
    """Estimate FLOPs for one bs=1 predict_step using torch.profiler."""
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception as exc:  # noqa: BLE001
        return None, f"torch.profiler unavailable: {exc}"

    try:
        batch = next(iter(loader))
    except StopIteration:
        return None, "dataloader is empty"

    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    batch = move_batch_to_device(batch, device)
    try:
        synchronize_device(device)
        with torch.no_grad():
            with profile(activities=activities, record_shapes=True, with_flops=True) as prof:
                model.predict_step(batch)
        synchronize_device(device)
        flops = sum(int(getattr(event, 'flops', 0) or 0) for event in prof.key_averages())
        reset_seg_metric(model)
        return flops, "torch.profiler with_flops=True; unsupported ops are not counted"
    except Exception as exc:  # noqa: BLE001
        reset_seg_metric(model)
        return None, f"FLOPs profiling failed: {exc}"


def write_speed_metrics_csv(log_dir: str, metrics: dict) -> str:
    """Write speed and model statistics to a CSV file in the log directory."""
    csv_path = os.path.join(log_dir, 'speed_metrics.csv')
    columns = [
        'timestamp', 'task', 'baseline', 'split', 'resolution', 'batch_size',
        'warmup_batches', 'measured_batches', 'measured_samples',
        'avg_per_sample_ms', 'fps',
        'cldm_param_count', 'cldm_param_million',
        'cldm_trainable_param_count', 'cldm_trainable_param_million',
        'peak_memory_allocated_mb', 'peak_memory_reserved_mb',
        'flops', 'gflops', 'flops_note',
        'config', 'ckpt',
    ]
    row = {key: ('' if metrics.get(key) is None else metrics.get(key)) for key in columns}
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)
    return csv_path


def measure_speed(model, loader, log_dir, args, cfg, baseline, device=None, warmup=20, n_batches=100):
    """Measure bs=1 speed, params, CUDA memory, FLOPs, and write a CSV report."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model = model.to(device)
    total_params, trainable_params = count_parameters(model)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    times, samples = [], 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            if i < warmup:
                model.predict_step(batch)
                continue
            synchronize_device(device)
            t0 = time.perf_counter()
            model.predict_step(batch)
            synchronize_device(device)
            times.append(time.perf_counter() - t0)
            samples += infer_batch_size(batch)
            if i - warmup + 1 >= n_batches:
                break
    if not times:
        print("[Speed] not enough batches to measure.")
        return

    if device.type == 'cuda':
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    else:
        peak_allocated = None
        peak_reserved = None

    flops, flops_note = profile_flops(model, loader, device)
    reset_seg_metric(model)

    total = sum(times)
    per_sample = total / samples * 1000.0
    fps = samples / total
    data_cfg = cfg.model.params.get('data_config', {}) if hasattr(cfg, 'model') else {}
    resolution = data_cfg.get('resolution', '')
    metrics = {
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'task': args.task,
        'baseline': baseline,
        'split': args.split,
        'resolution': resolution,
        'batch_size': 1,
        'warmup_batches': warmup,
        'measured_batches': len(times),
        'measured_samples': samples,
        'avg_per_sample_ms': per_sample,
        'fps': fps,
        'cldm_param_count': total_params,
        'cldm_param_million': total_params / 1e6,
        'cldm_trainable_param_count': trainable_params,
        'cldm_trainable_param_million': trainable_params / 1e6,
        'peak_memory_allocated_mb': bytes_to_mb(peak_allocated),
        'peak_memory_reserved_mb': bytes_to_mb(peak_reserved),
        'flops': flops,
        'gflops': None if flops is None else flops / 1e9,
        'flops_note': flops_note,
        'config': args.config,
        'ckpt': args.ckpt,
    }
    csv_path = write_speed_metrics_csv(log_dir, metrics)
    print(f"[Speed] avg per-sample: {per_sample:.2f} ms   |   FPS: {fps:.2f}   "
          f"(bs=1, samples={samples}, batches={len(times)})")
    print(f"[Params] CLDM params: {total_params:,} ({total_params / 1e6:.2f}M) | "
          f"trainable: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    if peak_allocated is not None:
        print(f"[Memory] peak allocated: {bytes_to_mb(peak_allocated):.2f} MiB | "
              f"peak reserved: {bytes_to_mb(peak_reserved):.2f} MiB")
    else:
        print("[Memory] CUDA is unavailable; GPU memory metrics were not collected.")
    if flops is not None:
        print(f"[FLOPs] {flops / 1e9:.2f} GFLOPs per bs=1 predict_step")
    else:
        print(f"[FLOPs] unavailable ({flops_note})")
    print(f"[Speed] metrics csv -> {csv_path}")


def main():
    multiprocessing.set_start_method('spawn', force=True)
    args = parse_args()

    cfg = OmegaConf.load(args.config)
    seed_everything(args.seed)

    baseline = (args.baseline
                or cfg.model.params.get('data_config', {}).get('model', '')
                or '')

    save_vis = args.save_vis
    if save_vis is None:
        save_vis = (args.task == 'ldm')

    if args.measure_speed:
        if args.batch_size not in (None, 1):
            print(f"[Speed] --measure-speed forces batch_size=1; ignoring --batch-size={args.batch_size}")
        args.batch_size = 1
        if hasattr(cfg, 'model') and 'data_config' in cfg.model.params:
            cfg.model.params.data_config.batch_size = 1

    if save_vis:
        if args.batch_size not in (None, 1):
            print(f"[Vis] --save-vis forces batch_size=1; ignoring --batch-size={args.batch_size}")
        args.batch_size = 1

    log_dir = make_log_dir(args.task, baseline)
    if hasattr(cfg, 'model'):
        OmegaConf.save(cfg, os.path.join(log_dir, 'config.yaml'))

    model = build_model(args, cfg)
    model.log_dir = log_dir
    loader = build_dataloader(args, cfg)

    if args.measure_speed:
        measure_speed(model, loader, log_dir=log_dir, args=args, cfg=cfg, baseline=baseline)
        return

    print(f"[Vis] save_vis = {save_vis} -> {log_dir}/image_log_predict/")
    if args.log_frame and not save_vis:
        print("[Vis] --log-frame ignored because visualisation saving is disabled.")

    image_logger = ImageLogger(
        batch_frequency=1,
        max_images=1 if save_vis else 4,
        rescale=False,
        disabled=not save_vis,
        image_style='video_frame' if (save_vis and args.log_frame) else 'normal',
        data_split=args.split,
        log_folder=log_dir,
    )
    trainer = pl.Trainer(
        strategy='auto', accelerator='gpu', devices=1, precision=32,
        callbacks=[image_logger], logger=False,
    )
    trainer.predict(model, dataloaders=loader)


if __name__ == '__main__':
    main()
