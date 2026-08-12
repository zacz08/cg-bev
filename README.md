# CG-BEV: Conditional Generative Bird's-Eye-View

CG-BEV is a conditional generative framework for refining bird's-eye-view
(BEV) semantic segmentation from perception features and coarse predictions.
This repository contains the official training, evaluation, data preparation,
and benchmarking code.

## Overview

CG-BEV uses a three-stage training pipeline:

1. A variational autoencoder (VAE) learns the latent representation of BEV
   semantic maps.
2. A latent diffusion model (LDM) learns the unconditional BEV map prior.
3. CG-BEV conditions the generative prior on BEV features and coarse semantic
   predictions from a perception backbone.

The unified entry points support VAE, LDM, CG-BEV, and optional end-to-end
training or evaluation. The CLI keeps `cldm` as the task identifier for the
conditional latent diffusion stage.

## Repository layout

```text
CG-BEV/
├── train.py                         # Unified training entry point
├── test.py                          # Unified evaluation entry point
├── nuScenesSegDataset.py            # nuScenes BEV segmentation dataset
├── configs/                         # Model, data, and training configurations
├── cldm/                            # CG-BEV conditional model
│   ├── cldm.py                      # Main CG-BEV implementation
│   ├── cgbev_stp3_2layer.py         # ST-P3-specific CG-BEV variant
│   ├── bevencoder.py                # Conditional BEV encoders
│   ├── bevfusion_seghead.py         # BEVFusion-style encoder components
│   ├── e2e_lss.py                   # Optional LSS end-to-end wrapper
│   └── e2e_stp3.py                  # Optional ST-P3 end-to-end wrapper
├── ldm/
│   ├── models/autoencoder.py        # VAE implementation
│   ├── models/diffusion/            # Diffusion samplers
│   ├── modules/                     # Neural network building blocks
│   └── diffusion/                   # Latent diffusion implementation
├── tools/                            # Data generation and evaluation utilities
├── baseline_speed_benchmark/         # Isolated baseline speed benchmarks
├── data/nuscenes/                    # nuScenes and generated BEV data
├── ckpts/                            # Downloaded or trained checkpoints
├── logs/                             # Training and evaluation outputs
├── lss/                              # Optional external LSS source tree
└── stp3/                             # Optional external ST-P3 source tree
```

The `lss/` and `stp3/` directories are optional external dependencies and are
not distributed with this repository. They are only required by the matching
end-to-end wrappers and feature extraction scripts.

## Installation

```bash
conda create -n cgbev python=3.12 -y
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Run the installation commands inside the `cgbev` environment.

## Data preparation

Download nuScenes and arrange the raw data and generated inputs as follows:

```text
data/nuscenes/
├── samples/
├── sweeps/
├── maps/
├── can_bus/
├── v1.0-trainval/
├── v1.0-test/
├── nuscenes_infos_train.pkl
├── nuscenes_infos_val.pkl
├── bev_seg_gt_mask_192/
│   ├── train/
│   └── val/
├── bev_seg_gt_mask_512/
│   ├── train/
│   └── val/
├── bev_feat_<baseline>/
│   ├── train/*.pt
│   └── val/*.pt
├── bev_pred_<baseline>/
│   ├── train/*.npy
│   └── val/*.npy
├── prompt_<baseline>_train.json
└── prompt_<baseline>_val.json
```

`<baseline>` can be `lss`, `bevformer`, `stp3`, `bevfusion`, or `vggt`.
Generate semantic ground-truth masks with:

```bash
python tools/generate_bev_gt.py \
  --ds-version v1.0-trainval \
  --data-split train \
  --resolution 192
```

Feature and coarse-prediction files must use the filenames recorded in the
corresponding `prompt_<baseline>_<split>.json` file.

## Pretrained models

The download links are placeholders and will be updated with the public model
release.

| Model | Output resolution | Download |
| --- | ---: | --- |
| VAE | — | TBA |
| LDM | 192 × 192 | TBA |
| CG-BEV | 192 × 192 | TBA |
| LDM | 512 × 512 | TBA |
| CG-BEV | 512 × 512 | TBA |

Place downloaded weights in `ckpts/`. Update the checkpoint paths in the YAML
configuration or pass them through the command line.

## Training

Train the three stages in order:

```bash
# Stage 1: VAE
python train.py --task vae --config configs/cldm_res_192.yaml

# Stage 2: unconditional LDM
python train.py --task ldm --config configs/ldm_res_192.yaml

# Stage 3: CG-BEV with a selected perception baseline
python train.py --task cldm \
  --config configs/cldm_res_192.yaml \
  --baseline lss \
  --ldm-ckpt ckpts/ldm_192.ckpt
```

Use `--init-weights` to warm-start weights without optimizer state, or
`--resume-state` to restore a complete PyTorch Lightning training state.
Training outputs are written to timestamped subdirectories under `logs/`.

For 512 × 512 training, use `configs/ldm_res_512.yaml` and
`configs/cldm_res_512.yaml` with matching checkpoints and generated masks.

## Evaluation

Evaluate a CG-BEV checkpoint on the nuScenes validation split:

```bash
python test.py --task cldm \
  --config configs/cldm_res_192.yaml \
  --ckpt ckpts/cgbev_192.ckpt \
  --baseline lss \
  --split val
```

Add `--save-vis` to save qualitative results or `--measure-speed` to report
latency, throughput, parameter counts, memory use, and profiler-supported
FLOPs. Evaluation outputs are stored under `logs/`.

## Citation

The citation entry will be added with the camera-ready paper release.

## License

This project is released under the [Apache License 2.0](LICENSE).
