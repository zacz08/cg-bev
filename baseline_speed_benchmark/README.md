# Baseline Speed Benchmark

This folder contains isolated speed benchmark scripts for the three baseline models requested for the sm_120 GPU environment.

## Environment

Use the cloned conda environment:

```bash
conda activate baseline_sm120
```

The environment uses Python 3.12, PyTorch 2.7.1 + CUDA 12.8, and runs on the RTX PRO 6000 Blackwell GPU.

## Commands

```bash
python baseline_speed_benchmark/bench_lss.py --warmup-batches 20 --measure-batches 100
python baseline_speed_benchmark/bench_bevformer.py --warmup-batches 20 --measure-batches 100
python baseline_speed_benchmark/bench_bevfusion.py --warmup-batches 20 --measure-batches 100
```

Each script writes CSV and JSON metrics under `baseline_speed_benchmark/logs/`.

## Compatibility Notes

- `bench_lss.py` uses the local Lift-Splat-Shoot checkpoint and adapts `final_dim` from the checkpoint frustum.
- `bench_bevformer.py` keeps the DCNv2 stages enabled. Missing DCNv2 CUDA ops are replaced by a PyTorch module with the same main convolution and `conv_offset` branch, and missing multi-scale deformable attention CUDA ops use a PyTorch fallback in the benchmark process only.
- `bench_bevfusion.py` uses the camera-only BEVFusion segmentation config and checkpoint by default, so LiDAR and fusion modules are not constructed. The missing BEVFusion CUDA `bev_pool` op is replaced by a PyTorch implementation in the benchmark process only.

## Latest Results

| Model | ms/sample | FPS | GFLOPs/sample |
| --- | ---: | ---: | ---: |
| LSS | 13.83 | 72.30 | 75.67 |
| BEVFormer | 173.77 | 5.75 | 3101.77 |
| BEVFusion camera-only | 45.51 | 21.98 | 429.16 |