#!/usr/bin/env bash
# Launches the CIFAR-10 scaling study (all runs concurrently on one GPU).
# Usage: PY=/path/to/python bash benchmarks/cifar/launch_study.sh
set -euo pipefail
PY=${PY:-python}
OUT=${OUT:-benchmarks/runs/cifar}
mkdir -p "$OUT/logs"
run() { local name=$1; shift; PYTHONPATH=src:. nohup "$PY" benchmarks/cifar/train.py --out "$OUT" "$@" > "$OUT/logs/$name.log" 2>&1 & }
# Wave A: small models, all base / coupling variants
for v in us-gauss us-ggmix us-spline us-mono affine; do
  run "small-$v" --variant "$v" --size small --epochs 16 --eval-every 2
done
# Wave B: medium models
for v in us-gauss us-ggmix affine; do
  run "medium-$v" --variant "$v" --size medium --epochs 10 --eval-every 2
done
# Sample-quality run: noise augmentation + Tweedie denoising
run "medium-us-ggmix-noise" --variant us-ggmix --size medium --epochs 10 --eval-every 1 --noise 0.05
