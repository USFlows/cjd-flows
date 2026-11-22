#!/bin/bash
# Runs the MNIST example experiment and serves TensorBoard (used by the dockerfile).
# Requires the `experiments` extra: `uv sync --extra experiments` (or the dockerfile).
mkdir -p ray_results reports
uv run --no-dev --extra experiments tensorboard --logdir=ray_results --bind_all &
RAY=$(realpath ./ray_results)
uv run --no-dev --extra experiments python scripts/run-experiment.py \
    --config experiments/mnist/mnist.yaml --report_dir reports --storage_path "$RAY"
