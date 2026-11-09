#!/bin/bash
# Runs every experiment config in experiments/*/*.yaml and serves TensorBoard.
# Requires the `experiments` extra: `uv sync --extra experiments` (or the dockerfile).

# Base directory for experiments
BASE_DIR="experiments"

# Directory to store reports
REPORT_DIR="reports"

# Make sure the report directory exists
mkdir -p "$REPORT_DIR"

# Start TensorBoard in the background
uv run --no-dev --extra experiments tensorboard --logdir="$BASE_DIR" --bind_all &

# Loop through all subdirectories in the experiments directory
for dataset in "$BASE_DIR"/*; do
    if [ -d "$dataset" ]; then
        # Loop through all .yaml files in the dataset directory
        for config in "$dataset"/*.yaml; do
            echo "Running experiment with config $config"

            # Run the experiment script with the given config and report directory
            uv run --no-dev --extra experiments python scripts/run-experiment.py \
                --config "$config" --report_dir "$REPORT_DIR" --storage_path "$(realpath "$BASE_DIR")"

            echo "Experiment with config $config completed"
        done
    fi
done
