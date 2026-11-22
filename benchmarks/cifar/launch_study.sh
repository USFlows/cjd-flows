#!/usr/bin/env bash
# Runs the CIFAR-10 scaling study as a job queue with at most $PARALLEL concurrent
# GPU jobs (many concurrent ROCm processes can trigger GPU memory access faults).
# Usage: PY=/path/to/python PARALLEL=3 setsid nohup bash benchmarks/cifar/launch_study.sh &
set -uo pipefail
PY=${PY:-python}
OUT=${OUT:-benchmarks/runs/cifar}
PARALLEL=${PARALLEL:-3}
mkdir -p "$OUT/logs"
JOBS=(
  "small-us-gauss|--variant us-gauss --size small --epochs 24 --eval-every 2"
  "small-us-ggmix|--variant us-ggmix --size small --epochs 24 --eval-every 2"
  "small-affine|--variant affine --size small --epochs 24 --eval-every 2"
  "medium-us-ggmix-noise|--variant us-ggmix --size medium --epochs 12 --eval-every 1 --noise 0.05"
  "small-us-spline|--variant us-spline --size small --epochs 24 --eval-every 2"
  "small-us-mono|--variant us-mono --size small --epochs 24 --eval-every 2"
  "medium-us-ggmix|--variant us-ggmix --size medium --epochs 12 --eval-every 2"
  "medium-us-gauss|--variant us-gauss --size medium --epochs 12 --eval-every 2"
  "medium-affine|--variant affine --size medium --epochs 12 --eval-every 2"
)
run_job() {
  local name=${1%%|*} args=${1#*|}
  for attempt in 1 2; do
    # shellcheck disable=SC2086
    PYTHONPATH=src:. "$PY" benchmarks/cifar/train.py --out "$OUT" $args >> "$OUT/logs/$name.log" 2>&1 && return 0
    echo "[queue] $name failed (attempt $attempt)" >> "$OUT/logs/$name.log"
  done
}
export -f run_job; export PY OUT
printf '%s\n' "${JOBS[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'run_job "$@"' _ {}
echo "[queue] all jobs finished $(date)" >> "$OUT/logs/queue.log"
