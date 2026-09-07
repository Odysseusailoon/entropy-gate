#!/usr/bin/env bash
# One node, five homogeneous GPUs, shared queue; one complete paired block per claim.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 RUN_DIRECTORY calibration|test|reference-N" >&2
  exit 2
fi
run_dir=$1
phase=$2
python_bin=${EGS_PYTHON:-python}
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
IFS=',' read -r -a selected_gpus <<< "${EGS_GPU_IDS:-0,1,2,3,4}"
if ((${#selected_gpus[@]} != 5)); then
  echo "Select exactly five idle GPUs with EGS_GPU_IDS." >&2
  exit 2
fi
required_gpu=$("$python_bin" -c 'import sys; from egs.net.protocol import open_run; print(open_run(sys.argv[1])[0].study.required_gpu_name)' "$run_dir")
"$python_bin" -m egs.net.devices --devices "${EGS_GPU_IDS:-0,1,2,3,4}" --required-name "$required_gpu"
if [[ "$phase" == calibration || "$phase" == test ]]; then
  "$python_bin" -m egs.net initialize --run "$run_dir" --phase "$phase"
fi
mkdir -p "$run_dir/logs"
pids=()
# Only this launcher's still-owned child processes may be terminated. Never use
# pkill/killall, process-name matching, device reset or other users' process IDs.
cleanup() { for pid in "${remaining[@]}"; do kill "$pid" 2>/dev/null || true; done; }
remaining=()
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for gpu in "${selected_gpus[@]}"; do
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m egs.net worker --run "$run_dir" --phase "$phase" --worker "gpu$gpu" >"$run_dir/logs/$phase-gpu$gpu.log" 2>&1 &
  pids+=("$!")
  remaining=("${pids[@]}")
done
remaining=("${pids[@]}")
while ((${#remaining[@]})); do
  active=()
  for pid in "${remaining[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      active+=("$pid")
    else
      if ! wait "$pid"; then
        # Remove every reaped PID before cleanup; retained PIDs are our children.
        kept=()
        for child in "${remaining[@]}"; do
          if [[ "$child" != "$pid" ]] && kill -0 "$child" 2>/dev/null; then kept+=("$child"); fi
        done
        remaining=("${kept[@]}")
        echo "Worker failed; stopping only this launcher's workers. Inspect logs and recover dead tasks explicitly." >&2
        exit 1
      fi
    fi
  done
  remaining=("${active[@]}")
  if ((${#remaining[@]})); then sleep 2; fi
done
pids=()
remaining=()
if [[ "$phase" == calibration ]]; then
  "$python_bin" -m egs.net calibrate --run "$run_dir"
elif [[ "$phase" == test ]]; then
  "$python_bin" -m egs.net analyze --run "$run_dir"
else
  "$python_bin" -m egs.net reference-check --run "$run_dir"
fi
