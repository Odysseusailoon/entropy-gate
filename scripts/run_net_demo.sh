#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin=${EGS_PYTHON:-python}
"$python_bin" -m egs.net freeze --config configs/net-demo.yaml
"$python_bin" -m egs.net preflight --run runs/net-demo
"$python_bin" -m egs.net initialize --run runs/net-demo --phase calibration
"$python_bin" -m egs.net worker --run runs/net-demo --phase calibration --worker cpu-demo
"$python_bin" -m egs.net calibrate --run runs/net-demo
"$python_bin" -m egs.net initialize --run runs/net-demo --phase test
"$python_bin" -m egs.net worker --run runs/net-demo --phase test --worker cpu-demo
"$python_bin" -m egs.net analyze --run runs/net-demo
