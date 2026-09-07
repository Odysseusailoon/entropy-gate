#!/usr/bin/env bash
# Compatibility entry point. Hardware is validated against the frozen run.
set -euo pipefail
exec bash "$(dirname "$0")/launch_net_five_gpu.sh" "$@"
