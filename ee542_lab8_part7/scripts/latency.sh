#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/latency.sh server|client [proxy]
base=$(cd "$(dirname "$0")/.." && pwd)
source "$base/part7/config/node.env"
role="${1:?Use server or client}"
if [[ "$GPU" == EDIT* || "$RDMA" == EDIT* ]]; then echo 'Edit config/node.env first'; exit 1; fi
export EX="$HOME/gpunetio/install/examples"
export LD_LIBRARY_PATH="$HOME/gpunetio/install/lib:/opt/mellanox/gdrcopy/src:${LD_LIBRARY_PATH:-}"
export DOCA_GPUNETIO_LOG=6
mkdir -p "$HOME/lab8_results"
proxy="${2:-1}"
cmd=("$EX/gpunetio_verbs_write_lat" -g "$GPU" -d "$RDMA" -l "$GID" -p "$proxy")
if [ "$role" = client ]; then
  cmd+=(-c "$SERVER")
elif [ "$role" != server ]; then echo 'Use server or client'; exit 1; fi
f="$HOME/lab8_results/gpu_latency_${role}_$(date +%Y%m%d_%H%M%S).log"
echo "Running: ${cmd[*]}" | tee "$f"
set +e
"${cmd[@]}" 2>&1 | tee -a "$f"
rc=${PIPESTATUS[0]}
set -e
echo "Exit code: $rc" | tee -a "$f"
exit "$rc"
