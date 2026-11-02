#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/rdma_baseline.sh server|client [proxy]
base=$(cd "$(dirname "$0")/.." && pwd)
source "$base/part7/config/node.env"
role="${1:?Use server or client}"
if [[ "$GPU" == EDIT* || "$RDMA" == EDIT* ]]; then echo 'Edit config/node.env first'; exit 1; fi
export EX="$HOME/gpunetio/install/examples"
export LD_LIBRARY_PATH="$HOME/gpunetio/install/lib:/opt/mellanox/gdrcopy/src:${LD_LIBRARY_PATH:-}"
export DOCA_GPUNETIO_LOG=6
mkdir -p "$HOME/lab8_results"
test="${2:?Use lat or bw}"
case "$test" in
  lat) cmd=(ib_write_lat -d "$RDMA" -x "$GID") ;;
  bw) cmd=(ib_write_bw -d "$RDMA" -x "$GID" --report_gbits) ;;
  *) echo "Use lat or bw"; exit 1 ;;
esac
if [ "$role" = client ]; then
  cmd+=("$SERVER")
elif [ "$role" != server ]; then echo 'Use server or client'; exit 1; fi
f="$HOME/lab8_results/cpu_rdma_${role}_$(date +%Y%m%d_%H%M%S).log"
echo "Running: ${cmd[*]}" | tee "$f"
set +e
"${cmd[@]}" 2>&1 | tee -a "$f"
rc=${PIPESTATUS[0]}
set -e
echo "Exit code: $rc" | tee -a "$f"
exit "$rc"
