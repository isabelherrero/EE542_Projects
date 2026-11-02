#!/usr/bin/env bash
set -euo pipefail
# Run on node0. Sample can fail at DOCA Flow capability, as lab describes.
base=$(cd "$(dirname "$0")/.." && pwd)
source "$base/part7/config/node.env"
if [[ "$GPU" == EDIT* || "$NIC" == EDIT* ]]; then echo 'Edit config/node.env GPU and NIC first'; exit 1; fi
mkdir -p "$HOME/lab8_results"
img=nvcr.io/nvidia/doca/doca:devel-cuda13.0.0-3.5.0-devel-host
sudo docker pull "$img"
# Write command output back to the host via mounted results directory.
sudo docker run --rm --privileged --net=host --gpus all   -v /dev/infiniband:/dev/infiniband   -v /dev/hugepages:/dev/hugepages   -v /opt/mellanox/gdrcopy/src:/opt/gdrcopy:ro   -v "$HOME/lab8_results:/results"   -e GPU="$GPU" -e NIC="$NIC" "$img" bash -lc '
    export LD_LIBRARY_PATH=/opt/gdrcopy:${LD_LIBRARY_PATH:-}
    cd /opt/mellanox/doca/samples/doca_gpunetio/gpunetio_simple_receive
    meson setup /tmp/build --buildtype=release && ninja -C /tmp/build || exit 1
    set +e
    /tmp/build/doca_gpunetio_simple_receive -n "$NIC" -g "$GPU" -e 0 > /results/udp_sample.log 2>&1
    rc=$?
    cat /results/udp_sample.log
    echo "Sample exit code: $rc"
    exit 0
  '
