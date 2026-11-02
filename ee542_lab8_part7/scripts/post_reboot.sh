#!/usr/bin/env bash
set -euo pipefail
sudo nvidia-smi -pm 1
sudo sysctl -w vm.nr_hugepages=2048
cd /opt/mellanox/gdrcopy
sudo ./insmod.sh
# sudo modprobe nvidia-peermem || true
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH="$HOME/gpunetio/install/lib:/opt/mellanox/gdrcopy/src:${LD_LIBRARY_PATH:-}"
cat /proc/cmdline
nvidia-smi
nvcc --version
# lsmod | grep -E 'gdrdrv|nvidia_peermem' || true
grep -i huge /proc/meminfo
ibv_devices
sudo docker info --format '{{.DockerRootDir}}'
echo 'ACS changes must be reviewed/applied manually on the actual PCIe bridges.'
