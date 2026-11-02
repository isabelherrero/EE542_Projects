#!/usr/bin/env bash
set -euo pipefail
# Run on EACH node. Install driver first if nvidia-smi does not work.
if ! nvidia-smi >/dev/null 2>&1; then
  echo 'GPU driver not ready. See README: install NVIDIA 580+ OPEN kernel driver and reboot.'
  exit 1
fi
if ! mountpoint -q /mydata; then echo '/mydata is not mounted. Stop.'; exit 1; fi
sudo apt-get update
sudo apt-get install -y build-essential git make gcc g++ pkg-config curl wget gnupg   ca-certificates linux-headers-$(uname -r) rdma-core ibverbs-utils   libibverbs-dev perftest pciutils ethtool check kmod meson ninja-build   docker.io python3 python3-pip
# CUDA toolkit only; leave installed host driver untouched.
if ! [ -x /usr/local/cuda-13.0/bin/nvcc ]; then
  cd /tmp
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get install -y cuda-toolkit-13-0
fi
# Configure Docker before downloading the large DOCA image.
if [ -e /etc/docker/daemon.json ]; then
  echo '/etc/docker/daemon.json exists. Review it manually before changing Docker data-root.'
  cat /etc/docker/daemon.json
  exit 1
fi
sudo systemctl stop docker || true
sudo mkdir -p /mydata/docker /etc/docker
printf '{"data-root":"/mydata/docker"}
' | sudo tee /etc/docker/daemon.json
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl enable --now docker
sudo systemctl restart docker
# Build the exact GDRCopy release in the lab.
sudo mkdir -p /opt/mellanox
if [ ! -d /opt/mellanox/gdrcopy/.git ]; then
  sudo git clone https://github.com/NVIDIA/gdrcopy.git /opt/mellanox/gdrcopy
fi
cd /opt/mellanox/gdrcopy
sudo git fetch --tags
sudo git checkout v2.5.2
sudo make -j "$(nproc)"
echo 'Setup finished. Configure IOMMU, reboot, then run post_reboot.sh.'
