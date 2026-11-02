# Part 7 — CloudLab Setup and Execution Commands

## Commands for Running the Scripts

## 1. Check Hardware

Run on both nodes:

```bash
hostname
ip -br addr
df -h /mydata
bash check_hw.sh
```

## 2. Install NVIDIA Driver and Dependencies

Run on both nodes:

```bash
sudo apt-get update

sudo apt-get install -y build-essential dkms \
    linux-headers-$(uname -r) \
    rdma-core ibverbs-utils libibverbs-dev pciutils

sudo apt-get install -y nvidia-driver-580-open
sudo reboot
```

After reboot:

```bash
nvidia-smi
ibv_devices

cd ~/ee542_lab8/part7
bash setup.sh
```

## 3. Configure IOMMU

Run on both nodes:

```bash
cd ~/ee542_lab8/part7

bash configure_iommu.sh --apply
sudo reboot
```

After reconnecting:

```bash
cd ~/ee542_lab8/part7

cat /proc/cmdline
bash post_reboot.sh
```

## 4. Verify Hardware After Reboot

Run on both nodes:

```bash
bash check_hw.sh

lspci -tv
nvidia-smi topo -m
```

## 5. Configure the RDMA Device

Run on both nodes:

```bash
cd ~/ee542_lab8/part7
mkdir -p config
nano config/node.env
```

Configuration used:

```bash
GPU="25:00.0"
NIC="41:00.0"
RDMA="rocep65s0"
GID="3"
SERVER="10.10.1.1"
```

## 6. Build GPUNetIO

Run on both nodes:

```bash
bash build_gpunetio.sh

ls ~/gpunetio/install/examples
```

## 7. GPU-to-GPU RDMA Latency


**node0 — Server:**

```bash
bash latency.sh server 1
```

**node1 — Client:**

```bash
bash latency.sh client 1
```

**node0 — Server:**

```bash
bash latency.sh server 2
```

**node1 — Client:**

```bash
bash latency.sh client 2
```


## 8. GPU-to-GPU RDMA Bandwidth

### PUT Bandwidth

**node0 — Server:**

```bash
bash bandwidth.sh server put
```

**node1 — Client:**

```bash
bash bandwidth.sh client put
```

### WRITE Bandwidth

**node0 — Server:**

```bash
bash bandwidth.sh server write
```

**node1 — Client:**

```bash
bash bandwidth.sh client write
```

## 9. CPU-Initiated RDMA Baseline

### CPU Latency

**node0 — Server:**

```bash
bash rdma_baseline.sh server lat
```

**node1 — Client:**

```bash
bash rdma_baseline.sh client lat
```

### CPU Bandwidth

**node0 — Server:**

```bash
bash rdma_baseline.sh server bw
```

**node1 — Client:**

```bash
bash rdma_baseline.sh client bw
```

## 10. DOCA UDP Receive Sample

Run on node0 only:

```bash
bash udp_sample.sh
```
