
#!/usr/bin/env bash
# EE542 Lab 8 - Part 7.4: CloudLab Hardware Check
# Run on both node0 and node1.

set -uo pipefail

LOG_DIR="$HOME/lab8_part7_logs"
mkdir -p "$LOG_DIR"

NODE=$(hostname -s)
LOG_FILE="$LOG_DIR/check_hw_${NODE}.log"

exec > >(tee "$LOG_FILE") 2>&1

run_check() {
    echo
    echo "========================================"
    echo "$1"
    echo "========================================"
    shift
    "$@" || echo "[WARN] Command failed"
}

echo "EE542 Lab 8 - Part 7.4 Hardware Check"
echo "Hostname: $NODE"
echo "Date: $(date)"

# 1. Network interfaces
run_check "1. Network Interfaces" ip -br addr

# Find the interface with the 10.10.1.x address
IFACE=$(ip -o -4 addr show | awk '$4 ~ /^10\.10\.1\./ {print $2; exit}')

if [[ -n "$IFACE" ]]; then
    echo "Experiment Interface: $IFACE"
    run_check "2. NIC Driver / Firmware / PCI Address" ethtool -i "$IFACE"
else
    echo "[WARN] No 10.10.1.x interface found"
fi

# 3. RDMA devices
run_check "3. RDMA Devices" ibv_devices

# 4. NVIDIA/Mellanox network cards
run_check "4. Mellanox PCI Devices" bash -c \
    "lspci | grep -i mellanox"

# 5. Linux kernel
run_check "5. Kernel Version" uname -r

# 6. GPU-to-NIC PCIe topology
run_check "6. GPU / NIC PCIe Topology" nvidia-smi topo -m

# 7. GPU PCI address
run_check "7. GPU PCI Bus IDs" nvidia-smi \
    --query-gpu=name,pci.bus_id --format=csv

echo
echo "Hardware check completed."
echo "Log saved to: $LOG_FILE"
