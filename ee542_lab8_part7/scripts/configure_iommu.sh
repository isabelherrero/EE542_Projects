#!/usr/bin/env bash
set -euo pipefail
# Changes next boot's AMD IOMMU settings. REVIEW before running.
if [ "${1:-}" != '--apply' ]; then
  echo 'Preview current setting:'
  grep '^GRUB_CMDLINE_LINUX_DEFAULT=' /etc/default/grub
  echo 'Run bash scripts/configure_iommu.sh --apply if approved. Then reboot manually.'
  exit 0
fi
sudo cp /etc/default/grub "/etc/default/grub.lab8.bak"
# Add arguments only once, preserving existing options.
if ! grep -q 'amd_iommu=off' /etc/default/grub; then
  sudo sed -i '/^GRUB_CMDLINE_LINUX_DEFAULT=/s/="/="amd_iommu=off iommu=off /' /etc/default/grub
fi
grep '^GRUB_CMDLINE_LINUX_DEFAULT=' /etc/default/grub
sudo update-grub
echo 'Now run sudo reboot on this node; reconnect and verify cat /proc/cmdline.'
