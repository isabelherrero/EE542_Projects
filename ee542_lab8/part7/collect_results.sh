#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$HOME/lab8_results"
"$(dirname "$0")/check_hw.sh" > "$HOME/lab8_results/final_inventory.log" 2>&1 || true
cp "$(dirname "$0")/../config/node.env" "$HOME/lab8_results/node.env" 2>/dev/null || true
out="$HOME/lab8_$(hostname)_results.tar.gz"
tar -czf "$out" -C "$HOME" lab8_results
ls -lh "$out"
echo 'Download archive to your computer BEFORE terminating CloudLab.'
