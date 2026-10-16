#!/usr/bin/env bash
set -euo pipefail

# Ensure working directory is repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "================================================================="
echo "⚡ Launching vLLM Timestral Forecaster Daemon (8x B300 HGX Host)"
echo "   CPU Core Pinning: Cores 0-7"
echo "   Inference Device: Host CPU (0 MB GPU VRAM Impact)"
echo "   Metrics Endpoint: http://0.0.0.0:9100/metrics"
echo "================================================================="

exec taskset -c 0-7 uv run python -m src.service "$@"
