# 🔮 Zero-Overhead vLLM Predictive Maintenance with Mistral Timestral on 8x B300 HGX

A zero-GPU-overhead, CPU-isolated time-series forecasting daemon utilizing Mistral's **Timestral** foundation models (`mistralai/Timestral-28M` and `Timestral-5M`) to forecast vLLM metrics (KV-cache usage and request queue depth) 15–30 minutes into the future on NVIDIA B300 HGX systems.

---

## 📋 Executive Summary

On ultra-high-throughput multi-GPU clusters serving frontier models (such as **Kimi K3** on **8x NVIDIA B300 GPUs**), **GPU memory (VRAM) is precious and must never be wasted on observability or telemetry tasks**. 

Traditional ML-based predictive maintenance tools often compete for GPU compute or allocate CUDA memory, causing VRAM fragmentation or triggering out-of-memory (OOM) failures under heavy inference traffic.

This system achieves **Zero Overhead** by:
1. **Host CPU Execution**: Pinning PyTorch inference strictly to **8 host CPU threads** (isolated via `taskset -c 0-7`).
2. **0 MB GPU VRAM Impact**: B300 GPUs maintain 100% of their VRAM dedicated to model weights and KV-cache blocks.
3. **Sub-40ms Foundation Model Latency**: Timestral-5M executes a 30-minute forward pass in **~16 ms**, while Timestral-28M executes in **~35 ms**.
4. **Multivariate Probabilistic Predictions**: Predicts 21 non-crossing quantiles (p10 optimistic bound, p50 median, p90/p95 pessimistic upper bound) over a 15–30 minute forecast horizon.
5. **Early Warning Capacity Breaches**: Generates alerts before KV-cache saturation or queue blow-ups trigger request drops or preemption cascades.

---

## 🏗️ System Architecture

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      8x NVIDIA B300 HGX System                         │
 └────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │   vLLM Server (Kimi K3 TP=8, FlashInfer, TOKENSPEED)   │
       │   Metrics: vllm:kv_cache_usage_perc, requests_waiting   │
       └─────────────────────────────────────────────────────────┘
                                    │ (:8000/metrics)
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │          Prometheus Scraper (Docker Container)          │
       │          Scrape Interval: 15s | Retention: 30d          │
       └─────────────────────────────────────────────────────────┘
                                    │ (Query API :9090)
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │            Timestral Forecaster Daemon (Host CPU Cores 0-7)            │
 │                                                                        │
 │  1. MetricsCollector:                                                  │
 │     - Extracts 512-step (128 min) historical context window           │
 │     - Generates 5 continuous temporal covariates (time-of-day,        │
 │       day-of-week sine/cosine, business hour indicator)                │
 │                                                                        │
 │  2. TimestralVLLMForecaster (mistralai/Timestral-28M / 5M):            │
 │     - PyTorch CPU execution (torch.set_num_threads(8))                 │
 │     - 0 MB CUDA VRAM allocated                                         │
 │     - Generates 21 quantiles (p10, p50, p90, p95) for +5m, +15m, +30m │
 │     - Evaluates breach thresholds (p90 KV Cache > 85%, Queue > 8)     │
 │                                                                        │
 │  3. MetricsExporter:                                                   │
 │     - Exposes Prometheus gauges on :9100/metrics                       │
 │     - Optional Pushgateway publishing                                  │
 └────────────────────────────────────────────────────────────────────────┘
          │                                              │
          ▼ (:9100/metrics)                              ▼ (:9100/api/forecast)
 ┌───────────────────────────────────┐        ┌───────────────────────────┐
 │   Grafana Dashboard (Container)   │        │ Autoscalers / Incident    │
 │   - KV-Cache Actual vs Forecast   │        │ Alerting Automations      │
 │   - Queue Depth Forecast & Breach │        │ (FastAPI JSON API)        │
 │   - Host CPU / RAM / 0 VRAM Stats │        │                           │
 └───────────────────────────────────┘        └───────────────────────────┘
```

---

## ⚡ Performance Benchmarks on Host CPU

Measured on the 8x B300 HGX host machine running with `torch.set_num_threads(8)`:

| Model | Context Steps | Horizon | Median Latency | Min Latency | Host RAM (RSS) | GPU VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Timestral-5M** | 512 (128m) | 32 steps (8m) | **16.05 ms** | 14.92 ms | ~580 MB | **0 MB** |
| **Timestral-5M** | 512 (128m) | 60 steps (15m) | **16.76 ms** | 15.78 ms | ~581 MB | **0 MB** |
| **Timestral-5M** | 512 (128m) | 120 steps (30m) | **17.10 ms** | 16.96 ms | ~581 MB | **0 MB** |
| **Timestral-28M** | 512 (128m) | 32 steps (8m) | **31.43 ms** | 29.50 ms | ~680 MB | **0 MB** |
| **Timestral-28M** | 512 (128m) | 60 steps (15m) | **39.40 ms** | 32.31 ms | ~698 MB | **0 MB** |
| **Timestral-28M** | 512 (128m) | 120 steps (30m) | **35.90 ms** | 35.66 ms | ~698 MB | **0 MB** |

> **Key Takeaway**: Both foundation models comfortably execute under the **< 50ms latency budget** without touching the NVIDIA B300 GPUs.

---

## 🚀 Setup & Quickstart Guide

### Prerequisites
- Linux OS (Ubuntu 22.04+ or 24.04+)
- `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Git

### 1. Installation
Clone the repository and install all dependencies in seconds using `uv`:

```bash
git clone https://github.com/joenaess/forectasttest.git
cd forectasttest

# Sync dependencies and Timestral from GitHub
uv sync
```

### 2. Run Tests & Quality Gate
Verify complete functionality, CPU device pinning, and metric normalization:

```bash
# Check formatting and linting
uv run ruff check .
uv run ruff format --check .

# Run the complete test suite with coverage
uv run pytest -v --cov=src
```

### 3. Launch the Forecaster Daemon
Isolate inference to host CPU cores 0–7:

```bash
# Direct launcher script
./scripts/run_daemon.sh

# Or via taskset and uv
taskset -c 0-7 uv run python -m src.service \
  --prometheus-url http://localhost:9090 \
  --vllm-url http://localhost:8000 \
  --model-name mistralai/Timestral-28M \
  --threads 8 \
  --port 9100 \
  --interval 60
```

### 4. Verify Endpoints
Once running, verify the HTTP and Prometheus endpoints:

```bash
# Liveness and model telemetry
curl -s http://localhost:9100/health | jq

# Latest forecast JSON with quantiles and lead times
curl -s http://localhost:9100/api/forecast | jq

# Prometheus metrics scrape output
curl -s http://localhost:9100/metrics | grep -E "vllm_kv_cache_forecast|timestral"
```

---

## 📊 Exported Prometheus Metrics

| Metric Name | Type | Labels | Description |
| :--- | :--- | :--- | :--- |
| `vllm_kv_cache_forecast` | Gauge | `quantile`, `lead_time` | Predicted KV-cache saturation factor (`0.10`, `0.50`, `0.90`, `0.95` at `5m`, `15m`, `30m`). |
| `vllm_requests_waiting_forecast` | Gauge | `quantile`, `lead_time` | Predicted waiting request queue depth (`0.10`, `0.50`, `0.90`, `0.95` at `5m`, `15m`, `30m`). |
| `vllm_kv_cache_step_forecast` | Gauge | `step`, `seconds_ahead`, `quantile` | High-resolution trajectory step predictions. |
| `timestral_inference_duration_seconds` | Gauge | `model` | Timestral CPU forward pass duration in seconds (~0.016s - 0.035s). |
| `timestral_inference_cpu_threads` | Gauge | None | Number of CPU threads pinned (8). |
| `timestral_memory_rss_bytes` | Gauge | None | Host RAM memory footprint in bytes. |
| `timestral_gpu_vram_bytes` | Gauge | None | GPU VRAM consumed by the forecaster daemon (**0.0**). |
| `timestral_forecast_breach_warning` | Gauge | `reason` | `1.0` if projected p90 breaches threshold (KV-cache > 85% or queue > 8), else `0.0`. |
| `timestral_forecast_breach_lead_time_seconds` | Gauge | None | Seconds until predicted breach. |

---

## 📈 Grafana Dashboard Import Guide

A production-ready dashboard is included in `dashboards/vllm_timestral_forecast.json`.

### Pre-provisioned in Kimi K3 Cluster
If running on the HGX host with the Kimi K3 cluster, the dashboard is automatically auto-discovered under the `Kimi-K3` dashboard folder:
1. Open Grafana: `http://<HGX-IP>:3000`
2. Navigate to **Dashboards** $\rightarrow$ **Kimi-K3** $\rightarrow$ **vLLM Timestral Observability & Forecasting (8x B300 HGX)**.

### Manual Import Instructions
1. Open Grafana in your browser (`http://<HOST-IP>:3000`).
2. Click **Dashboards** in the left sidebar $\rightarrow$ **New** $\rightarrow$ **Import**.
3. Click **Upload dashboard JSON file** and select `dashboards/vllm_timestral_forecast.json`.
4. Ensure the **Prometheus** datasource is selected, then click **Import**.

### Dashboard Panels Included
- **Panel 1: KV-Cache Saturation: Real-Time vs. Forecasted Band (+15m)**
  - Solid green line: real-time actual KV-cache usage.
  - Dashed blue line: p50 (median) forecast.
  - Translucent shaded fan between p10 and p90.
  - Red threshold line at 85% capacity.
- **Panel 2: Request Queue Depth Forecast & Early Warning Breach**
  - Solid green line: actual waiting requests.
  - Dashed line: p50 forecast.
  - Dotted red line: p90 upper bound forecast.
- **Panel 3: Early Warning Status**
  - Color-coded indicator: Green (`HEALTHY`) vs Red (`EARLY WARNING: CAPACITY BREACH`).
- **Panel 4: CPU Footprint & Inference Latency**
  - CPU inference latency history in ms (< 40 ms).
  - Host RAM footprint vs GPU VRAM overhead (**flat 0 MB**).

---

## 🧪 Simulating Traffic Spikes

To generate realistic prompt bursts and observe the predictive fan project cache saturation in real time:

```bash
uv run python scripts/stress_workload.py \
  --url http://localhost:8000 \
  --model kimi-k3 \
  --concurrency 16 \
  --rounds 10
```

---

## 📦 Systemd Production Service

To run the daemon as a resilient background service managed by `systemd`:

```bash
sudo cp systemd/vllm-timestral-forecast.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-timestral-forecast.service

# Check service status
systemctl status vllm-timestral-forecast.service
```

---

## 📄 License
This project is licensed under Apache 2.0. Mistral Timestral model weights are licensed under the Modified MIT license by Mistral AI.
