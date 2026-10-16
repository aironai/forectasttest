# 🔮 Zero-Overhead vLLM Predictive Maintenance with Mistral Timestral on 8x B300 HGX

A zero-GPU-overhead, CPU-isolated time-series forecasting daemon utilizing Mistral's **Timestral** foundation models (`mistralai/Timestral-28M` and `Timestral-5M`) to forecast vLLM metrics (KV-cache usage and request queue depth) 15–30 minutes into the future on NVIDIA B300 HGX systems.

---

## 📋 Executive Summary

On ultra-high-throughput multi-GPU clusters serving frontier models (such as **Kimi K3** on **8x NVIDIA B300 GPUs**), **GPU memory (VRAM) is precious and must never be wasted on observability or telemetry tasks**. 

Traditional ML-based predictive maintenance tools often compete for GPU compute or allocate CUDA memory, causing VRAM fragmentation or triggering out-of-memory (OOM) failures under heavy inference traffic.

This system achieves **Zero Overhead** by:
1. **Host CPU Execution**: Pinning PyTorch inference strictly to **8 host CPU threads** (isolated via `taskset -c 0-7`).
2. **0 MB GPU VRAM Impact**: B300 GPUs maintain 100% of their VRAM dedicated to model weights and KV-cache blocks.
3. **Sub-100ms Foundation Model Latency**: Timestral-5M produces a 30-minute forecast in **~42 ms** and Timestral-28M in **~99 ms** on 8 pinned host cores (see [benchmarks](#-performance-benchmarks-on-host-cpu)), a negligible cost for a cycle that runs once per minute.
4. **Multivariate Probabilistic Predictions**: Predicts 21 non-crossing quantiles (p10 optimistic bound, p50 median, p90/p95 pessimistic upper bound) over a 15–30 minute forecast horizon.
5. **Early Warning Capacity Breaches**: Sends Slack and/or email alerts (with cooldown and recovery messages) before KV-cache saturation or queue blow-ups trigger request drops or preemption cascades.
6. **Honest Inputs**: Forecasts only run on real Prometheus history by default. Synthetic history is an explicit opt-in for demos and is always flagged in metrics, the API, and logs, and never triggers notifications unless you ask it to.

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
       │   Scrape Interval: 15s | Retention: Prometheus default  │
       └─────────────────────────────────────────────────────────┘
                                    │ (Query API :9090)
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │            Timestral Forecaster Daemon (Host CPU Cores 0-7)            │
 │                                                                        │
 │  1. MetricsCollector:                                                  │
 │     - Extracts 512-step (128 min) history: max() across vLLM engines  │
 │     - Too little history -> cycle skipped (or synthetic, if opted in) │
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
 │                                                                        │
 │  4. EarlyWarningNotifier:                                              │
 │     - Slack webhook and/or SMTP email on breach / recovery             │
 │     - Cooldown de-duplication, retry on failed delivery                │
 └────────────────────────────────────────────────────────────────────────┘
          │                         │                         │
          ▼ (:9100/metrics)         ▼ (:9100/api/forecast)    ▼ (Slack / SMTP)
 ┌──────────────────────────┐ ┌─────────────────────────┐ ┌──────────────────────┐
 │ Grafana Dashboard        │ │ Autoscalers / Incident  │ │ On-call engineers    │
 │ - KV-Cache vs Forecast   │ │ Automations             │ │ - Breach warnings    │
 │ - Queue Forecast/Breach  │ │ (FastAPI JSON API)      │ │ - Recovery notices   │
 │ - CPU / RAM / 0 VRAM     │ │                         │ │                      │
 └──────────────────────────┘ └─────────────────────────┘ └──────────────────────┘
```

---

## 🖼️ Visual Dashboard & Forecast Anatomy

### 1. Live Grafana Dashboard
The dashboard visualizes real-time metrics side-by-side with Timestral's forward projections, latency history, and zero-GPU telemetry:

![Grafana Dashboard Preview](assets/grafana_dashboard_preview.png)

### 2. Anatomy of the Forecast Cone
How past telemetry transitions into probabilistic predictive bounds:

![Forecast Interpretation Guide](assets/forecast_concept_diagram.png)

---

## 🧠 Beginner's Guide: How to Read & Interpret the Forecast

### Why Does KV-Cache Saturation Matter?
When serving frontier Large Language Models (like **Kimi K3** on 8x B300 GPUs), every active request requires storing token Key-Value pairs in GPU memory (the **KV-cache**). 

- **At < 75% Capacity**: vLLM operates smoothly at peak generation throughput.
- **At > 85% Capacity (Danger Zone)**: vLLM risks exhausting block allocations. When full, vLLM must **preempt** (pause and evict) running user requests, causing inter-token latency to spike from **20ms to 2000ms+** or dropping requests entirely with `503 Service Unavailable`.
- **The Core Problem**: Conventional monitoring only alerts *after* the cache exceeds 85%, which is too late to spin up new replicas or reroute incoming queries.

### What Do the 3 Forecast Bands Mean?

| Visual Element | Metric Meaning | Intuitive Interpretation |
| :--- | :--- | :--- |
| **Solid Green Line** | **Actual KV-Cache** | What is happening right now and over the past 2 hours. |
| **Dashed Blue Line** | **p50 (Median Forecast)** | The **"most expected path"**. If incoming traffic follows normal trends, this is where the system will be in 15–30 minutes. |
| **Bottom Ribbon Edge** | **p10 (Optimistic Bound)** | The **"best-case floor"**. If active sessions finish fast and user traffic slows down, usage won't drop below this line. |
| **Top Ribbon Edge** | **p90 / p95 (Pessimistic Upper Bound)** | The **"worst-case safety barrier"**. If user prompt bursts continue, long context documents arrive, or sessions linger, usage could reach this ceiling. |

### The Power of "Early Warning Lead Time"
Notice the shaded **Early Warning Window** in the diagram:
- At **$T = 0$ (Now)**, actual KV-cache usage is only **34%** (everything looks green and safe to standard monitors).
- However, Timestral projects the **p90 worst-case ceiling crossing 85% in 15 minutes**.
- **Actionable Lead Time**: Platform engineers and automated autoscalers have a **15-minute window** to warm up a secondary B300 node or steer new long prompts via Envoy *before any request suffers degradation*.

---

## 🔬 Simulated Real-World Scenario: Anatomy of a Breach

Here is a step-by-step example showing what happens during a real traffic surge:

```
Timeline (UTC)    Actual KV Cache    Timestral +15m Forecast (p50 / p90)     System Status & Automated Action
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
14:00:00          18.2% (Quiet)      p50: 21.0%  |  p90: 28.5%               🟢 HEALTHY (No Action Needed)
14:05:00          24.5% (Steady)     p50: 27.2%  |  p90: 36.1%               🟢 HEALTHY
14:10:00          38.0% (Burst)      p50: 55.4%  |  p90: 71.0%               🟢 HEALTHY (Observing upward drift)
14:15:00          54.2% (Surge!)     p50: 74.0%  |  p90: 89.4% (BREACH!)     🔴 EARLY WARNING: BREACH IN 11 MIN
                                                                             ↳ Envoy sheds low-priority batch traffic
                                                                             ↳ Cluster autoscaler triggers node warmup
14:20:00          68.1% (Stabilized) p50: 71.5%  |  p90: 82.0% (Defused)     🟢 HEALTHY (Traffic safely managed)
```

### JSON Inspection via API
When the breach occurs, the daemon's `/api/forecast` REST endpoint immediately outputs structured telemetry for automated orchestrators:

```json
{
  "model_name": "mistralai/Timestral-28M",
  "inference_latency_ms": 35.9,
  "cpu_threads": 8,
  "device": "cpu",
  "breach_warning": true,
  "breach_lead_time_seconds": 660,
  "breach_reason": "KV Cache p90 forecast exceeds 85% at +11m (89.4%)",
  "synthetic_input": false,
  "synthetic_model": false,
  "summaries": [
    {
      "lead_time": "15m",
      "kv_cache": {
        "p10": 0.582,
        "p50": 0.740,
        "p90": 0.894,
        "p95": 0.921
      },
      "requests_waiting": {
        "p10": 0.0,
        "p50": 2.1,
        "p90": 9.4,
        "p95": 14.2
      }
    }
  ]
}
```

---

## ⚡ Performance Benchmarks on Host CPU

Measured on the HGX host (Intel Xeon 6776P), pinned with `taskset -c 0-7` and `torch.set_num_threads(8)`. Each row is the median and minimum of 20 warm `forecast()` calls (after 3 warm-up calls) with a 512-step context, both target variates, and the 5 known covariates. This is exactly what the daemon runs each cycle.

| Model | Context Steps | Horizon | Median Latency | Min Latency | Host RAM (RSS) | GPU VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Timestral-5M** | 512 (128m) | 32 steps (8m) | **62.0 ms** | 60.8 ms | ~580 MB | **0 MB** |
| **Timestral-5M** | 512 (128m) | 60 steps (15m) | **39.5 ms** | 39.2 ms | ~580 MB | **0 MB** |
| **Timestral-5M** | 512 (128m) | 120 steps (30m) | **42.4 ms** | 42.2 ms | ~580 MB | **0 MB** |
| **Timestral-28M** | 512 (128m) | 32 steps (8m) | **88.1 ms** | 87.8 ms | ~680 MB | **0 MB** |
| **Timestral-28M** | 512 (128m) | 60 steps (15m) | **94.2 ms** | 93.8 ms | ~680 MB | **0 MB** |
| **Timestral-28M** | 512 (128m) | 120 steps (30m) | **98.9 ms** | 98.6 ms | ~680 MB | **0 MB** |

> **Key Takeaways**
> - Both models finish well under one second of CPU time per 60-second cycle, without touching the NVIDIA B300 GPUs.
> - Timestral runs **two forward passes per forecast** (`symmetric_averaging=True`, which averages with the mirrored forecast of `-context`). That is why latency is roughly double a single pass.
> - The **first** forecast after start-up is slower (about 0.5 s for 28M) while PyTorch warms up. The `timestral_inference_duration_seconds` gauge shows the live value.
> - Earlier versions of this README listed ~16 ms (5M) and ~36 ms (28M). Those numbers could not be reproduced with the configuration above and have been replaced.

To reproduce, load each model with `TimestralVLLMForecaster(model_name=...)`, collect a window with `MetricsCollector(context_steps=512, horizon_steps=H, synthetic_fallback=True).collect()`, and time repeated `forecast()` calls under `taskset -c 0-7`.

---

## 🚀 Setup & Quickstart Guide

### Prerequisites
- Linux OS (Ubuntu 22.04+ or 24.04+)
- `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Git

### 1. Installation
Clone the repository and install all dependencies in seconds using `uv`:

```bash
git clone https://github.com/aironai/forectasttest.git
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

The suite has 37 tests. All of them run offline except `test_end_to_end_real_timestral_cpu_pass`, which loads the real `mistralai/Timestral-5M` checkpoint from the Hugging Face cache (or downloads it). Slack and SMTP are always mocked; tests never send real notifications.

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

> **Note:** By default the daemon **skips** a forecast cycle when Prometheus has less than half a context window of history, which is 64 minutes for the default 512 × 15 s window. On a fresh Prometheus you will see `Forecast cycle skipped: ...` warnings until enough history has built up. For a demo without history, add `--synthetic-fallback`. See [Data Sources & Synthetic Fallback](#-data-sources--synthetic-fallback).

### Configuration Reference

Every option can be set with a CLI flag or an environment variable. The flag wins when both are set.

| CLI Flag | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `--prometheus-url` | `PROMETHEUS_URL` | `http://localhost:9090` | Prometheus server used for history. |
| `--vllm-url` | `VLLM_URL` | `http://localhost:8000` | vLLM server, scraped directly only to anchor synthetic history. |
| `--host` | `HOST` | `0.0.0.0` | HTTP listen address. |
| `--port` | `PORT` | `9100` | HTTP listen port. |
| `--interval` | `FORECAST_INTERVAL` | `60` | Seconds between forecast cycles. |
| `--model-name` | `MODEL_NAME` | `mistralai/Timestral-28M` | Primary Timestral checkpoint. |
| `--fallback-model` | `FALLBACK_MODEL` | `mistralai/Timestral-5M` | Checkpoint tried if the primary fails to load. |
| `--threads` | `CPU_THREADS` | `8` | PyTorch CPU threads. |
| `--context-steps` | `CONTEXT_STEPS` | `512` | History length in 15 s steps (512 = 128 min). |
| `--horizon-steps` | `HORIZON_STEPS` | `120` | Forecast length in 15 s steps (120 = 30 min). |
| `--cache-threshold` | `CACHE_BREACH_THRESHOLD` | `0.85` | KV-cache p90 level (fraction) that counts as a breach. |
| `--queue-threshold` | `QUEUE_BREACH_THRESHOLD` | `8` | Waiting-queue p90 depth (requests) that counts as a breach. |
| `--synthetic-fallback` | `SYNTHETIC_FALLBACK` | off | Forecast on generated history when real history is too short. **Demo only.** |
| `--notify-on-synthetic` | `NOTIFY_ON_SYNTHETIC` | off | Also send notifications for synthetic forecasts. |
| `--pushgateway-url` | `PUSHGATEWAY_URL` | unset | Optional Pushgateway to push metrics to each cycle. |
| `--mock` | – | off | Use the built-in mock forecaster instead of Timestral (tests/offline). |

Environment flags accept `true`, `1` or `yes`. Notification settings are covered in [Early Warning Notifications](#-early-warning-notifications).

### 4. Verify Endpoints
Once running, verify the HTTP and Prometheus endpoints:

```bash
# Liveness and model telemetry
curl -s http://localhost:9100/health | jq

# Latest forecast JSON with quantiles and lead times (503 until the first forecast exists)
curl -s http://localhost:9100/api/forecast | jq

# Run a cycle immediately (503 with "status": "skipped" if history is insufficient)
curl -s -X POST http://localhost:9100/api/trigger | jq

# Prometheus metrics scrape output
curl -s http://localhost:9100/metrics | grep -E "vllm_kv_cache_forecast|timestral"
```

---

## 🌐 HTTP API Reference

| Method & Path | Description |
| :--- | :--- |
| `GET /metrics` | Prometheus exposition of all metrics below. |
| `GET /health` | Liveness and status. Always returns 200 while the process is up. |
| `GET /api/forecast` | Latest forecast as JSON (`summaries`, breach fields, `synthetic_input`, `synthetic_model`). Returns 503 before the first successful cycle. |
| `POST /api/trigger` | Runs one cycle immediately and returns its result. Returns 503 `{"status": "skipped"}` when real history is insufficient and synthetic fallback is off. Manual triggers and the background loop share a lock, so they never run at the same time. |

`/health` fields:

| Field | Meaning |
| :--- | :--- |
| `model` | Active model: a Timestral checkpoint, `mock`, or `synthetic_fallback` (both checkpoints failed to load). |
| `latest_latency_ms` | Inference latency of the latest forecast. |
| `breach_warning` | Whether the latest forecast breaches a threshold. |
| `synthetic_input` / `synthetic_model` | Whether the latest forecast used generated history / non-Timestral output (`null` before the first forecast). |
| `synthetic_fallback_enabled` | Whether `--synthetic-fallback` is on. |
| `last_cycle_error` | Why the most recent cycle was skipped, or `null` if it succeeded. |
| `alerting.enabled`, `.slack`, `.email` | Which notification channels are configured. |
| `alerting.in_breach_state` | Whether the notifier has an active (delivered) breach alert. |
| `alerting.last_status` | Result of the latest notification evaluation (`alert_type`, `dispatched`, `slack_sent`, `email_sent`). |

---

## 🧾 Data Sources & Synthetic Fallback

Each cycle builds a 512-step context from Prometheus with these range queries:

```promql
max(vllm:kv_cache_usage_perc)     # falls back to vllm:gpu_cache_usage_perc, then vllm:gpu_cache_usage_factor
max(vllm:num_requests_waiting)
```

**Why `max()`:** vLLM exports one series per `engine` / `model_name` label set, so a data-parallel deployment or several models produce several series. Taking `max()` forecasts the **most loaded** engine. Earlier versions read whichever series Prometheus happened to return first.

**When history is too short** (fewer than `context_steps / 2` points for either metric):

| Mode | Behaviour |
| :--- | :--- |
| Default | The cycle is **skipped**. No forecast is produced, `timestral_forecast_cycles_skipped_total{reason="insufficient_history"}` is incremented, `/health.last_cycle_error` explains why, and `/api/trigger` returns 503. The previous forecast (if any) stays published; check `timestral_last_forecast_timestamp_seconds` for its age. |
| `--synthetic-fallback` | A random but realistic history is generated around the latest real value. That value comes from Prometheus or a direct scrape of vLLM `/metrics`, which only uses exact metric-name matches and takes the max across engines. The forecast is marked `synthetic_input: true`, and `timestral_forecast_synthetic_input` is set to `1`. |

**If both Timestral checkpoints fail to load**, the daemon keeps running on a mock forecaster. `model` becomes `synthetic_fallback` and `timestral_forecast_synthetic_model` is set to `1`.

Synthetic forecasts can still set `timestral_forecast_breach_warning`. Filter them out with `and on() timestral_forecast_synthetic_input == 0` in your own alert rules. **Notifications are never sent for synthetic forecasts** unless `--notify-on-synthetic` is set.

---

## 🔔 Early Warning Notifications

The daemon checks every successful forecast for a breach and notifies on-call engineers via a **Slack incoming webhook**, **SMTP email**, or both. Notifications are off until at least one channel is configured. Startup logs `Early warning alerts enabled (slack=..., email=...)` when they are on.

### What counts as a breach
A forecast is a breach when, at any step in the horizon, either:
- **KV-cache p90 ≥ `CACHE_BREACH_THRESHOLD`** (default 0.85), or
- **waiting-queue p90 ≥ `QUEUE_BREACH_THRESHOLD`** (default 8 requests).

The first step that crosses sets the lead time (for example `+11m`, or `+15s` when the breach is imminent).

### Alert lifecycle

```
           breach forecast                  breach still forecast, cooldown elapsed
  SAFE ───────────────────────▶ BREACH ─────────────────────────────────────────┐
   ▲     sends "new_breach"       │  ▲      sends "recurring_breach"            │
   │                              │  └──────────────────────────────────────────┘
   │   forecast clear again       │
   └──────────────────────────────┘
          sends "recovery"
```

- **new_breach**: sent on the first breach forecast.
- **cooldown_active**: while a breach continues, nothing is re-sent until `ALERT_COOLDOWN_SECONDS` (default 900 s) have passed since the last alert.
- **recurring_breach**: re-sent once per cooldown while the breach continues.
- **recovery**: sent once when forecasts drop back below both thresholds for the whole horizon.
- **Failed delivery is retried.** State only advances when at least one channel accepts the message, so an outage at Slack or the mail server doesn't swallow the alert.
- Alerts report the **peak p90** for KV-cache and queue across the whole horizon, plus the lead time to the first crossing.
- Skipped cycles (insufficient history) don't change alert state.

### Configuration (environment variables)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SLACK_WEBHOOK_URL` | unset | Slack [incoming webhook](https://api.slack.com/messaging/webhooks) URL. Enables Slack. |
| `SMTP_HOST` | unset | SMTP server. Email is enabled when this **and** `ALERT_EMAIL_TO` are set. |
| `SMTP_PORT` | `587` | SMTP port. |
| `SMTP_USE_TLS` | `true` | Run `STARTTLS` after connecting. |
| `SMTP_USER` / `SMTP_PASSWORD` | unset | SMTP login (skipped if either is empty). |
| `ALERT_EMAIL_TO` | unset | Recipient address. |
| `ALERT_EMAIL_FROM` | `vllm-monitor@hgx.local` | Sender address. |
| `ALERT_COOLDOWN_SECONDS` | `900` | Minimum seconds between repeated breach alerts. |
| `GRAFANA_URL` | `http://localhost:3000/d/vllm-timestral-forecast/5846bdd` | Dashboard link in alerts. Set it to an address your on-call engineers can reach. |

Example (`/etc/default/vllm-timestral-forecast`, loaded by the systemd unit; keep it `chmod 600`):

```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T000/B000/XXXXXXXX
SMTP_HOST=smtp.example.com
SMTP_USER=alerts@example.com
SMTP_PASSWORD=change-me
ALERT_EMAIL_TO=oncall@example.com
ALERT_EMAIL_FROM=alerts@example.com
GRAFANA_URL=http://hgx-01.internal:3000/d/vllm-timestral-forecast
CACHE_BREACH_THRESHOLD=0.85
QUEUE_BREACH_THRESHOLD=8
```

### Observing notifications
- `timestral_alert_notifications_total{alert_type, channel, outcome}` counts each delivery attempt (`outcome` is `sent` or `failed`).
- `/health` → `alerting.last_status` shows the latest evaluation.
- The logs show `Slack alert successfully dispatched`, `Email alert sent to ...`, or `No notification channel delivered ... retrying next cycle`.

### Testing a channel end to end
Temporarily lower the threshold so that the current load counts as a breach, then trigger a cycle:

```bash
SLACK_WEBHOOK_URL=... uv run python -m src.service --cache-threshold 0.0 --port 9101 &
curl -s -X POST http://localhost:9101/api/trigger | jq .result.breach_warning
```

This needs real Prometheus history. Without it, add `--synthetic-fallback --notify-on-synthetic`.

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
| `timestral_forecast_breach_lead_time_seconds` | Gauge | None | Seconds until predicted breach (0 when there is none). |
| `timestral_last_forecast_timestamp_seconds` | Gauge | None | Unix time of the latest successful forecast. Use it to alert on staleness. |
| `timestral_forecast_synthetic_input` | Gauge | None | `1` if the latest forecast used generated history, `0` if it used real Prometheus history. |
| `timestral_forecast_synthetic_model` | Gauge | None | `1` if the latest forecast came from the mock/fallback generator instead of Timestral. |
| `timestral_forecast_cycles_skipped_total` | Counter | `reason` | Cycles that produced no forecast (`insufficient_history`). |
| `timestral_alert_notifications_total` | Counter | `alert_type`, `channel`, `outcome` | Notification delivery attempts (`new_breach` / `recurring_breach` / `recovery`; `slack` / `email`; `sent` / `failed`). |
| `timestral_inference_duration_histogram_seconds` | Histogram | `model` | Distribution of inference latency. |

**Lead-time labels:** `lead_time="5m" | "15m" | "30m"` are only exported when they fit inside the horizon. With `--horizon-steps 60` (15 min), no `30m` series exists. Earlier versions filled `30m` with the 15-minute value. Step indices are derived from the step size, so the labels stay correct if the step size changes.

---

## 📈 Grafana Dashboard Import Guide

A production-ready dashboard is included in `dashboards/vllm_timestral_forecast.json`.

### Pre-provisioned in Kimi K3 Cluster
If the HGX host's Kimi K3 cluster Grafana already has dashboard provisioning set up for this file (configured outside this repository), the dashboard appears in the `Kimi-K3` folder:
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

Secrets for notifications go in the optional `EnvironmentFile` `/etc/default/vllm-timestral-forecast` (see [Early Warning Notifications](#-early-warning-notifications)). Restart the service after you edit it:

```bash
sudo install -m 600 /dev/null /etc/default/vllm-timestral-forecast   # then edit it
sudo systemctl restart vllm-timestral-forecast.service
journalctl -u vllm-timestral-forecast.service -f
```

---

## ⚠️ Known Limitations & Security Notes

These are known and not yet addressed:

- **Grafana in `docker-compose.yml` is wide open**: `admin`/`admin` credentials, anonymous access with the **Admin** role, and port 3000 bound on all interfaces. Change this before exposing the host to any network.
- **The HTTP API has no authentication.** `POST /api/trigger` can be called by anyone who can reach port 9100. Bind to `--host 127.0.0.1` or firewall the port if that matters.
- **`timestral_forecast_breach_warning{reason=...}` uses the free-text reason as a label**, which creates a new series for each distinct breach message. Alert on `max(timestral_forecast_breach_warning) == 1` rather than on specific `reason` values.
- **Dashboard and datasource provisioning is not included.** The compose file mounts `dashboards/`, but Grafana needs a provisioning config to load it automatically. Otherwise use the manual import above.
- **Prometheus retention** uses the image default (15 days). Add `--storage.tsdb.retention.time` to the compose file to change it.

---

## 📄 License
This project is licensed under Apache 2.0. Mistral Timestral model weights are licensed under the Modified MIT license by Mistral AI.
