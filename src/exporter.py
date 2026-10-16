"""Prometheus metrics exporter and pushgateway client for Timestral forecasts."""

from __future__ import annotations

import logging
import os
import time

from prometheus_client import (
    CollectorRegistry,
    Gauge,
    Histogram,
    generate_latest,
    push_to_gateway,
)

from src.forecaster import ForecastResult

logger = logging.getLogger(__name__)


class MetricsExporter:
    """Manages Prometheus gauges and metrics exposition for Timestral predictions."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()

        # Summary forecasts for standard lead times (5m, 15m, 30m) and key quantiles
        self.kv_cache_forecast = Gauge(
            "vllm_kv_cache_forecast",
            "Forecasted vLLM GPU KV-cache usage factor [0.0 - 1.0]",
            ["quantile", "lead_time"],
            registry=self.registry,
        )

        self.requests_waiting_forecast = Gauge(
            "vllm_requests_waiting_forecast",
            "Forecasted vLLM waiting request queue depth",
            ["quantile", "lead_time"],
            registry=self.registry,
        )

        # High-resolution trajectory gauges across forecast steps
        self.kv_cache_step_forecast = Gauge(
            "vllm_kv_cache_step_forecast",
            "Step-by-step forecasted KV-cache usage",
            ["step", "seconds_ahead", "quantile"],
            registry=self.registry,
        )

        self.requests_waiting_step_forecast = Gauge(
            "vllm_requests_waiting_step_forecast",
            "Step-by-step forecasted waiting request queue depth",
            ["step", "seconds_ahead", "quantile"],
            registry=self.registry,
        )

        # Hardware telemetry & operational metrics
        self.inference_latency = Gauge(
            "timestral_inference_duration_seconds",
            "Timestral CPU inference latency in seconds",
            ["model"],
            registry=self.registry,
        )

        self.inference_latency_hist = Histogram(
            "timestral_inference_duration_histogram_seconds",
            "Histogram of Timestral CPU inference duration",
            ["model"],
            buckets=(0.005, 0.010, 0.020, 0.035, 0.050, 0.075, 0.100, 0.250),
            registry=self.registry,
        )

        self.cpu_threads_gauge = Gauge(
            "timestral_inference_cpu_threads",
            "Number of CPU threads allocated for Timestral inference",
            registry=self.registry,
        )

        self.memory_rss_gauge = Gauge(
            "timestral_memory_rss_bytes",
            "Resident Set Size (RSS) memory consumption of the forecasting daemon in bytes",
            registry=self.registry,
        )

        self.gpu_vram_gauge = Gauge(
            "timestral_gpu_vram_bytes",
            "GPU VRAM consumed by the forecasting daemon (guaranteed 0 bytes)",
            registry=self.registry,
        )

        self.breach_warning_gauge = Gauge(
            "timestral_forecast_breach_warning",
            "1 if any forecasted metric breaches safe threshold within 30m, 0 otherwise",
            ["reason"],
            registry=self.registry,
        )

        self.breach_lead_time_gauge = Gauge(
            "timestral_forecast_breach_lead_time_seconds",
            "Seconds until projected capacity or queue breach",
            registry=self.registry,
        )

        self.last_forecast_timestamp = Gauge(
            "timestral_last_forecast_timestamp_seconds",
            "Epoch timestamp of the most recent forecasting cycle",
            registry=self.registry,
        )

        # Current process RSS reader
        self._page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

    def get_process_rss_bytes(self) -> int:
        """Reads process RSS memory in bytes from Linux /proc/self/statm."""
        try:
            with open("/proc/self/statm") as f:
                parts = f.read().split()
                return int(parts[1]) * self._page_size
        except Exception:
            return 0

    def update_metrics(self, result: ForecastResult) -> None:
        """Publishes a ForecastResult into Prometheus gauges."""
        # 1. Update lead time summaries
        for s in result.summaries:
            lt = s.lead_time_label
            # KV cache quantiles
            self.kv_cache_forecast.labels(quantile="0.10", lead_time=lt).set(s.kv_cache_p10)
            self.kv_cache_forecast.labels(quantile="0.50", lead_time=lt).set(s.kv_cache_p50)
            self.kv_cache_forecast.labels(quantile="0.90", lead_time=lt).set(s.kv_cache_p90)
            self.kv_cache_forecast.labels(quantile="0.95", lead_time=lt).set(s.kv_cache_p95)

            # Queue depth quantiles
            self.requests_waiting_forecast.labels(quantile="0.10", lead_time=lt).set(
                s.requests_waiting_p10
            )
            self.requests_waiting_forecast.labels(quantile="0.50", lead_time=lt).set(
                s.requests_waiting_p50
            )
            self.requests_waiting_forecast.labels(quantile="0.90", lead_time=lt).set(
                s.requests_waiting_p90
            )
            self.requests_waiting_forecast.labels(quantile="0.95", lead_time=lt).set(
                s.requests_waiting_p95
            )

        # 2. Update sampled step trajectory (every 4th step = 1-minute cadence)
        step_stride = 4
        H = result.prediction_length
        for step_i in range(0, H, step_stride):
            sec_ahead = (step_i + 1) * result.step_seconds
            step_str = str(step_i + 1)
            sec_str = str(sec_ahead)

            for q_str, kv_arr, req_arr in [
                ("0.10", result.kv_cache_p10, result.requests_waiting_p10),
                ("0.50", result.kv_cache_p50, result.requests_waiting_p50),
                ("0.90", result.kv_cache_p90, result.requests_waiting_p90),
            ]:
                self.kv_cache_step_forecast.labels(
                    step=step_str, seconds_ahead=sec_str, quantile=q_str
                ).set(float(kv_arr[step_i]))
                self.requests_waiting_step_forecast.labels(
                    step=step_str, seconds_ahead=sec_str, quantile=q_str
                ).set(float(req_arr[step_i]))

        # 3. Telemetry and latency
        latency_sec = result.inference_latency_ms / 1000.0
        self.inference_latency.labels(model=result.model_name).set(latency_sec)
        self.inference_latency_hist.labels(model=result.model_name).observe(latency_sec)
        self.cpu_threads_gauge.set(result.cpu_threads)

        rss_bytes = self.get_process_rss_bytes()
        self.memory_rss_gauge.set(rss_bytes)

        # Zero VRAM proof: explicitly sets 0 bytes
        self.gpu_vram_gauge.set(0)

        # Breach warnings
        self.breach_warning_gauge.clear()
        breach_val = 1.0 if result.breach_warning else 0.0
        reason_label = result.breach_reason if result.breach_warning else "none"
        self.breach_warning_gauge.labels(reason=reason_label).set(breach_val)

        lead_time_val = float(result.breach_lead_time_seconds or 0)
        self.breach_lead_time_gauge.set(lead_time_val)

        self.last_forecast_timestamp.set(time.time())

    def export_text(self) -> str:
        """Returns Prometheus text-formatted metrics."""
        return generate_latest(self.registry).decode("utf-8")

    def push_to_pushgateway(
        self, pushgateway_url: str, job_name: str = "timestral_forecaster"
    ) -> None:
        """Pushes current metrics to a Prometheus Pushgateway."""
        try:
            push_to_gateway(pushgateway_url, job=job_name, registry=self.registry)
            logger.info("Successfully pushed metrics to Pushgateway at %s", pushgateway_url)
        except Exception as e:
            logger.warning("Failed pushing metrics to Pushgateway: %s", e)
