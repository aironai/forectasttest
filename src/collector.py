"""Historical metrics collector for vLLM and Prometheus with temporal covariates."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)


@dataclass
class ObservationWindow:
    """Represents a standardized time series observation window for Timestral."""

    timestamps: np.ndarray  # (T,) unix timestamps
    kv_cache_usage: np.ndarray  # (T,) values in [0, 1]
    requests_waiting: np.ndarray  # (T,) values >= 0
    covariates: np.ndarray  # (1, C, T + H) known temporal covariates
    target_matrix: np.ndarray  # (1, V, T) formatted for Timestral (V=2: kv_cache, requests_waiting)
    interval_seconds: int = 15
    context_steps: int = 512
    horizon_steps: int = 120


class MetricsCollector:
    """Collects and prepares time-series metrics from Prometheus or direct vLLM metrics."""

    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        vllm_url: str = "http://localhost:8000",
        context_steps: int = 512,
        horizon_steps: int = 120,
        step_seconds: int = 15,
        synthetic_fallback: bool = True,
    ) -> None:
        self.prometheus_url = prometheus_url.rstrip("/")
        self.vllm_url = vllm_url.rstrip("/")
        self.context_steps = context_steps
        self.horizon_steps = horizon_steps
        self.step_seconds = step_seconds
        self.synthetic_fallback = synthetic_fallback

        # Internal buffer for rolling direct observations if Prometheus isn't accumulating
        self._local_buffer: list[tuple[float, float, float]] = []

    def generate_temporal_covariates(
        self, start_timestamp: float, total_steps: int, step_seconds: int
    ) -> np.ndarray:
        """Generates continuous sine/cosine and calendar covariates over context and horizon.

        Returns:
            np.ndarray: shape (1, 5, total_steps)
            Covariates:
                0: Time-of-day sin (daily periodicity = 86400s)
                1: Time-of-day cos
                2: Day-of-week sin (weekly periodicity = 7 days)
                3: Day-of-week cos
                4: Business-hours indicator (1.0 during 08:00-18:00 UTC on weekdays, else 0.0)
        """
        timestamps = start_timestamp + np.arange(total_steps) * step_seconds
        seconds_in_day = timestamps % 86400.0
        days_in_week = (timestamps // 86400 + 4) % 7.0  # Unix epoch 1970-01-01 was Thursday (day 4)

        tod_sin = np.sin(2.0 * math.pi * seconds_in_day / 86400.0)
        tod_cos = np.cos(2.0 * math.pi * seconds_in_day / 86400.0)
        dow_sin = np.sin(2.0 * math.pi * days_in_week / 7.0)
        dow_cos = np.cos(2.0 * math.pi * days_in_week / 7.0)

        # Business hours: weekdays (day 0-4 = Mon-Fri), hours 08:00 to 18:00 UTC
        hour_of_day = seconds_in_day / 3600.0
        is_weekday = days_in_week < 5.0
        is_work_hour = (hour_of_day >= 8.0) & (hour_of_day < 18.0)
        biz_indicator = (is_weekday & is_work_hour).astype(np.float32)

        covariates = np.stack([tod_sin, tod_cos, dow_sin, dow_cos, biz_indicator], axis=0).astype(
            np.float32
        )
        return covariates[np.newaxis, :, :]  # (1, 5, total_steps)

    def query_prometheus_range(
        self, query: str, start_time: float, end_time: float, step: int
    ) -> list[tuple[float, float]]:
        """Queries Prometheus range endpoint and extracts sorted (timestamp, value) pairs."""
        endpoint = f"{self.prometheus_url}/api/v1/query_range"
        params = {
            "query": query,
            "start": str(start_time),
            "end": str(end_time),
            "step": f"{step}s",
        }
        try:
            resp = requests.get(endpoint, params=params, timeout=5)
            if resp.status_code == 200:
                data: dict[str, Any] = resp.json()
                if data.get("status") == "success" and data.get("data", {}).get("result"):
                    result_series = data["data"]["result"][0]
                    values = result_series.get("values", [])
                    return [(float(ts), float(val)) for ts, val in values]
        except Exception as e:
            logger.warning("Prometheus range query '%s' failed: %s", query, e)
        return []

    def sample_direct_vllm_metrics(self) -> tuple[float, float] | None:
        """Directly parses /metrics endpoint of vLLM if Prometheus is not available."""
        endpoint = f"{self.vllm_url}/metrics"
        try:
            resp = requests.get(endpoint, timeout=3)
            if resp.status_code == 200:
                kv_cache = 0.0
                waiting = 0.0
                found_cache = False
                for line in resp.text.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # Match kv cache usage metrics
                    if (
                        line.startswith("vllm:kv_cache_usage_perc ")
                        or line.startswith("vllm:gpu_cache_usage_factor ")
                        or "kv_cache_usage_perc" in line
                        or "gpu_cache_usage_factor" in line
                    ):
                        parts = line.split()
                        kv_cache = float(parts[-1])
                        found_cache = True
                    elif (
                        line.startswith("vllm:num_requests_waiting ")
                        or "num_requests_waiting" in line
                    ):
                        parts = line.split()
                        waiting = float(parts[-1])
                if found_cache:
                    return kv_cache, waiting
        except Exception as e:
            logger.debug("Direct vLLM scrape failed: %s", e)
        return None

    def generate_synthetic_context(
        self, current_time: float, current_cache: float = 0.15, current_waiting: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates realistic synthetic history matching vLLM traffic patterns.

        Used as an automated fallback or offline PoC mode.
        Produces diurnal traffic, random prompt burst arrivals, KV cache allocations,
        and request completions.
        """
        T = self.context_steps
        dt = self.step_seconds
        timestamps = current_time - (T - 1 - np.arange(T)) * dt

        # Base diurnal activity curve
        time_phase = (timestamps % 86400) / 86400 * 2 * math.pi
        diurnal = 0.2 + 0.15 * np.sin(time_phase - math.pi / 2)  # Peak in afternoon

        # Random Poisson burst events
        rng = np.random.default_rng(seed=int(current_time) % 100000)
        cache_series = np.zeros(T, dtype=np.float32)
        waiting_series = np.zeros(T, dtype=np.float32)

        cache_level = max(0.05, min(0.85, current_cache))
        waiting_level = max(0.0, current_waiting)

        for i in range(T):
            # Bursty incoming requests
            burst = rng.exponential(scale=1.2) if rng.random() < 0.15 else 0.0
            drain = 0.02 * (1.0 + rng.random() * 0.5)

            cache_level = cache_level + burst * 0.08 - drain
            cache_level = max(0.02, min(0.95, cache_level + diurnal[i] * 0.01))

            if cache_level > 0.80:
                waiting_level = max(0.0, waiting_level + rng.integers(0, 3) - 1.0)
            else:
                waiting_level = max(0.0, waiting_level - 1.0)

            cache_series[i] = cache_level
            waiting_series[i] = waiting_level

        # Ensure the final point matches current observation
        cache_series[-1] = max(0.0, min(1.0, current_cache))
        waiting_series[-1] = max(0.0, current_waiting)

        return timestamps, cache_series, waiting_series

    def align_series_to_grid(
        self, raw_points: list[tuple[float, float]], target_timestamps: np.ndarray
    ) -> np.ndarray:
        """Aligns irregular or sparse time series points to the regular target timestamp grid."""
        if not raw_points:
            return np.zeros(len(target_timestamps), dtype=np.float32)

        pts = sorted(raw_points, key=lambda x: x[0])
        src_ts = np.array([p[0] for p in pts], dtype=np.float64)
        src_val = np.array([p[1] for p in pts], dtype=np.float32)

        # Linear interpolation across target grid, extrapolating with nearest valid values
        aligned = np.interp(target_timestamps, src_ts, src_val, left=src_val[0], right=src_val[-1])
        return np.nan_to_num(aligned, nan=0.0).astype(np.float32)

    def collect(self) -> ObservationWindow:
        """Collects the observation window from Prometheus, direct vLLM, or synthetic fallback."""
        now = time.time()
        T = self.context_steps
        H = self.horizon_steps
        dt = self.step_seconds

        target_timestamps = now - (T - 1 - np.arange(T)) * dt
        start_time = target_timestamps[0]
        end_time = now

        # 1. Attempt to query Prometheus
        kv_cache_points = self.query_prometheus_range(
            "vllm:kv_cache_usage_perc", start_time, end_time, dt
        )
        if not kv_cache_points:
            kv_cache_points = self.query_prometheus_range(
                "vllm:gpu_cache_usage_factor", start_time, end_time, dt
            )

        waiting_points = self.query_prometheus_range(
            "vllm:num_requests_waiting", start_time, end_time, dt
        )

        has_sufficient_history = len(kv_cache_points) >= (T // 2) and len(waiting_points) >= (
            T // 2
        )

        if has_sufficient_history:
            logger.info("Collected %d history points from Prometheus", len(kv_cache_points))
            kv_cache = self.align_series_to_grid(kv_cache_points, target_timestamps)
            requests_waiting = self.align_series_to_grid(waiting_points, target_timestamps)
            timestamps = target_timestamps
        else:
            # Check if direct live point exists from Prometheus or direct vLLM
            latest_cache = 0.07  # Realistic baseline default
            latest_waiting = 0.0

            if kv_cache_points:
                latest_cache = kv_cache_points[-1][1]
            if waiting_points:
                latest_waiting = waiting_points[-1][1]

            direct = self.sample_direct_vllm_metrics()
            if direct is not None:
                latest_cache, latest_waiting = direct

            logger.info(
                "Prometheus history sparse (%d points); applying synthetic replay with live anchor "
                "(cache=%.3f, waiting=%.1f)",
                len(kv_cache_points),
                latest_cache,
                latest_waiting,
            )
            timestamps, kv_cache, requests_waiting = self.generate_synthetic_context(
                now, current_cache=latest_cache, current_waiting=latest_waiting
            )

        # Normalize and clip
        kv_cache = np.clip(kv_cache, 0.0, 1.0).astype(np.float32)
        requests_waiting = np.maximum(requests_waiting, 0.0).astype(np.float32)

        # Covariates over context (T) + horizon (H)
        total_steps = T + H
        covariates = self.generate_temporal_covariates(timestamps[0], total_steps, dt)

        # Target matrix for Timestral: shape (1, 2, T)
        # Variate 0: kv_cache, Variate 1: requests_waiting
        target_matrix = np.stack([kv_cache, requests_waiting], axis=0)[np.newaxis, :, :]

        return ObservationWindow(
            timestamps=timestamps,
            kv_cache_usage=kv_cache,
            requests_waiting=requests_waiting,
            covariates=covariates,
            target_matrix=target_matrix,
            interval_seconds=dt,
            context_steps=T,
            horizon_steps=H,
        )
