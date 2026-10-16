"""Timestral time-series forecaster strictly pinned to host CPU."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ForecastLeadTimeSummary:
    """Summary of quantiles for a specific future lead time."""

    lead_time_label: str  # e.g., "5m", "15m", "30m"
    lead_time_seconds: int
    step_index: int
    target_timestamp: float
    # KV cache quantiles [0.0, 1.0]
    kv_cache_p10: float
    kv_cache_p50: float
    kv_cache_p90: float
    kv_cache_p95: float
    # Request queue depth quantiles (requests count)
    requests_waiting_p10: float
    requests_waiting_p50: float
    requests_waiting_p90: float
    requests_waiting_p95: float


@dataclass
class ForecastResult:
    """Result of a Timestral forecasting inference pass."""

    model_name: str
    inference_latency_ms: float
    cpu_threads: int
    device: str
    prediction_length: int
    step_seconds: int
    timestamps: np.ndarray  # (H,) future unix timestamps
    # Quantiles tensor: shape (2, H, 21) where variate 0=kv_cache, 1=requests_waiting
    quantiles: np.ndarray
    quantile_levels: tuple[float, ...]
    # Key quantiles unpacked: shape (H,)
    kv_cache_p10: np.ndarray
    kv_cache_p50: np.ndarray
    kv_cache_p90: np.ndarray
    kv_cache_p95: np.ndarray
    requests_waiting_p10: np.ndarray
    requests_waiting_p50: np.ndarray
    requests_waiting_p90: np.ndarray
    requests_waiting_p95: np.ndarray
    # Summaries for standard lead times
    summaries: list[ForecastLeadTimeSummary] = field(default_factory=list)
    # Early warning flag
    breach_warning: bool = False
    breach_lead_time_seconds: int | None = None
    breach_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Converts result to a JSON-serializable dictionary."""
        return {
            "model_name": self.model_name,
            "inference_latency_ms": round(self.inference_latency_ms, 2),
            "cpu_threads": self.cpu_threads,
            "device": self.device,
            "prediction_length": self.prediction_length,
            "step_seconds": self.step_seconds,
            "breach_warning": self.breach_warning,
            "breach_lead_time_seconds": self.breach_lead_time_seconds,
            "breach_reason": self.breach_reason,
            "summaries": [
                {
                    "lead_time": s.lead_time_label,
                    "lead_time_seconds": s.lead_time_seconds,
                    "kv_cache": {
                        "p10": round(s.kv_cache_p10, 4),
                        "p50": round(s.kv_cache_p50, 4),
                        "p90": round(s.kv_cache_p90, 4),
                        "p95": round(s.kv_cache_p95, 4),
                    },
                    "requests_waiting": {
                        "p10": round(s.requests_waiting_p10, 2),
                        "p50": round(s.requests_waiting_p50, 2),
                        "p90": round(s.requests_waiting_p90, 2),
                        "p95": round(s.requests_waiting_p95, 2),
                    },
                }
                for s in self.summaries
            ],
        }


class TimestralVLLMForecaster:
    """Zero-overhead CPU forecaster utilizing Mistral's Timestral foundation model."""

    DEFAULT_QUANTILE_LEVELS = (
        0.01,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        0.99,
    )

    def __init__(
        self,
        model_name: str = "mistralai/Timestral-28M",
        fallback_model_name: str = "mistralai/Timestral-5M",
        cpu_threads: int = 8,
        prediction_length: int = 120,  # 120 steps * 15s = 30 minutes
        step_seconds: int = 15,
        cache_breach_threshold: float = 0.85,
        queue_breach_threshold: float = 8.0,
        mock_mode: bool = False,
    ) -> None:
        self.requested_model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.cpu_threads = cpu_threads
        self.prediction_length = prediction_length
        self.step_seconds = step_seconds
        self.cache_breach_threshold = cache_breach_threshold
        self.queue_breach_threshold = queue_breach_threshold
        self.mock_mode = mock_mode

        # Enforce PyTorch CPU thread count and isolation
        torch.set_num_threads(self.cpu_threads)
        os.environ["OMP_NUM_THREADS"] = str(self.cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.cpu_threads)

        self.model: Any = None
        self.active_model_name = "mock" if mock_mode else model_name
        self.device = "cpu"

        if not self.mock_mode:
            self._load_model()

    def _load_model(self) -> None:
        """Loads Timestral model on CPU with fallback to 5M."""
        from timestral import TimestralForecaster

        torch.set_num_threads(self.cpu_threads)
        logger.info(
            "Loading Timestral forecaster '%s' on CPU (%d threads)...",
            self.requested_model_name,
            self.cpu_threads,
        )

        for candidate in [self.requested_model_name, self.fallback_model_name]:
            try:
                t0 = time.perf_counter()
                self.model = TimestralForecaster.from_pretrained(candidate, device="cpu")
                load_time = time.perf_counter() - t0
                self.active_model_name = candidate
                logger.info(
                    "Successfully loaded '%s' on CPU in %.2fs (0 GPU memory allocated)",
                    candidate,
                    load_time,
                )
                return
            except Exception as e:
                logger.warning("Failed loading candidate model '%s': %s", candidate, e)

        logger.error(
            "Both '%s' and '%s' failed to load. Enabling synthetic forecaster fallback.",
            self.requested_model_name,
            self.fallback_model_name,
        )
        self.mock_mode = True
        self.active_model_name = "synthetic_fallback"

    def _mock_forecast(
        self,
        context: np.ndarray,
        prediction_length: int,
    ) -> tuple[np.ndarray, tuple[float, ...]]:
        """Simulates Timestral output tensors for unit testing or offline environments."""
        # context shape: (1, 2, T)
        H = prediction_length
        Q = len(self.DEFAULT_QUANTILE_LEVELS)
        quantiles = np.zeros((1, 2, H, Q), dtype=np.float32)

        last_cache = float(context[0, 0, -1])
        last_waiting = float(context[0, 1, -1])

        # Generate realistic trajectory
        t = np.arange(1, H + 1)
        drift = 0.0005 * t
        spread = 0.002 * np.sqrt(t)

        for q_idx, q_level in enumerate(self.DEFAULT_QUANTILE_LEVELS):
            z = (q_level - 0.5) * 2.5
            quantiles[0, 0, :, q_idx] = last_cache + drift + z * spread
            quantiles[0, 1, :, q_idx] = last_waiting + z * 0.5 * np.sqrt(t)

        return quantiles, self.DEFAULT_QUANTILE_LEVELS

    def forecast(
        self,
        context_matrix: np.ndarray,
        known_covariates: np.ndarray | None = None,
        last_timestamp: float | None = None,
    ) -> ForecastResult:
        """Executes Timestral probabilistic forward pass on CPU.

        Args:
            context_matrix: shape (1, 2, T) or (2, T)
            known_covariates: shape (1, C, T + H)
            last_timestamp: Unix epoch timestamp of context[-1]
        """
        if context_matrix.ndim == 2:
            context_matrix = context_matrix[np.newaxis, :, :]

        if last_timestamp is None:
            last_timestamp = time.time()

        H = self.prediction_length
        future_timestamps = last_timestamp + (np.arange(1, H + 1) * self.step_seconds)

        t_start = time.perf_counter()

        if self.mock_mode or self.model is None:
            raw_quantiles, q_levels = self._mock_forecast(context_matrix, H)
        else:
            # Native Timestral inference
            raw_forecast = self.model.forecast(
                context_matrix,
                prediction_length=H,
                known_covariates=known_covariates,
            )
            raw_quantiles = raw_forecast.quantiles.detach().cpu().numpy()
            q_levels = tuple(float(x) for x in raw_forecast.quantile_levels)

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        # Reshape to (2, H, 21)
        quantiles = raw_quantiles[0]  # (2, H, 21)

        # Map quantiles
        p10_idx = q_levels.index(0.1) if 0.1 in q_levels else 2
        p50_idx = q_levels.index(0.5) if 0.5 in q_levels else 10
        p90_idx = q_levels.index(0.9) if 0.9 in q_levels else 18
        p95_idx = q_levels.index(0.95) if 0.95 in q_levels else 19

        # Clip values to physically meaningful ranges
        kv_p10 = np.clip(quantiles[0, :, p10_idx], 0.0, 1.0)
        kv_p50 = np.clip(quantiles[0, :, p50_idx], 0.0, 1.0)
        kv_p90 = np.clip(quantiles[0, :, p90_idx], 0.0, 1.0)
        kv_p95 = np.clip(quantiles[0, :, p95_idx], 0.0, 1.0)

        req_p10 = np.maximum(quantiles[1, :, p10_idx], 0.0)
        req_p50 = np.maximum(quantiles[1, :, p50_idx], 0.0)
        req_p90 = np.maximum(quantiles[1, :, p90_idx], 0.0)
        req_p95 = np.maximum(quantiles[1, :, p95_idx], 0.0)

        # Standard lead time summaries
        # 5m = 300s -> step 20 (index 19)
        # 15m = 900s -> step 60 (index 59)
        # 30m = 1800s -> step 120 (index 119)
        lead_time_configs = [
            ("5m", 300, min(19, H - 1)),
            ("15m", 900, min(59, H - 1)),
            ("30m", 1800, min(119, H - 1)),
        ]

        summaries: list[ForecastLeadTimeSummary] = []
        for label, sec, idx in lead_time_configs:
            if idx < H:
                summaries.append(
                    ForecastLeadTimeSummary(
                        lead_time_label=label,
                        lead_time_seconds=sec,
                        step_index=idx,
                        target_timestamp=float(future_timestamps[idx]),
                        kv_cache_p10=float(kv_p10[idx]),
                        kv_cache_p50=float(kv_p50[idx]),
                        kv_cache_p90=float(kv_p90[idx]),
                        kv_cache_p95=float(kv_p95[idx]),
                        requests_waiting_p10=float(req_p10[idx]),
                        requests_waiting_p50=float(req_p50[idx]),
                        requests_waiting_p90=float(req_p90[idx]),
                        requests_waiting_p95=float(req_p95[idx]),
                    )
                )

        # Check breach threshold on p90
        breach_warning = False
        breach_lead_time_sec = None
        breach_reason = ""

        for step_i in range(H):
            if kv_p90[step_i] >= self.cache_breach_threshold:
                breach_warning = True
                breach_lead_time_sec = (step_i + 1) * self.step_seconds
                breach_reason = (
                    f"KV Cache p90 forecast exceeds {self.cache_breach_threshold * 100:.0f}% "
                    f"at +{breach_lead_time_sec // 60}m ({kv_p90[step_i] * 100:.1f}%)"
                )
                break
            if req_p90[step_i] >= self.queue_breach_threshold:
                breach_warning = True
                breach_lead_time_sec = (step_i + 1) * self.step_seconds
                breach_reason = (
                    f"Queue depth p90 forecast exceeds {self.queue_breach_threshold:.0f} requests "
                    f"at +{breach_lead_time_sec // 60}m ({req_p90[step_i]:.1f})"
                )
                break

        return ForecastResult(
            model_name=self.active_model_name,
            inference_latency_ms=latency_ms,
            cpu_threads=self.cpu_threads,
            device=self.device,
            prediction_length=H,
            step_seconds=self.step_seconds,
            timestamps=future_timestamps,
            quantiles=quantiles,
            quantile_levels=q_levels,
            kv_cache_p10=kv_p10,
            kv_cache_p50=kv_p50,
            kv_cache_p90=kv_p90,
            kv_cache_p95=kv_p95,
            requests_waiting_p10=req_p10,
            requests_waiting_p50=req_p50,
            requests_waiting_p90=req_p90,
            requests_waiting_p95=req_p95,
            summaries=summaries,
            breach_warning=breach_warning,
            breach_lead_time_seconds=breach_lead_time_sec,
            breach_reason=breach_reason,
        )
