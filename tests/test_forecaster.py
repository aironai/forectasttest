"""Unit tests for TimestralVLLMForecaster: CPU pinning, quantile reshaping, breach detection."""

from __future__ import annotations

import numpy as np
import torch

from src.forecaster import ForecastResult, TimestralVLLMForecaster, format_lead_time


def test_cpu_device_pinning() -> None:
    forecaster = TimestralVLLMForecaster(
        cpu_threads=8,
        prediction_length=60,
        step_seconds=15,
        mock_mode=True,
    )

    assert forecaster.device == "cpu"
    assert forecaster.cpu_threads == 8
    assert torch.get_num_threads() == 8


def test_quantile_tensor_reshaping_and_quantiles() -> None:
    H = 60
    forecaster = TimestralVLLMForecaster(
        prediction_length=H,
        step_seconds=15,
        mock_mode=True,
    )

    context = np.random.uniform(0.1, 0.4, size=(1, 2, 256)).astype(np.float32)
    known = np.zeros((1, 5, 256 + H), dtype=np.float32)

    result = forecaster.forecast(context, known_covariates=known, last_timestamp=1000.0)

    assert isinstance(result, ForecastResult)
    assert result.device == "cpu"
    assert result.prediction_length == H
    assert len(result.timestamps) == H

    # Full quantiles tensor shape must be (2, H, 21)
    assert result.quantiles.shape == (2, H, 21)
    assert len(result.quantile_levels) == 21

    # Key quantiles must have shape (H,)
    assert result.kv_cache_p10.shape == (H,)
    assert result.kv_cache_p50.shape == (H,)
    assert result.kv_cache_p90.shape == (H,)
    assert result.kv_cache_p95.shape == (H,)

    assert result.requests_waiting_p10.shape == (H,)
    assert result.requests_waiting_p50.shape == (H,)
    assert result.requests_waiting_p90.shape == (H,)
    assert result.requests_waiting_p95.shape == (H,)

    # Value ranges must be valid
    assert np.all(result.kv_cache_p10 >= 0.0) and np.all(result.kv_cache_p90 <= 1.0)
    assert np.all(result.requests_waiting_p10 >= 0.0)


def test_lead_time_summaries() -> None:
    forecaster = TimestralVLLMForecaster(prediction_length=120, mock_mode=True)
    context = np.full((1, 2, 100), 0.2, dtype=np.float32)

    result = forecaster.forecast(context, last_timestamp=1000.0)

    summary_labels = [s.lead_time_label for s in result.summaries]
    assert "5m" in summary_labels
    assert "15m" in summary_labels
    assert "30m" in summary_labels

    s15 = next(s for s in result.summaries if s.lead_time_label == "15m")
    assert s15.lead_time_seconds == 900
    assert 0.0 <= s15.kv_cache_p50 <= 1.0
    assert s15.requests_waiting_p50 >= 0.0


def test_breach_detection_triggered() -> None:
    forecaster = TimestralVLLMForecaster(
        prediction_length=60,
        cache_breach_threshold=0.85,
        mock_mode=True,
    )
    # High initial cache load that will cross 0.85 in p90
    context = np.full((1, 2, 64), 0.86, dtype=np.float32)

    result = forecaster.forecast(context, last_timestamp=1000.0)

    assert result.breach_warning is True
    assert result.breach_lead_time_seconds is not None
    assert "KV Cache" in result.breach_reason or "Queue" in result.breach_reason


def test_breach_detection_normal() -> None:
    forecaster = TimestralVLLMForecaster(
        prediction_length=60,
        cache_breach_threshold=0.85,
        queue_breach_threshold=10.0,
        mock_mode=True,
    )
    # Low initial load
    context = np.full((1, 2, 64), 0.05, dtype=np.float32)

    result = forecaster.forecast(context, last_timestamp=1000.0)

    assert result.breach_warning is False
    assert result.breach_lead_time_seconds is None
    assert result.breach_reason == ""


def test_forecast_result_serialization() -> None:
    forecaster = TimestralVLLMForecaster(prediction_length=30, mock_mode=True)
    context = np.full((1, 2, 32), 0.1, dtype=np.float32)

    result = forecaster.forecast(context, last_timestamp=1000.0)
    data = result.to_dict()

    assert data["prediction_length"] == 30
    assert data["cpu_threads"] == 8
    assert data["device"] == "cpu"
    assert "summaries" in data
    assert isinstance(data["summaries"], list)


def test_lead_time_summaries_skip_beyond_horizon() -> None:
    # 60 steps * 15s = 15 minutes: the 30m summary must not exist
    forecaster = TimestralVLLMForecaster(prediction_length=60, mock_mode=True)
    result = forecaster.forecast(np.full((1, 2, 64), 0.2, dtype=np.float32), last_timestamp=0.0)

    assert [s.lead_time_label for s in result.summaries] == ["5m", "15m"]
    s15 = result.summaries[1]
    assert s15.step_index == 59
    assert s15.lead_time_seconds == 900
    assert s15.target_timestamp == 900.0


def test_lead_time_summaries_respect_step_seconds() -> None:
    forecaster = TimestralVLLMForecaster(prediction_length=60, step_seconds=30, mock_mode=True)
    result = forecaster.forecast(np.full((1, 2, 64), 0.2, dtype=np.float32), last_timestamp=0.0)

    by_label = {s.lead_time_label: s for s in result.summaries}
    assert by_label["5m"].step_index == 9
    assert by_label["15m"].step_index == 29
    assert by_label["30m"].step_index == 59
    assert all(s.target_timestamp == s.lead_time_seconds for s in result.summaries)


def test_format_lead_time() -> None:
    assert format_lead_time(15) == "15s"
    assert format_lead_time(45) == "45s"
    assert format_lead_time(60) == "1m"
    assert format_lead_time(660) == "11m"
    assert format_lead_time(675) == "11m15s"


def test_imminent_breach_reason_is_not_zero_minutes() -> None:
    forecaster = TimestralVLLMForecaster(prediction_length=60, mock_mode=True)
    # Already above threshold: breach is on the very first step (+15s)
    result = forecaster.forecast(np.full((1, 2, 64), 0.95, dtype=np.float32), last_timestamp=0.0)

    assert result.breach_lead_time_seconds == 15
    assert "+15s" in result.breach_reason
    assert "+0m" not in result.breach_reason


def test_synthetic_flags() -> None:
    forecaster = TimestralVLLMForecaster(prediction_length=8, mock_mode=True)
    result = forecaster.forecast(np.full((1, 2, 16), 0.2, dtype=np.float32), last_timestamp=0.0)

    assert result.synthetic_model is True
    assert result.synthetic_input is False
    assert result.is_synthetic is True
    data = result.to_dict()
    assert data["synthetic_model"] is True
    assert data["synthetic_input"] is False
