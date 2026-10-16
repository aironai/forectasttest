"""Unit tests for MetricsCollector: normalization, NaN handling, and covariates."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.collector import (
    DATA_SOURCE_PROMETHEUS,
    DATA_SOURCE_SYNTHETIC,
    InsufficientHistoryError,
    MetricsCollector,
    ObservationWindow,
)


def test_temporal_covariates_generation() -> None:
    collector = MetricsCollector(context_steps=512, horizon_steps=120, step_seconds=15)
    total_steps = 512 + 120
    start_time = 1700000000.0  # arbitrary epoch timestamp

    covariates = collector.generate_temporal_covariates(start_time, total_steps, 15)

    assert covariates.shape == (1, 5, total_steps)
    assert covariates.dtype == np.float32

    # Check bounds of sine and cosine features
    tod_sin = covariates[0, 0, :]
    tod_cos = covariates[0, 1, :]
    dow_sin = covariates[0, 2, :]
    dow_cos = covariates[0, 3, :]
    biz_ind = covariates[0, 4, :]

    assert np.all(tod_sin >= -1.0) and np.all(tod_sin <= 1.0)
    assert np.all(tod_cos >= -1.0) and np.all(tod_cos <= 1.0)
    assert np.all(dow_sin >= -1.0) and np.all(dow_sin <= 1.0)
    assert np.all(dow_cos >= -1.0) and np.all(dow_cos <= 1.0)

    # Check business indicator is strictly binary {0.0, 1.0}
    unique_vals = set(np.unique(biz_ind))
    assert unique_vals.issubset({0.0, 1.0})


def test_align_series_to_grid_and_nan_handling() -> None:
    collector = MetricsCollector()
    target_ts = np.array([100.0, 115.0, 130.0, 145.0, 160.0])

    # Irregular raw points with a gap
    raw_points = [(100.0, 0.1), (130.0, 0.4), (160.0, 0.8)]
    aligned = collector.align_series_to_grid(raw_points, target_ts)

    assert len(aligned) == 5
    assert aligned[0] == 0.1
    assert aligned[2] == 0.4
    assert aligned[4] == 0.8
    # Intermediate interpolated points
    assert 0.1 < aligned[1] < 0.4
    assert 0.4 < aligned[3] < 0.8
    assert not np.isnan(aligned).any()

    # Empty raw points fallback
    empty_aligned = collector.align_series_to_grid([], target_ts)
    assert len(empty_aligned) == 5
    assert (empty_aligned == 0.0).all()


def test_synthetic_context_generation() -> None:
    collector = MetricsCollector(context_steps=256, step_seconds=15)
    now = time.time()
    current_cache = 0.45
    current_waiting = 3.0

    ts, cache, waiting = collector.generate_synthetic_context(
        now, current_cache=current_cache, current_waiting=current_waiting
    )

    assert len(ts) == 256
    assert len(cache) == 256
    assert len(waiting) == 256

    # Verify physical bounds
    assert np.all(cache >= 0.0) and np.all(cache <= 1.0)
    assert np.all(waiting >= 0.0)

    # Anchoring to current observation
    assert abs(cache[-1] - current_cache) < 1e-4
    assert abs(waiting[-1] - current_waiting) < 1e-4


def test_collector_full_collection_synthetic_fallback() -> None:
    collector = MetricsCollector(
        prometheus_url="http://invalid-prom-url:9999",
        vllm_url="http://invalid-vllm-url:9999",
        context_steps=128,
        horizon_steps=32,
        step_seconds=15,
        synthetic_fallback=True,
    )

    window: ObservationWindow = collector.collect()

    assert isinstance(window, ObservationWindow)
    assert window.context_steps == 128
    assert window.horizon_steps == 32
    assert window.target_matrix.shape == (1, 2, 128)
    assert window.covariates.shape == (1, 5, 128 + 32)
    assert len(window.timestamps) == 128
    assert len(window.kv_cache_usage) == 128
    assert len(window.requests_waiting) == 128

    # All values are finite and non-NaN
    assert np.all(np.isfinite(window.target_matrix))
    assert np.all(np.isfinite(window.covariates))


def test_collector_prometheus_query_mocked() -> None:
    collector = MetricsCollector(
        prometheus_url="http://mock-prom:9090",
        context_steps=10,
        horizon_steps=4,
        step_seconds=15,
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"__name__": "vllm:kv_cache_usage_perc"},
                    "values": [[1000.0 + i * 15, "0.15"] for i in range(10)],
                }
            ]
        },
    }

    with patch("requests.get", return_value=mock_resp):
        pts = collector.query_prometheus_range("vllm:kv_cache_usage_perc", 1000.0, 1150.0, 15)
        assert len(pts) == 10
        assert pts[0] == (1000.0, 0.15)


def _prom_response(n_points: int, value: str = "0.15") -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "status": "success",
        "data": {
            "result": [
                {"metric": {}, "values": [[1000.0 + i * 15, value] for i in range(n_points)]}
            ]
        },
    }
    return resp


def test_collect_raises_without_history_when_fallback_disabled() -> None:
    collector = MetricsCollector(
        prometheus_url="http://invalid-prom-url:9999",
        vllm_url="http://invalid-vllm-url:9999",
        context_steps=64,
        horizon_steps=16,
    )
    assert collector.synthetic_fallback is False

    with pytest.raises(InsufficientHistoryError, match="at least 32"):
        collector.collect()


def test_collect_marks_synthetic_window() -> None:
    collector = MetricsCollector(
        prometheus_url="http://invalid-prom-url:9999",
        vllm_url="http://invalid-vllm-url:9999",
        context_steps=64,
        horizon_steps=16,
        synthetic_fallback=True,
    )
    window = collector.collect()
    assert window.data_source == DATA_SOURCE_SYNTHETIC
    assert window.is_synthetic is True


def test_collect_uses_max_aggregation_and_marks_real_history() -> None:
    collector = MetricsCollector(prometheus_url="http://mock-prom:9090", context_steps=10)

    with patch("requests.get", return_value=_prom_response(10)) as mock_get:
        window = collector.collect()

    queries = [c.kwargs["params"]["query"] for c in mock_get.call_args_list]
    assert queries == ["max(vllm:kv_cache_usage_perc)", "max(vllm:num_requests_waiting)"]
    assert window.data_source == DATA_SOURCE_PROMETHEUS
    assert window.is_synthetic is False


def test_direct_scrape_matches_exact_names_and_takes_max() -> None:
    collector = MetricsCollector(vllm_url="http://mock-vllm:8000")
    body = "\n".join(
        [
            "# HELP vllm:kv_cache_usage_perc KV-cache usage.",
            "# TYPE vllm:kv_cache_usage_perc gauge",
            'vllm:kv_cache_usage_perc{engine="0",model_name="kimi-k3"} 0.42',
            'vllm:kv_cache_usage_perc{engine="1",model_name="kimi-k3"} 0.61',
            # Similar names that must not be matched
            'vllm:kv_cache_usage_perc_created{engine="0"} 1792165000.0',
            'vllm:num_requests_waiting_by_reason{reason="capacity"} 99.0',
            'vllm:num_requests_waiting{engine="0"} 3.0',
            'vllm:num_requests_waiting{engine="1"} 5.0 1792165000000',
        ]
    )
    resp = MagicMock(status_code=200, text=body)

    with patch("requests.get", return_value=resp):
        assert collector.sample_direct_vllm_metrics() == (0.61, 5.0)


def test_direct_scrape_without_kv_metric_returns_none() -> None:
    collector = MetricsCollector(vllm_url="http://mock-vllm:8000")
    resp = MagicMock(status_code=200, text="vllm:num_requests_waiting 3.0\n")
    with patch("requests.get", return_value=resp):
        assert collector.sample_direct_vllm_metrics() is None
