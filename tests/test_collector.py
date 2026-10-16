"""Unit tests for MetricsCollector: normalization, NaN handling, and covariates."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np

from src.collector import MetricsCollector, ObservationWindow


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
