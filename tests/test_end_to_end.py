"""End-to-end integration tests with synthetic vLLM spike patterns and alert validation."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from src.service import ForecastingDaemon, create_app


def test_end_to_end_spike_pattern_and_breach_alert() -> None:
    # 1. Initialize daemon in mock mode for fast deterministic integration test
    daemon = ForecastingDaemon(
        prometheus_url="http://mock-offline-prom:9999",
        mock_mode=True,
        context_steps=128,
        horizon_steps=60,  # 15 minutes
        step_seconds=15,
        forecast_interval_seconds=60,
    )

    # 2. Simulate baseline normal state
    normal_now = time.time()
    ts, cache, waiting = daemon.collector.generate_synthetic_context(
        normal_now, current_cache=0.15, current_waiting=0.0
    )
    spike_now = normal_now + 60.0
    ts_spike, cache_spike, waiting_spike = daemon.collector.generate_synthetic_context(
        spike_now, current_cache=0.92, current_waiting=14.0
    )

    # Inject baseline directly into collector
    def mock_normal(now: float, current_cache: float = 0.15, current_waiting: float = 0.0):
        return ts, cache, waiting

    daemon.collector.generate_synthetic_context = mock_normal

    result_normal = daemon.run_single_cycle()
    assert result_normal.breach_warning is False
    assert result_normal.inference_latency_ms < 500.0  # CPU forward pass latency budget

    metrics_normal = daemon.exporter.export_text()
    assert 'reason="none"} 0.0' in metrics_normal

    # 3. Simulate sudden traffic spike (burst arrival: KV cache 92%, queue depth 14)
    def mock_spike(now: float, current_cache: float = 0.92, current_waiting: float = 14.0):
        return ts_spike, cache_spike, waiting_spike

    daemon.collector.generate_synthetic_context = mock_spike

    result_spike = daemon.run_single_cycle()

    # 4. Verify breach warning is triggered
    assert result_spike.breach_warning is True
    assert result_spike.breach_lead_time_seconds is not None
    assert result_spike.breach_lead_time_seconds > 0
    assert len(result_spike.breach_reason) > 0

    # 5. Verify Prometheus export reflects breach alert
    metrics_spike = daemon.exporter.export_text()
    assert "timestral_forecast_breach_warning{" in metrics_spike
    assert "} 1.0" in metrics_spike
    assert "timestral_forecast_breach_lead_time_seconds" in metrics_spike

    # 6. Verify FastAPI inspection endpoints
    app = create_app(daemon)
    client = TestClient(app)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["breach_warning"] is True

    forecast_resp = client.get("/api/forecast")
    assert forecast_resp.status_code == 200
    forecast_data = forecast_resp.json()
    assert forecast_data["breach_warning"] is True
    assert forecast_data["breach_lead_time_seconds"] == result_spike.breach_lead_time_seconds


def test_end_to_end_real_timestral_cpu_pass() -> None:
    """End-to-end integration test with the real loaded Timestral-5M model on CPU."""
    daemon = ForecastingDaemon(
        model_name="mistralai/Timestral-5M",
        cpu_threads=8,
        context_steps=128,
        horizon_steps=60,
        step_seconds=15,
        mock_mode=False,
    )

    result = daemon.run_single_cycle()

    assert result.device == "cpu"
    assert result.model_name == "mistralai/Timestral-5M"
    assert result.cpu_threads == 8
    # Verify quantiles output tensor dimensions
    assert result.quantiles.shape == (2, 60, 21)
    assert len(result.summaries) >= 2

    # Check exported Prometheus metrics
    exported = daemon.exporter.export_text()
    assert "vllm_kv_cache_forecast" in exported
    assert "timestral_gpu_vram_bytes 0.0" in exported
