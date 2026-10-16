"""Unit tests for Prometheus exporter and FastAPI service endpoints."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from src.exporter import MetricsExporter
from src.forecaster import TimestralVLLMForecaster
from src.pusher import PushgatewayClient
from src.service import ForecastingDaemon, create_app


def test_metrics_exporter_updates_and_formatting() -> None:
    registry = CollectorRegistry()
    exporter = MetricsExporter(registry=registry)

    forecaster = TimestralVLLMForecaster(prediction_length=60, mock_mode=True)
    context = np.full((1, 2, 64), 0.25, dtype=np.float32)
    result = forecaster.forecast(context, last_timestamp=1000.0)

    exporter.update_metrics(result)
    metrics_text = exporter.export_text()

    # Check that required metric names exist in the output text
    assert "vllm_kv_cache_forecast{" in metrics_text
    assert 'lead_time="15m"' in metrics_text
    assert 'quantile="0.90"' in metrics_text
    assert "vllm_requests_waiting_forecast{" in metrics_text
    assert "timestral_inference_duration_seconds" in metrics_text
    assert "timestral_gpu_vram_bytes 0.0" in metrics_text
    assert "timestral_memory_rss_bytes" in metrics_text
    assert "timestral_inference_cpu_threads 8.0" in metrics_text


def test_fastapi_endpoints() -> None:
    daemon = ForecastingDaemon(mock_mode=True, context_steps=64, horizon_steps=30)
    # Run one cycle to populate latest_result
    daemon.run_single_cycle()

    app = create_app(daemon)
    client = TestClient(app)

    # Test /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data["status"] == "healthy"
    assert health_data["cpu_threads"] == 8
    assert health_data["vram_allocated_bytes"] == 0

    # Test /metrics
    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert "vllm_kv_cache_forecast" in metrics_resp.text

    # Test /api/forecast
    forecast_resp = client.get("/api/forecast")
    assert forecast_resp.status_code == 200
    f_data = forecast_resp.json()
    assert "summaries" in f_data
    assert "inference_latency_ms" in f_data

    # Test /api/trigger
    trigger_resp = client.post("/api/trigger")
    assert trigger_resp.status_code == 200
    assert trigger_resp.json()["status"] == "success"


def test_pushgateway_client() -> None:
    client = PushgatewayClient("http://mock-pushgateway:9091", job_name="test_job")
    registry = CollectorRegistry()

    with patch("src.pusher.push_to_gateway") as mock_push:
        res = client.push(registry)
        assert res is True
        mock_push.assert_called_once_with(
            "http://mock-pushgateway:9091", job="test_job", registry=registry
        )

    with patch("src.pusher.push_to_gateway", side_effect=RuntimeError("Network down")):
        res_fail = client.push(registry)
        assert res_fail is False
