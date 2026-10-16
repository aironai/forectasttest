"""Main forecasting daemon and HTTP metrics service for vLLM on 8x B300 HGX."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from src.collector import MetricsCollector
from src.exporter import MetricsExporter
from src.forecaster import ForecastResult, TimestralVLLMForecaster
from src.pusher import PushgatewayClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("timestral_daemon")


class ForecastingDaemon:
    """Coordinates metric collection, CPU-bound Timestral inference, and metric export."""

    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        vllm_url: str = "http://localhost:8000",
        model_name: str = "mistralai/Timestral-28M",
        fallback_model_name: str = "mistralai/Timestral-5M",
        cpu_threads: int = 8,
        context_steps: int = 512,
        horizon_steps: int = 120,
        step_seconds: int = 15,
        forecast_interval_seconds: int = 60,
        pushgateway_url: str | None = None,
        mock_mode: bool = False,
    ) -> None:
        self.prometheus_url = prometheus_url
        self.vllm_url = vllm_url
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.cpu_threads = cpu_threads
        self.context_steps = context_steps
        self.horizon_steps = horizon_steps
        self.step_seconds = step_seconds
        self.forecast_interval_seconds = forecast_interval_seconds
        self.pushgateway_url = pushgateway_url
        self.mock_mode = mock_mode

        self.collector = MetricsCollector(
            prometheus_url=self.prometheus_url,
            vllm_url=self.vllm_url,
            context_steps=self.context_steps,
            horizon_steps=self.horizon_steps,
            step_seconds=self.step_seconds,
        )

        self.forecaster = TimestralVLLMForecaster(
            model_name=self.model_name,
            fallback_model_name=self.fallback_model_name,
            cpu_threads=self.cpu_threads,
            prediction_length=self.horizon_steps,
            step_seconds=self.step_seconds,
            mock_mode=self.mock_mode,
        )

        self.exporter = MetricsExporter()
        self.pusher = PushgatewayClient(self.pushgateway_url) if self.pushgateway_url else None

        self.latest_result: ForecastResult | None = None
        self.is_running = False
        self._task: asyncio.Task[None] | None = None

    def run_single_cycle(self) -> ForecastResult:
        """Runs one full forecasting cycle synchronously with detailed profiling."""
        cycle_start = time.perf_counter()

        # Step 1: Collect historical window & temporal covariates
        t_col_start = time.perf_counter()
        window = self.collector.collect()
        col_duration_ms = (time.perf_counter() - t_col_start) * 1000.0

        # Step 2: CPU-bound Timestral forward pass
        last_ts = float(window.timestamps[-1])
        result = self.forecaster.forecast(
            context_matrix=window.target_matrix,
            known_covariates=window.covariates,
            last_timestamp=last_ts,
        )

        # Step 3: Export metrics to Prometheus registry
        t_exp_start = time.perf_counter()
        self.exporter.update_metrics(result)
        if self.pusher:
            self.pusher.push(self.exporter.registry)
        exp_duration_ms = (time.perf_counter() - t_exp_start) * 1000.0

        total_duration_ms = (time.perf_counter() - cycle_start) * 1000.0
        self.latest_result = result

        # Step 4: Structured logging
        s15 = next((s for s in result.summaries if s.lead_time_label == "15m"), None)
        s30 = next((s for s in result.summaries if s.lead_time_label == "30m"), None)

        cache_15_str = (
            f"p10={s15.kv_cache_p10 * 100:.1f}%, p50={s15.kv_cache_p50 * 100:.1f}%, "
            f"p90={s15.kv_cache_p90 * 100:.1f}%"
            if s15
            else "N/A"
        )
        cache_30_str = (
            f"p50={s30.kv_cache_p50 * 100:.1f}%, p90={s30.kv_cache_p90 * 100:.1f}%"
            if s30
            else "N/A"
        )
        queue_15_str = (
            f"p50={s15.requests_waiting_p50:.1f}, p90={s15.requests_waiting_p90:.1f}"
            if s15
            else "N/A"
        )

        breach_status = (
            f"⚠️ ALERT ({result.breach_reason})" if result.breach_warning else "OK (Normal)"
        )

        logger.info(
            "Forecast complete: model=%s | latency=%.2fms (threads=%d) | "
            "collect=%.1fms, export=%.1fms, total=%.1fms | "
            "KV+15m: [%s] | KV+30m: [%s] | Queue+15m: [%s] | Status: %s | RSS: %.1fMB | VRAM: 0MB",
            result.model_name,
            result.inference_latency_ms,
            result.cpu_threads,
            col_duration_ms,
            exp_duration_ms,
            total_duration_ms,
            cache_15_str,
            cache_30_str,
            queue_15_str,
            breach_status,
            self.exporter.get_process_rss_bytes() / (1024 * 1024),
        )

        return result

    async def start_loop(self) -> None:
        """Starts the recurring forecasting loop."""
        self.is_running = True
        logger.info(
            "Starting Timestral daemon loop (interval=%ds, context=%d steps, horizon=%d steps)",
            self.forecast_interval_seconds,
            self.context_steps,
            self.horizon_steps,
        )

        # Run first cycle immediately
        try:
            self.run_single_cycle()
        except Exception as e:
            logger.error("Initial forecast cycle error: %s", e, exc_info=True)

        while self.is_running:
            try:
                await asyncio.sleep(self.forecast_interval_seconds)
                if not self.is_running:
                    break
                await asyncio.to_thread(self.run_single_cycle)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in forecasting daemon loop: %s", e, exc_info=True)

    def stop(self) -> None:
        """Stops the recurring forecasting loop."""
        self.is_running = False
        if self._task and not self._task.done():
            self._task.cancel()


# Global daemon instance for FastAPI app
daemon_instance: ForecastingDaemon | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages daemon background task lifecycle with FastAPI."""
    global daemon_instance
    if daemon_instance:
        loop_task = asyncio.create_task(daemon_instance.start_loop())
        daemon_instance._task = loop_task
    yield
    if daemon_instance:
        daemon_instance.stop()


def create_app(daemon: ForecastingDaemon) -> FastAPI:
    """Builds FastAPI application with Prometheus exposition and inspection API."""
    global daemon_instance
    daemon_instance = daemon

    app = FastAPI(
        title="vLLM Timestral Forecaster",
        description="Zero-overhead CPU time-series forecasting for vLLM on 8x B300 HGX",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        """Prometheus metrics endpoint."""
        body = daemon.exporter.export_text()
        return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/health")
    def health_endpoint() -> dict[str, Any]:
        """Liveness and status check."""
        return {
            "status": "healthy",
            "model": daemon.forecaster.active_model_name,
            "cpu_threads": daemon.cpu_threads,
            "device": daemon.forecaster.device,
            "vram_allocated_bytes": 0,
            "latest_latency_ms": (
                round(daemon.latest_result.inference_latency_ms, 2)
                if daemon.latest_result
                else None
            ),
            "breach_warning": (
                daemon.latest_result.breach_warning if daemon.latest_result else False
            ),
        }

    @app.get("/api/forecast")
    def get_forecast() -> JSONResponse:
        """Returns JSON representation of the latest forecast result."""
        if not daemon.latest_result:
            return JSONResponse(status_code=503, content={"error": "No forecast generated yet"})
        return JSONResponse(content=daemon.latest_result.to_dict())

    @app.post("/api/trigger")
    def trigger_forecast() -> dict[str, Any]:
        """Manually triggers an immediate forecasting cycle."""
        result = daemon.run_single_cycle()
        return {"status": "success", "result": result.to_dict()}

    return app


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments and environment variables."""
    parser = argparse.ArgumentParser(
        description="Zero-overhead vLLM time-series forecasting daemon using Mistral Timestral"
    )
    parser.add_argument(
        "--prometheus-url",
        default=os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
        help="Prometheus server URL (default: http://localhost:9090)",
    )
    parser.add_argument(
        "--vllm-url",
        default=os.getenv("VLLM_URL", "http://localhost:8000"),
        help="vLLM metrics server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "0.0.0.0"),
        help="HTTP listen host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "9100")),
        help="HTTP listen port (default: 9100)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=int(os.getenv("FORECAST_INTERVAL", "60")),
        help="Forecast loop interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("MODEL_NAME", "mistralai/Timestral-28M"),
        help="Primary Timestral model name (default: mistralai/Timestral-28M)",
    )
    parser.add_argument(
        "--fallback-model",
        default=os.getenv("FALLBACK_MODEL", "mistralai/Timestral-5M"),
        help="Fallback Timestral model name (default: mistralai/Timestral-5M)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.getenv("CPU_THREADS", "8")),
        help="CPU threads allocated for PyTorch inference (default: 8)",
    )
    parser.add_argument(
        "--context-steps",
        type=int,
        default=int(os.getenv("CONTEXT_STEPS", "512")),
        help="Context window steps (default: 512 = 128m at 15s steps)",
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=int(os.getenv("HORIZON_STEPS", "120")),
        help="Forecast horizon steps (default: 120 = 30m at 15s steps)",
    )
    parser.add_argument(
        "--pushgateway-url",
        default=os.getenv("PUSHGATEWAY_URL", None),
        help="Prometheus Pushgateway URL (optional)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Enable mock mode for unit testing or offline environments",
    )
    return parser.parse_args()


def main() -> None:
    """Daemon CLI entry point."""
    args = parse_args()

    logger.info("=" * 70)
    logger.info("🚀 vLLM Timestral Forecaster Daemon (8x B300 HGX CPU Isolation)")
    logger.info("   Target Model: %s (Fallback: %s)", args.model_name, args.fallback_model)
    logger.info("   CPU Threads: %d | Device: CPU | GPU VRAM Impact: 0 MB", args.threads)
    logger.info(
        "   Context: %d steps | Horizon: %d steps (30 min)",
        args.context_steps,
        args.horizon_steps,
    )
    logger.info("   Prometheus Source: %s | Exporter Port: %d", args.prometheus_url, args.port)
    logger.info("=" * 70)

    daemon = ForecastingDaemon(
        prometheus_url=args.prometheus_url,
        vllm_url=args.vllm_url,
        model_name=args.model_name,
        fallback_model_name=args.fallback_model,
        cpu_threads=args.threads,
        context_steps=args.context_steps,
        horizon_steps=args.horizon_steps,
        forecast_interval_seconds=args.interval,
        pushgateway_url=args.pushgateway_url,
        mock_mode=args.mock,
    )

    app = create_app(daemon)

    def handle_exit(sig: int, frame: Any) -> None:
        logger.info("Shutdown signal received (%s), terminating daemon...", sig)
        daemon.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
