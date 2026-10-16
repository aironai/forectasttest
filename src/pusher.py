"""Pushgateway publisher for Timestral predictions."""

from __future__ import annotations

import logging

from prometheus_client import CollectorRegistry, push_to_gateway

logger = logging.getLogger(__name__)


class PushgatewayClient:
    """Helper client to push Prometheus metrics directly to a Pushgateway."""

    def __init__(self, pushgateway_url: str, job_name: str = "vllm_forecast") -> None:
        self.pushgateway_url = pushgateway_url.rstrip("/")
        self.job_name = job_name

    def push(self, registry: CollectorRegistry) -> bool:
        """Pushes the given registry to the configured Pushgateway."""
        try:
            push_to_gateway(self.pushgateway_url, job=self.job_name, registry=registry)
            logger.debug("Pushed metrics to %s", self.pushgateway_url)
            return True
        except Exception as e:
            logger.warning("Error pushing to Pushgateway %s: %s", self.pushgateway_url, e)
            return False
