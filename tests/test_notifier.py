"""Unit tests for EarlyWarningNotifier state machine, Slack/email delivery, and daemon wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from src.forecaster import ForecastResult, TimestralVLLMForecaster
from src.notifier import AlertConfig, EarlyWarningNotifier
from src.service import ForecastingDaemon, create_app

SLACK_URL = "https://hooks.slack.test/services/T000/B000/XXX"


def _forecast(level: float, horizon: int = 60) -> ForecastResult:
    forecaster = TimestralVLLMForecaster(prediction_length=horizon, mock_mode=True)
    return forecaster.forecast(np.full((1, 2, 64), level, dtype=np.float32), last_timestamp=0.0)


def _slack_notifier(cooldown: int = 900) -> EarlyWarningNotifier:
    return EarlyWarningNotifier(AlertConfig(slack_webhook_url=SLACK_URL, cooldown_seconds=cooldown))


def _ok() -> MagicMock:
    return MagicMock(status_code=200, text="ok")


def test_is_configured() -> None:
    assert EarlyWarningNotifier(AlertConfig()).is_configured() is False
    assert _slack_notifier().is_configured() is True
    # Email needs both a host and a recipient
    assert EarlyWarningNotifier(AlertConfig(smtp_host="smtp.test")).is_configured() is False
    email_cfg = AlertConfig(smtp_host="smtp.test", email_to="ops@example.com")
    assert EarlyWarningNotifier(email_cfg).is_configured() is True


def test_from_env(monkeypatch) -> None:
    monkeypatch.setenv("SLACK_WEBHOOK_URL", SLACK_URL)
    monkeypatch.setenv("SMTP_HOST", "smtp.test")
    monkeypatch.setenv("SMTP_PORT", "2525")
    monkeypatch.setenv("ALERT_EMAIL_TO", "ops@example.com")
    monkeypatch.setenv("SMTP_USE_TLS", "false")
    monkeypatch.setenv("ALERT_COOLDOWN_SECONDS", "60")

    cfg = AlertConfig.from_env()
    assert cfg.slack_webhook_url == SLACK_URL
    assert cfg.smtp_port == 2525
    assert cfg.use_tls is False
    assert cfg.cooldown_seconds == 60
    assert cfg.slack_enabled and cfg.email_enabled


def test_breach_cooldown_and_recovery_lifecycle() -> None:
    notifier = _slack_notifier(cooldown=900)
    breach, normal = _forecast(0.95), _forecast(0.05)
    assert breach.breach_warning and not normal.breach_warning

    with patch("src.notifier.requests.post", return_value=_ok()) as post:
        assert notifier.notify_forecast(normal)["alert_type"] == "none"

        status = notifier.notify_forecast(breach)
        assert status["alert_type"] == "new_breach"
        assert status["dispatched"] is True
        assert status["slack_sent"] is True
        assert status["email_sent"] is None  # email not configured
        assert notifier.in_breach_state is True

        # Within the cooldown: no repeat
        assert notifier.notify_forecast(breach)["alert_type"] == "cooldown_active"

        # Cooldown elapsed: re-alert
        notifier.last_alert_timestamp -= 901
        assert notifier.notify_forecast(breach)["alert_type"] == "recurring_breach"

        recovery = notifier.notify_forecast(normal)
        assert recovery["alert_type"] == "recovery"
        assert recovery["dispatched"] is True
        assert notifier.in_breach_state is False

    assert post.call_count == 3


def test_failed_delivery_does_not_latch_state() -> None:
    notifier = _slack_notifier()
    breach = _forecast(0.95)

    with patch("src.notifier.requests.post", return_value=MagicMock(status_code=500, text="x")):
        status = notifier.notify_forecast(breach)
    assert status["dispatched"] is False
    assert notifier.in_breach_state is False

    # Next cycle retries as a new breach
    with patch("src.notifier.requests.post", return_value=_ok()):
        assert notifier.notify_forecast(breach)["alert_type"] == "new_breach"
    assert notifier.in_breach_state is True


def test_slack_payload_contents() -> None:
    notifier = _slack_notifier()
    breach = _forecast(0.95)

    with patch("src.notifier.requests.post", return_value=_ok()) as post:
        notifier.notify_forecast(breach)
        notifier.notify_forecast(_forecast(0.05))

    alert_payload = post.call_args_list[0].kwargs["json"]
    blocks = alert_payload["attachments"][0]["blocks"]
    text = str(blocks)
    assert "+15s" in text  # imminent breach is not rendered as "+0 minutes"
    peak = float(breach.kv_cache_p90.max()) * 100
    assert f"{peak:.1f}%" in text
    assert blocks[-1]["elements"][0]["style"] == "primary"

    recovery_payload = post.call_args_list[1].kwargs["json"]
    button = recovery_payload["attachments"][0]["blocks"][-1]["elements"][0]
    # Slack rejects style="default"; the recovery button must omit the field
    assert "style" not in button
    assert "next 15m" in str(recovery_payload)


def test_email_delivery_and_html_escaping() -> None:
    cfg = AlertConfig(
        smtp_host="smtp.test",
        smtp_port=2525,
        smtp_user="user",
        smtp_password="secret",
        email_to="ops@example.com",
        email_from="monitor@example.com",
    )
    notifier = EarlyWarningNotifier(cfg)
    breach = _forecast(0.95)
    breach.breach_reason = "<script>alert(1)</script>"

    with patch("src.notifier.smtplib.SMTP") as smtp_cls:
        status = notifier.notify_forecast(breach)

    assert status["email_sent"] is True
    assert status["slack_sent"] is None
    smtp_cls.assert_called_once_with("smtp.test", 2525, timeout=10)
    server = smtp_cls.return_value.__enter__.return_value
    server.starttls.assert_called_once()
    server.login.assert_called_once_with("user", "secret")
    from_addr, to_addrs, raw = server.sendmail.call_args.args
    assert from_addr == "monitor@example.com"
    assert to_addrs == ["ops@example.com"]

    html_body = notifier._render_email_html(breach.breach_reason, "+15s", 0.9, 3.0, False)
    assert "<script>" not in html_body
    assert "&lt;script&gt;" in html_body


def _daemon(notifier, **kwargs) -> ForecastingDaemon:
    return ForecastingDaemon(
        prometheus_url="http://mock-offline-prom:9999",
        vllm_url="http://mock-offline-vllm:9999",
        mock_mode=True,
        synthetic_fallback=True,
        context_steps=64,
        horizon_steps=60,
        notifier=notifier,
        **kwargs,
    )


def _force_cache_level(daemon: ForecastingDaemon, level: float) -> None:
    original = daemon.collector.generate_synthetic_context

    def fake(now: float, current_cache: float = 0.0, current_waiting: float = 0.0):
        return original(now, current_cache=level, current_waiting=0.0)

    daemon.collector.generate_synthetic_context = fake


def test_daemon_dispatches_notifications_and_exports_counters() -> None:
    daemon = _daemon(_slack_notifier(), notify_on_synthetic=True)
    _force_cache_level(daemon, 0.95)

    with patch("src.notifier.requests.post", return_value=_ok()) as post:
        result = daemon.run_single_cycle()

    assert result.breach_warning is True
    post.assert_called_once()
    assert daemon.last_alert_status["alert_type"] == "new_breach"

    metrics = daemon.exporter.export_text()
    assert (
        'timestral_alert_notifications_total{alert_type="new_breach",channel="slack",'
        'outcome="sent"} 1.0' in metrics
    )
    assert "timestral_forecast_synthetic_input 1.0" in metrics
    assert "timestral_forecast_synthetic_model 1.0" in metrics

    health = TestClient(create_app(daemon)).get("/health").json()
    assert health["alerting"]["enabled"] is True
    assert health["alerting"]["slack"] is True
    assert health["alerting"]["in_breach_state"] is True


def test_daemon_suppresses_notifications_for_synthetic_forecasts() -> None:
    notifier = _slack_notifier()
    daemon = _daemon(notifier)
    _force_cache_level(daemon, 0.95)

    with patch("src.notifier.requests.post", return_value=_ok()) as post:
        result = daemon.run_single_cycle()

    assert result.breach_warning is True
    post.assert_not_called()
    assert notifier.in_breach_state is False


def test_unconfigured_notifier_is_disabled() -> None:
    daemon = _daemon(EarlyWarningNotifier(AlertConfig()))
    assert daemon.notifier is None
    daemon.run_single_cycle()  # must not raise
    assert TestClient(create_app(daemon)).get("/health").json()["alerting"]["enabled"] is False


def test_notifier_exception_does_not_break_cycle() -> None:
    notifier = MagicMock()
    notifier.is_configured.return_value = True
    notifier.notify_forecast.side_effect = RuntimeError("boom")
    daemon = _daemon(notifier, notify_on_synthetic=True)

    result = daemon.run_single_cycle()
    assert daemon.latest_result is result


def test_insufficient_history_skips_cycle() -> None:
    daemon = ForecastingDaemon(
        prometheus_url="http://mock-offline-prom:9999",
        vllm_url="http://mock-offline-vllm:9999",
        mock_mode=True,
        context_steps=64,
        horizon_steps=30,
    )
    client = TestClient(create_app(daemon))

    resp = client.post("/api/trigger")
    assert resp.status_code == 503
    assert resp.json()["status"] == "skipped"
    assert daemon.latest_result is None

    metrics = client.get("/metrics").text
    assert 'timestral_forecast_cycles_skipped_total{reason="insufficient_history"} 1.0' in metrics
    health = client.get("/health").json()
    assert health["synthetic_fallback_enabled"] is False
    assert "at least 32" in health["last_cycle_error"]
