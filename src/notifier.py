"""Early warning notification service supporting Slack webhooks and SMTP email."""

from __future__ import annotations

import html
import logging
import os
import smtplib
import time
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import requests

from src.forecaster import ForecastResult, format_lead_time

logger = logging.getLogger(__name__)

ALERT_NEW_BREACH = "new_breach"
ALERT_RECURRING_BREACH = "recurring_breach"
ALERT_RECOVERY = "recovery"
ALERT_COOLDOWN = "cooldown_active"
ALERT_NONE = "none"

BREACH_TITLE = "⚠️ Early Warning: vLLM Capacity Breach Projected"
RECOVERY_TITLE = "✅ vLLM Metrics Stabilized"

FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
LABEL_CELL = "padding: 8px 0; color: #586069;"
VALUE_CELL = "padding: 8px 0; font-family: monospace; font-size: 14px;"


@dataclass
class AlertConfig:
    """Configuration for Slack and Email alerting."""

    slack_webhook_url: str | None = None
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    email_to: str | None = None
    email_from: str | None = None
    use_tls: bool = True
    cooldown_seconds: int = 900  # 15 minutes cooldown to avoid notification storms
    grafana_url: str = "http://localhost:3000/d/vllm-timestral-forecast/5846bdd"

    @classmethod
    def from_env(cls) -> AlertConfig:
        """Loads alert configuration from environment variables."""
        return cls(
            slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
            smtp_host=os.getenv("SMTP_HOST"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_user=os.getenv("SMTP_USER"),
            smtp_password=os.getenv("SMTP_PASSWORD"),
            email_to=os.getenv("ALERT_EMAIL_TO"),
            email_from=os.getenv("ALERT_EMAIL_FROM", "vllm-monitor@hgx.local"),
            use_tls=os.getenv("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes"),
            cooldown_seconds=int(os.getenv("ALERT_COOLDOWN_SECONDS", "900")),
            grafana_url=os.getenv(
                "GRAFANA_URL", "http://localhost:3000/d/vllm-timestral-forecast/5846bdd"
            ),
        )

    @property
    def slack_enabled(self) -> bool:
        return bool(self.slack_webhook_url)

    @property
    def email_enabled(self) -> bool:
        return bool(self.smtp_host and self.email_to)


class EarlyWarningNotifier:
    """Manages threshold-based predictive warning alerts with deduplication and cooldown.

    State machine:
        safe   --breach-->  breach   sends "new_breach"
        breach --breach-->  breach   sends "recurring_breach" once the cooldown has elapsed
        breach --clear--->  safe     sends "recovery"

    State only advances when at least one channel delivered the message, so a failed
    delivery is retried on the next forecast cycle.
    """

    def __init__(self, config: AlertConfig | None = None) -> None:
        self.config = config or AlertConfig.from_env()
        self.last_alert_timestamp: float = 0.0
        self.in_breach_state: bool = False

    def is_configured(self) -> bool:
        """Returns True if at least one notification channel is active."""
        return self.config.slack_enabled or self.config.email_enabled

    def should_trigger_alert(self, breach_active: bool) -> tuple[bool, str]:
        """Determines if a notification should be dispatched based on state and cooldown.

        Returns:
            (should_send, alert_type) where alert_type is 'new_breach', 'recurring_breach',
            'recovery', 'cooldown_active', or 'none'.
        """
        now = time.time()

        if breach_active:
            if not self.in_breach_state:
                return True, ALERT_NEW_BREACH
            if (now - self.last_alert_timestamp) >= self.config.cooldown_seconds:
                return True, ALERT_RECURRING_BREACH
            return False, ALERT_COOLDOWN
        if self.in_breach_state:
            return True, ALERT_RECOVERY
        return False, ALERT_NONE

    @staticmethod
    def _lead_str(lead_time_seconds: int | None) -> str:
        return f"+{format_lead_time(lead_time_seconds)}" if lead_time_seconds else "Imminent"

    def send_slack(
        self,
        headline: str,
        reason: str,
        lead_time_seconds: int | None,
        kv_p90: float | None,
        queue_p90: float | None,
        is_recovery: bool = False,
    ) -> bool:
        """Sends a rich formatted Slack message using Block Kit."""
        if not self.config.slack_enabled:
            return False

        color = "#36a64f" if is_recovery else "#e01e5a"
        lead_str = self._lead_str(lead_time_seconds)

        fields = []
        if not is_recovery:
            if kv_p90 is not None:
                fields.append(
                    {"type": "mrkdwn", "text": f"*Peak KV-Cache p90:*\n`{kv_p90 * 100:.1f}%`"}
                )
            if queue_p90 is not None:
                fields.append(
                    {"type": "mrkdwn", "text": f"*Peak Queue p90:*\n`{queue_p90:.1f} reqs`"}
                )
            fields.append({"type": "mrkdwn", "text": f"*Estimated Lead Time:*\n`{lead_str}`"})
            fields.append({"type": "mrkdwn", "text": "*Inference Device:*\n`Host CPU (0 MB VRAM)`"})

        action_text = (
            "Check the Grafana predictive fan to evaluate workload steering or cluster scaling "
            "before requests are throttled."
        )
        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": RECOVERY_TITLE if is_recovery else BREACH_TITLE,
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{headline}*\n\n{reason}\n\n👉 *Action Required*: {action_text}",
                },
            },
        ]

        if fields:
            blocks.append({"type": "section", "fields": fields})

        button: dict[str, Any] = {
            "type": "button",
            "text": {"type": "plain_text", "text": "📊 View Grafana Dashboard"},
            "url": self.config.grafana_url,
        }
        # Slack only accepts "primary" or "danger"; omit the field for the default style
        if not is_recovery:
            button["style"] = "primary"
        blocks.append({"type": "actions", "elements": [button]})

        payload = {"attachments": [{"color": color, "blocks": blocks}]}

        try:
            resp = requests.post(self.config.slack_webhook_url, json=payload, timeout=8)
            if resp.status_code == 200:
                logger.info("Slack alert successfully dispatched to webhook")
                return True
            logger.warning("Slack webhook returned status %d: %s", resp.status_code, resp.text)
        except Exception as e:
            logger.error("Error sending Slack alert: %s", e)
        return False

    def _render_email_html(
        self,
        reason: str,
        lead_str: str,
        kv_p90: float | None,
        queue_p90: float | None,
        is_recovery: bool,
    ) -> str:
        """Renders the HTML email body. All dynamic values are HTML-escaped."""
        status_color = "#28a745" if is_recovery else "#dc3545"
        title = f"{RECOVERY_TITLE} (Recovery)" if is_recovery else BREACH_TITLE
        kv_str = f"{kv_p90 * 100:.1f}%" if kv_p90 is not None else "N/A"
        queue_str = f"{queue_p90:.1f} reqs" if queue_p90 is not None else "N/A"
        rows = [
            ("Estimated Lead Time", lead_str),
            ("Peak KV-Cache p90", kv_str),
            ("Peak Queue Depth p90", queue_str),
            ("Inference Overhead", "0 MB GPU VRAM (Host CPU Pinned)"),
        ]
        table_rows = "".join(
            f'<tr><td style="{LABEL_CELL}"><strong>{html.escape(label)}:</strong></td>'
            f'<td style="{VALUE_CELL}">{html.escape(value)}</td></tr>'
            for label, value in rows
        )
        grafana_url = html.escape(self.config.grafana_url, quote=True)

        return f"""
        <html>
        <body style="font-family: {FONT_STACK}; line-height: 1.6; color: #333;
                     max-width: 650px; margin: 0 auto; padding: 20px;">
            <div style="background-color: {status_color}; color: #ffffff; padding: 15px 20px;
                        border-radius: 6px 6px 0 0;">
                <h2 style="margin: 0; font-size: 20px;">{html.escape(title)}</h2>
            </div>
            <div style="border: 1px solid #e1e4e8; border-top: none; padding: 25px;
                        border-radius: 0 0 6px 6px; background-color: #fafbfc;">
                <p style="font-size: 16px; margin-top: 0;">
                    <strong>System Alert:</strong> {html.escape(reason)}
                </p>
                <div style="background-color: #ffffff; border: 1px solid #e1e4e8;
                            border-radius: 6px; padding: 15px; margin: 20px 0;">
                    <table style="width: 100%; border-collapse: collapse;">{table_rows}</table>
                </div>
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{grafana_url}" style="background-color: #0366d6; color: #ffffff;
                       padding: 12px 24px; text-decoration: none; border-radius: 6px;
                       font-weight: 600; display: inline-block;">
                        Open Grafana Predictive Dashboard
                    </a>
                </div>
                <p style="font-size: 12px; color: #6a737d; margin-bottom: 0;">
                    This automated alert was generated by Mistral Timestral time-series
                    forecasting on the 8x B300 HGX system.
                </p>
            </div>
        </body>
        </html>
        """

    def send_email(
        self,
        subject: str,
        reason: str,
        lead_time_seconds: int | None,
        kv_p90: float | None,
        queue_p90: float | None,
        is_recovery: bool = False,
    ) -> bool:
        """Sends a notification email via SMTP."""
        if not self.config.email_enabled:
            return False

        lead_str = self._lead_str(lead_time_seconds)
        html_content = self._render_email_html(reason, lead_str, kv_p90, queue_p90, is_recovery)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.config.email_from or "vllm-monitor@hgx.local"
        msg["To"] = self.config.email_to

        plain_text = (
            f"{subject}\n\n"
            f"Details: {reason}\n"
            f"Lead Time: {lead_str}\n"
            f"Grafana: {self.config.grafana_url}\n"
        )

        msg.attach(MIMEText(plain_text, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        try:
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port, timeout=10) as server:
                if self.config.use_tls:
                    server.starttls()
                if self.config.smtp_user and self.config.smtp_password:
                    server.login(self.config.smtp_user, self.config.smtp_password)
                server.sendmail(msg["From"], [self.config.email_to], msg.as_string())
            logger.info("Email alert sent to %s", self.config.email_to)
            return True
        except Exception as e:
            logger.error("Error sending email alert: %s", e)
        return False

    def notify_forecast(self, result: ForecastResult) -> dict[str, Any]:
        """Evaluates a forecast result and dispatches notifications on state changes.

        Returns a status dict with ``alert_type``, ``dispatched`` and per-channel results
        (``slack_sent``/``email_sent`` are None when that channel is not configured).
        """
        should_send, alert_type = self.should_trigger_alert(result.breach_warning)
        status: dict[str, Any] = {
            "evaluated": True,
            "breach_warning": result.breach_warning,
            "alert_type": alert_type,
            "dispatched": False,
            "slack_sent": None,
            "email_sent": None,
        }

        if not should_send:
            return status

        # Report the worst projected p90 across the whole horizon, which is what triggered
        # (or cleared) the breach, rather than a fixed lead time that may not contain it
        kv_p90 = float(result.kv_cache_p90.max())
        queue_p90 = float(result.requests_waiting_p90.max())
        horizon_str = format_lead_time(result.prediction_length * result.step_seconds)
        is_recovery = alert_type == ALERT_RECOVERY

        if is_recovery:
            headline = "✅ vLLM Metrics Stabilized Within Normal Operating Bounds"
            reason = (
                "Timestral forecast projects all metrics safely below threshold for the next "
                f"{horizon_str}."
            )
            subject = "✅ [Resolved] vLLM KV-Cache & Queue Normal"
            lead_time = None
        else:
            headline = "⚠️ Projected vLLM KV-Cache / Queue Depth Capacity Exhaustion"
            reason = result.breach_reason or "Projected load exceeds configured safety threshold."
            subject = f"⚠️ [Alert] vLLM Capacity Breach Projected ({reason})"
            lead_time = result.breach_lead_time_seconds

        common = {
            "reason": reason,
            "lead_time_seconds": lead_time,
            "kv_p90": kv_p90,
            "queue_p90": queue_p90,
            "is_recovery": is_recovery,
        }
        if self.config.slack_enabled:
            status["slack_sent"] = self.send_slack(headline=headline, **common)
        if self.config.email_enabled:
            status["email_sent"] = self.send_email(subject=subject, **common)

        dispatched = bool(status["slack_sent"]) or bool(status["email_sent"])
        status["dispatched"] = dispatched

        if dispatched:
            self.in_breach_state = not is_recovery
            if not is_recovery:
                self.last_alert_timestamp = time.time()
        else:
            logger.warning(
                "No notification channel delivered the '%s' alert; retrying next cycle",
                alert_type,
            )

        return status
