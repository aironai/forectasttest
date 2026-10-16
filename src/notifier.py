"""Early warning notification service supporting Slack webhooks and SMTP email."""

from __future__ import annotations

import logging
import os
import smtplib
import time
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import requests

from src.forecaster import ForecastResult

logger = logging.getLogger(__name__)


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


class EarlyWarningNotifier:
    """Manages threshold-based predictive warning alerts with deduplication and cooldown."""

    def __init__(self, config: AlertConfig | None = None) -> None:
        self.config = config or AlertConfig.from_env()
        self.last_alert_timestamp: float = 0.0
        self.in_breach_state: bool = False

    def is_configured(self) -> bool:
        """Returns True if at least one notification channel is active."""
        has_slack = bool(self.config.slack_webhook_url)
        has_email = bool(self.config.smtp_host and self.config.email_to)
        return has_slack or has_email

    def should_trigger_alert(self, breach_active: bool) -> tuple[bool, str]:
        """Determines if a notification should be dispatched based on state and cooldown.

        Returns:
            (should_send, alert_type) where alert_type is 'new_breach', 'recurring_breach',
            'recovery', or 'none'.
        """
        now = time.time()

        if breach_active:
            if not self.in_breach_state:
                # Transition: Safe -> Breach
                return True, "new_breach"
            elif (now - self.last_alert_timestamp) >= self.config.cooldown_seconds:
                # Re-alert after cooldown expires
                return True, "recurring_breach"
            return False, "cooldown_active"
        else:
            if self.in_breach_state:
                # Transition: Breach -> Safe (Recovery)
                return True, "recovery"
            return False, "none"

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
        if not self.config.slack_webhook_url:
            return False

        color = "#36a64f" if is_recovery else "#e01e5a"
        lead_str = f"+{lead_time_seconds // 60} minutes" if lead_time_seconds else "Imminent"

        fields = []
        if not is_recovery:
            if kv_p90 is not None:
                fields.append(
                    {"type": "mrkdwn", "text": f"*Projected KV-Cache (p90):*\n`{kv_p90*100:.1f}%`"}
                )
            if queue_p90 is not None:
                fields.append(
                    {"type": "mrkdwn", "text": f"*Projected Queue (p90):*\n`{queue_p90:.1f} reqs`"}
                )
            fields.append({"type": "mrkdwn", "text": f"*Estimated Lead Time:*\n`{lead_str}`"})
            fields.append({"type": "mrkdwn", "text": "*Inference Device:*\n`Host CPU (0 MB VRAM)`"})

        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": (
                        "✅ vLLM Metrics Stabilized"
                        if is_recovery
                        else "⚠️ Early Warning: vLLM Capacity Breach Projected"
                    ),
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*{headline}*\n\n{reason}\n\n"
                        f"👉 *Action Required*: Check the Grafana predictive fan to evaluate "
                        f"workload steering or cluster scaling before requests are throttled."
                    ),
                },
            },
        ]

        if fields:
            blocks.append({"type": "section", "fields": fields})

        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "📊 View Grafana Dashboard"},
                        "url": self.config.grafana_url,
                        "style": "primary" if not is_recovery else "default",
                    }
                ],
            }
        )

        payload = {"attachments": [{"color": color, "blocks": blocks}]}

        try:
            resp = requests.post(self.config.slack_webhook_url, json=payload, timeout=8)
            if resp.status_code == 200:
                logger.info("Slack alert successfully dispatched to webhook")
                return True
            else:
                logger.warning("Slack webhook returned status %d: %s", resp.status_code, resp.text)
        except Exception as e:
            logger.error("Error sending Slack alert: %s", e)
        return False

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
        if not (self.config.smtp_host and self.config.email_to):
            return False

        lead_str = f"+{lead_time_seconds // 60}m" if lead_time_seconds else "Imminent"
        status_color = "#28a745" if is_recovery else "#dc3545"

        html_content = f"""
        <html>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 650px; margin: 0 auto; padding: 20px;">
            <div style="background-color: {status_color}; color: #ffffff; padding: 15px 20px; border-radius: 6px 6px 0 0;">
                <h2 style="margin: 0; font-size: 20px;">
                    {'✅ vLLM Metrics Stabilized (Recovery)' if is_recovery else '⚠️ Early Warning: vLLM Capacity Breach Projected'}
                </h2>
            </div>
            <div style="border: 1px solid #e1e4e8; border-top: none; padding: 25px; border-radius: 0 0 6px 6px; background-color: #fafbfc;">
                <p style="font-size: 16px; margin-top: 0;"><strong>System Alert:</strong> {reason}</p>
                <div style="background-color: #ffffff; border: 1px solid #e1e4e8; border-radius: 6px; padding: 15px; margin: 20px 0;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 8px 0; color: #586069;"><strong>Estimated Lead Time:</strong></td>
                            <td style="padding: 8px 0; font-family: monospace; font-size: 14px;">{lead_str}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #586069;"><strong>Projected KV-Cache (p90):</strong></td>
                            <td style="padding: 8px 0; font-family: monospace; font-size: 14px;">{f'{kv_p90*100:.1f}%' if kv_p90 is not None else 'N/A'}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #586069;"><strong>Projected Queue Depth (p90):</strong></td>
                            <td style="padding: 8px 0; font-family: monospace; font-size: 14px;">{f'{queue_p90:.1f} reqs' if queue_p90 is not None else 'N/A'}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #586069;"><strong>Inference Overhead:</strong></td>
                            <td style="padding: 8px 0; font-family: monospace; font-size: 14px;">0 MB GPU VRAM (Host CPU Pinned)</td>
                        </tr>
                    </table>
                </div>
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{self.config.grafana_url}" style="background-color: #0366d6; color: #ffffff; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: 600; display: inline-block;">
                        Open Grafana Predictive Dashboard
                    </a>
                </div>
                <p style="font-size: 12px; color: #6a737d; margin-bottom: 0;">
                    This automated alert was generated by Mistral Timestral time-series forecasting on the 8x B300 HGX system.
                </p>
            </div>
        </body>
        </html>
        """

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
        """Evaluates forecast result and dispatches notifications if state changes."""
        should_send, alert_type = self.should_trigger_alert(result.breach_warning)
        status = {
            "evaluated": True,
            "breach_warning": result.breach_warning,
            "alert_type": alert_type,
            "dispatched": False,
            "slack_sent": False,
            "email_sent": False,
        }

        if not should_send:
            return status

        # Extract 15m summary quantiles if available
        s15 = next((s for s in result.summaries if s.lead_time_label == "15m"), None)
        kv_p90 = s15.kv_cache_p90 if s15 else float(result.kv_cache_p90[-1])
        queue_p90 = s15.requests_waiting_p90 if s15 else float(result.requests_waiting_p90[-1])

        if alert_type in ("new_breach", "recurring_breach"):
            headline = "⚠️ Projected vLLM KV-Cache / Queue Depth Capacity Exhaustion"
            reason = result.breach_reason or "Projected load exceeds configured safety threshold."
            subject = f"⚠️ [Alert] vLLM Capacity Breach Projected ({reason})"

            slack_ok = self.send_slack(
                headline=headline,
                reason=reason,
                lead_time_seconds=result.breach_lead_time_seconds,
                kv_p90=kv_p90,
                queue_p90=queue_p90,
                is_recovery=False,
            )
            email_ok = self.send_email(
                subject=subject,
                reason=reason,
                lead_time_seconds=result.breach_lead_time_seconds,
                kv_p90=kv_p90,
                queue_p90=queue_p90,
                is_recovery=False,
            )

            self.in_breach_state = True
            self.last_alert_timestamp = time.time()
            status["dispatched"] = slack_ok or email_ok
            status["slack_sent"] = slack_ok
            status["email_sent"] = email_ok

        elif alert_type == "recovery":
            headline = "✅ vLLM Metrics Stabilized Within Normal Operating Bounds"
            reason = "Timestral forecast projects all metrics safely below threshold for next 30 minutes."
            subject = "✅ [Resolved] vLLM KV-Cache & Queue Normal"

            slack_ok = self.send_slack(
                headline=headline,
                reason=reason,
                lead_time_seconds=None,
                kv_p90=kv_p90,
                queue_p90=queue_p90,
                is_recovery=True,
            )
            email_ok = self.send_email(
                subject=subject,
                reason=reason,
                lead_time_seconds=None,
                kv_p90=kv_p90,
                queue_p90=queue_p90,
                is_recovery=True,
            )

            self.in_breach_state = False
            status["dispatched"] = slack_ok or email_ok
            status["slack_sent"] = slack_ok
            status["email_sent"] = email_ok

        return status
