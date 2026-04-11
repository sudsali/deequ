import logging
import requests

logger = logging.getLogger("issue_bot")


class SlackClient:
    def __init__(self, cfg):
        self._webhook = cfg.slack_webhook_url
        self._enabled = cfg.enable_slack
        self._dry_run = cfg.dry_run

    def send_escalation(self, number, title, url, labels):
        if not self._enabled:
            return
        if self._dry_run:
            logger.info(f"[DRY RUN] Slack escalation for #{number}")
            return
        label_text = ", ".join(labels) if labels else "none"
        text = (
            f"*Deequ #{number} needs attention*\n"
            f"<{url}|{title}>\n"
            f"Labels: {label_text}\n"
            f"Bot posted analysis on the issue."
        )
        self._send({"text": text})

    def send_error(self, message):
        if not self._enabled:
            return
        self._send({"text": f"Deequ Bot Error: {message}"})

    def _send(self, payload):
        try:
            resp = requests.post(self._webhook, json=payload, timeout=10)
            if resp.status_code != 200:
                logger.error(f"Slack: {resp.status_code}")
        except Exception as e:
            logger.error(f"Slack failed: {e}")
