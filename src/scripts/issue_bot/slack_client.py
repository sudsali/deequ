import logging
import requests

logger = logging.getLogger("issue_bot")


class SlackClient:
    def __init__(self, cfg):
        self._webhook = cfg.slack_webhook_url
        self._enabled = cfg.enable_slack
        self._dry_run = cfg.dry_run

    def send_escalation(self, number, title, url, category, summary):
        if not self._enabled:
            return
        if self._dry_run:
            logger.info(f"[DRY RUN] Slack escalation for #{number}")
            return
        text = (
            f"*Deequ #{number} - Escalation*\n"
            f"*Issue:* <{url}|{title}>\n"
            f"*Category:* {category}\n\n"
            f"*Bot Analysis:*\n{summary[:500] if summary else 'No AI analysis available'}\n\n"
            f"<{url}|View on GitHub>"
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
