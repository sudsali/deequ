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
        self._send({
            "text": f"Deequ #{number} needs attention",
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": f"Deequ #{number} Escalation"}},
                {"type": "section", "fields": [
                    {"type": "mrkdwn", "text": f"*Issue:* <{url}|{title}>"},
                    {"type": "mrkdwn", "text": f"*Category:* {category}"},
                ]},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*Summary:*\n{summary[:500]}"}},
                {"type": "actions", "elements": [
                    {"type": "button", "text": {"type": "plain_text", "text": "View on GitHub"}, "url": url, "style": "primary"}
                ]},
            ],
        })

    def send_error(self, message):
        if not self._enabled:
            return
        self._send({"text": f":warning: Deequ Bot Error: {message}"})

    def _send(self, payload):
        try:
            resp = requests.post(self._webhook, json=payload, timeout=10)
            if resp.status_code != 200:
                logger.error(f"Slack: {resp.status_code}")
        except Exception as e:
            logger.error(f"Slack failed: {e}")
