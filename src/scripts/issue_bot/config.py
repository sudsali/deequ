import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue_bot")


class Config:
    def __init__(self):
        self.github_token = _require("GITHUB_TOKEN")
        self.event_type = _require("EVENT_TYPE")
        self.event_action = os.getenv("EVENT_ACTION", "")
        self.issue_number = _require("ISSUE_NUMBER")
        self.repo = _require("GITHUB_REPOSITORY")
        self.actor = os.getenv("GITHUB_ACTOR", "")

        self.bedrock_model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-opus-4-6-v1")
        self.bedrock_api_version = os.getenv("BEDROCK_API_VERSION", "bedrock-2023-05-20")

        self.kb_s3_bucket = os.getenv("KB_S3_BUCKET", "")
        self.kb_s3_key = os.getenv("KB_S3_KEY", "")

        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")

        self.dry_run = os.getenv("DRY_RUN", "false").lower() == "true"
        self.enable_auto_close = os.getenv("ENABLE_AUTO_CLOSE", "true").lower() == "true"
        self.enable_slack = bool(self.slack_webhook_url)
        self.enable_repo_search = os.getenv("ENABLE_REPO_SEARCH", "true").lower() == "true"

        self.upstream_repo = os.getenv("UPSTREAM_REPO", "awslabs/deequ")

        self.bedrock_timeout = 30
        self.max_context_chars = 32000
        self.max_body_chars = 12000
        self.max_github_search_results = 5
        self.github_api_timeout = 10


def _require(name):
    val = os.getenv(name)
    if not val:
        logger.error(f"Missing required env var: {name}")
        sys.exit(1)
    return val
