"""
Deequ Bot — two-phase orchestration.

  analyze: read-only phase, produces JSON artifact
  act:     write-only phase, reads artifact and posts to GitHub/Slack
"""

import json
import sys
import os
import datetime
import logging

from .config import Config
from .bedrock_client import BedrockClient
from .github_client import GitHubClient
from .knowledge_base import KnowledgeBase
from .slack_client import SlackClient
from .sanitizer import sanitize
from . import prompts

logger = logging.getLogger("issue_bot")

ARTIFACT_PATH = os.getenv("ARTIFACT_PATH", "/tmp/bot_result.json")
_MAX_BOT_REPLIES = 2


def analyze():
    cfg = Config()
    gh = GitHubClient(cfg)
    bedrock = BedrockClient(cfg)
    kb = KnowledgeBase(cfg)
    kb.load()

    number = cfg.issue_number
    is_pr = cfg.event_type == "pull_request"
    is_followup = cfg.event_type == "issue_comment" and cfg.event_action == "created"

    item = gh.get_pr(number) if is_pr else gh.get_issue(number)
    if not item:
        _write_artifact({"action": "SKIP", "reason": "fetch_failed"})
        return

    author = item.get("user", {}).get("login", "")
    if author.endswith("[bot]"):
        _write_artifact({"action": "SKIP", "reason": "author_is_bot"})
        return

    title = (item.get("title", "") or "")[:200]
    body = (item.get("body", "") or "")[:cfg.max_body_chars]
    html_url = item.get("html_url", "")
    comments_data = gh.get_comments(number)
    comments_text = _format_comments(comments_data)

    if not is_followup and gh.has_bot_commented(number):
        _write_artifact({"action": "SKIP", "reason": "already_commented"})
        return

    if is_followup and comments_data:
        if comments_data[-1].get("user", {}).get("login") == "github-actions[bot]":
            _write_artifact({"action": "SKIP", "reason": "bot_last_comment"})
            return
        if _already_replied_to_latest(comments_data):
            _write_artifact({"action": "SKIP", "reason": "already_replied_to_comment"})
            return
        if _bot_reply_count(comments_data) >= _MAX_BOT_REPLIES:
            _write_artifact({
                "action": "ESCALATE", "labels": [], "response": "",
                "reason": "max_replies_reached", "title": title,
                "html_url": html_url, "number": number, "is_pr": is_pr,
                "prompt_id": "n/a", "model_id": cfg.bedrock_model_id,
            })
            return
        if _user_dissatisfied(comments_data):
            _write_artifact({
                "action": "ESCALATE", "labels": [], "response": "",
                "reason": "user_dissatisfied", "title": title,
                "html_url": html_url, "number": number, "is_pr": is_pr,
                "prompt_id": "n/a", "model_id": cfg.bedrock_model_id,
            })
            return

    issue_text = f"{title} {body}"
    context = kb.build_context(issue_text)

    if is_pr:
        diff = gh.get_pr_diff(number)[:15000]
        files = gh.get_pr_files(number)
        files_summary = "\n".join(
            f"- {f.get('filename', '')} (+{f.get('additions', 0)}/-{f.get('deletions', 0)})"
            for f in files[:30]
        )
        prompt = prompts.PR_CLASSIFY_PROMPT.format(
            context=context, title=title, body=body, files=files_summary, diff=diff,
        )
        prompt_id = prompts.prompt_version(prompts.PR_CLASSIFY_PROMPT)
    elif is_followup:
        prompt = prompts.FOLLOWUP_PROMPT.format(
            context=context, title=title, body=body, comments=comments_text,
        )
        prompt_id = prompts.prompt_version(prompts.FOLLOWUP_PROMPT)
    else:
        prompt = prompts.ISSUE_CLASSIFY_PROMPT.format(
            context=context, title=title, body=body, comments=comments_text,
        )
        prompt_id = prompts.prompt_version(prompts.ISSUE_CLASSIFY_PROMPT)

    raw = bedrock.invoke(prompt)

    if raw is None:
        _write_artifact({
            "action": "ESCALATE", "labels": [], "response": "",
            "reason": "bedrock_unavailable", "title": title,
            "html_url": html_url, "number": number, "is_pr": is_pr,
            "prompt_id": prompt_id, "model_id": cfg.bedrock_model_id,
        })
        return

    parsed = _parse_response(raw, is_pr)

    if parsed.get("needs_search") and cfg.enable_repo_search and not is_pr:
        snippets = _fetch_repo_snippets(gh, parsed.get("search_terms", ""), cfg)
        if snippets:
            enriched_context = kb.build_context(issue_text, snippets)
            prompt2 = prompts.ISSUE_CLASSIFY_PROMPT.format(
                context=enriched_context, title=title, body=body, comments=comments_text,
            )
            raw2 = bedrock.invoke(prompt2)
            if raw2:
                parsed = _parse_response(raw2, is_pr)

    _write_artifact({
        "action": parsed["action"], "labels": parsed.get("labels", []),
        "response": parsed.get("response", ""), "title": title,
        "html_url": html_url, "number": number, "is_pr": is_pr,
        "prompt_id": prompt_id, "model_id": cfg.bedrock_model_id,
    })


def act():
    cfg = Config()
    gh = GitHubClient(cfg)
    slack = SlackClient(cfg)

    result = _read_artifact()
    if not result:
        logger.error("No artifact found")
        return

    action = result.get("action", "SKIP")
    number = result.get("number", cfg.issue_number)
    is_pr = result.get("is_pr", False)
    title = result.get("title", "")
    html_url = result.get("html_url", "")
    labels = result.get("labels", [])
    response = result.get("response", "")
    prompt_id = result.get("prompt_id", "unknown")
    model_id = result.get("model_id", "unknown")

    if action == "SKIP":
        logger.info(f"Skip #{number}: {result.get('reason')}")
        return

    ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    kind = "PR" if is_pr else "Issue"
    footer = (
        f"\n\n---\n*{kind} analyzed with AI assistance "
        f"(model: `{model_id}`, prompt: `{prompt_id}`, ts: `{ts}`). "
        f"If this doesn't help, please reply and a maintainer will assist.*"
    )

    if action == "RESPOND":
        safe = sanitize(response)
        if safe is None:
            action = "ESCALATE"
            response = ""
        else:
            gh.post_comment(number, safe + footer)
            gh.add_labels(number, labels)
            logger.info(f"Responded to #{number}")

    if action == "ESCALATE":
        reason = result.get("reason", "")
        if reason == "user_dissatisfied":
            ack = (
                "I understand my previous response wasn't helpful. "
                "I've notified the maintainer team and they will follow up directly." + footer
            )
        elif reason == "max_replies_reached":
            ack = (
                "I've reached the limit of what I can assist with on this issue. "
                "The maintainer team has been notified and will take over." + footer
            )
        else:
            ack = (
                "Thank you for this submission.\n\n"
                "This has been flagged for review by our maintainer team. "
                "We'll get back to you as soon as possible." + footer
            )
        gh.post_comment(number, ack)
        gh.add_labels(number, labels)
        slack.send_escalation(number, title, html_url, "escalation", response or "No AI analysis available")
        logger.info(f"Escalated #{number}")

    if action == "CLOSE" and cfg.enable_auto_close and not is_pr:
        msg = (
            "This issue does not appear to be related to the Deequ data quality library "
            "and has been automatically closed.\n\n"
            "If this was a mistake, please reopen with additional context." + footer
        )
        safe = sanitize(msg)
        if safe:
            gh.post_comment(number, safe)
            gh.close_issue(number)
            logger.info(f"Closed #{number}")


def _bot_reply_count(comments):
    return sum(1 for c in comments if c.get("user", {}).get("login") == "github-actions[bot]")


def _already_replied_to_latest(comments):
    """True if the bot already posted after the most recent non-bot comment."""
    last_user_idx = -1
    last_bot_idx = -1
    for i, c in enumerate(comments):
        if c.get("user", {}).get("login") == "github-actions[bot]":
            last_bot_idx = i
        else:
            last_user_idx = i
    return last_bot_idx > last_user_idx >= 0


_DISSATISFACTION_SIGNALS = [
    "that's wrong", "thats wrong", "that is wrong",
    "this is wrong", "this is incorrect", "incorrect answer",
    "didn't help", "doesn't help", "not helpful", "unhelpful",
    "wrong answer", "bad answer", "not correct", "that's not right",
    "still broken", "still not working", "doesn't work",
    "please escalate", "need a human", "talk to a human",
    "maintainer", "real person",
]


def _user_dissatisfied(comments):
    bot_has_replied = any(c.get("user", {}).get("login") == "github-actions[bot]" for c in comments)
    if not bot_has_replied:
        return False
    for c in reversed(comments):
        login = c.get("user", {}).get("login", "")
        if login == "github-actions[bot]":
            break
        if not login:
            continue
        body = (c.get("body") or "").lower()
        if any(s in body for s in _DISSATISFACTION_SIGNALS):
            return True
    return False


def _parse_response(raw, is_pr):
    lines = raw.strip().split("\n")
    result = {"action": "ESCALATE", "labels": [], "response": "", "needs_search": False, "search_terms": ""}
    response_lines = []
    header_done = False

    for line in lines:
        upper = line.strip().upper()
        if not header_done:
            if upper.startswith("ACTION:"):
                val = line.split(":", 1)[1].strip().upper()
                if val in ("RESPOND", "ESCALATE", "CLOSE"):
                    result["action"] = val
                continue
            elif upper.startswith("LABELS:"):
                raw_labels = line.split(":", 1)[1].strip()
                result["labels"] = [l.strip() for l in raw_labels.split(",") if l.strip().lower() not in ("none", "")]
                continue
            elif upper.startswith("SEARCH:"):
                result["needs_search"] = "true" in line.lower()
                continue
            elif upper.startswith("SEARCH_TERMS:"):
                result["search_terms"] = line.split(":", 1)[1].strip()
                continue
            else:
                header_done = True
        response_lines.append(line)

    result["response"] = "\n".join(response_lines).strip()
    if is_pr and result["action"] == "CLOSE":
        result["action"] = "ESCALATE"
    return result


def _format_comments(comments):
    if not comments:
        return "(none)"
    return "\n".join(
        f"{c.get('user', {}).get('login', '?')}: {(c.get('body', '') or '')[:500]}"
        for c in comments[-5:]
    )


def _fetch_repo_snippets(gh, search_terms, cfg):
    if not search_terms:
        return ""
    items = gh.search_code(search_terms, repo_override=cfg.upstream_repo)
    snippets = []
    for item in items[:cfg.max_github_search_results]:
        path = item.get("path", "")
        content = gh.get_file_content(path, repo=cfg.upstream_repo)
        if content:
            if len(content) > 5000:
                content = content[:5000] + "\n... (truncated)"
            snippets.append(f"### {path}\n```\n{content}\n```")
    return "\n\n".join(snippets)


def _write_artifact(data):
    os.makedirs(os.path.dirname(ARTIFACT_PATH) or "/tmp", exist_ok=True)
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(data, f)
    logger.info(f"Artifact: action={data.get('action')}")


def _read_artifact():
    try:
        with open(ARTIFACT_PATH) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Artifact read failed: {e}")
        return None


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("analyze", "act"):
        print("Usage: python -m issue_bot.main <analyze|act>")
        sys.exit(1)
    {"analyze": analyze, "act": act}[sys.argv[1]]()


if __name__ == "__main__":
    main()
