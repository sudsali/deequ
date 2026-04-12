"""
Deequ Bot — two-phase orchestration.

  analyze: read-only phase, produces JSON artifact
  act:     write-only phase, reads artifact and posts to GitHub/Slack
"""

import json
import sys
import os
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
    codebase_map = gh.get_codebase_map() if not is_followup else ""

    if is_pr:
        diff = gh.get_pr_diff(number)[:15000]
        files = gh.get_pr_files(number)
        files_summary = "\n".join(
            f"- {f.get('filename', '')} (+{f.get('additions', 0)}/-{f.get('deletions', 0)})"
            for f in files[:30]
        )
        tmpl = prompts.get_pr_prompt()
        if not tmpl:
            _write_artifact({"action": "ESCALATE", "labels": [], "response": "",
                "reason": "prompt_load_failed", "title": title, "html_url": html_url,
                "number": number, "is_pr": is_pr, "prompt_id": "n/a", "model_id": cfg.bedrock_model_id})
            return
        prompt = tmpl.format(context=context, title=title, body=body,
                             files=files_summary, diff=diff, codebase_map=codebase_map)
        prompt_id = prompts.prompt_version(tmpl)
    elif is_followup:
        tmpl = prompts.get_followup_prompt()
        if not tmpl:
            _write_artifact({"action": "ESCALATE", "labels": [], "response": "",
                "reason": "prompt_load_failed", "title": title, "html_url": html_url,
                "number": number, "is_pr": is_pr, "prompt_id": "n/a", "model_id": cfg.bedrock_model_id})
            return
        prompt = tmpl.format(context=context, title=title, body=body, comments=comments_text)
        prompt_id = prompts.prompt_version(tmpl)
    else:
        tmpl = prompts.get_issue_prompt()
        if not tmpl:
            _write_artifact({"action": "ESCALATE", "labels": [], "response": "",
                "reason": "prompt_load_failed", "title": title, "html_url": html_url,
                "number": number, "is_pr": is_pr, "prompt_id": "n/a", "model_id": cfg.bedrock_model_id})
            return
        prompt = tmpl.format(context=context, title=title, body=body,
                             comments=comments_text, codebase_map=codebase_map)
        prompt_id = prompts.prompt_version(tmpl)

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

    if parsed.get("read_files") and cfg.enable_repo_search:
        snippets = _read_requested_files(gh, parsed["read_files"], cfg)
        if snippets:
            if is_pr:
                respond_tmpl = prompts.get_pr_respond_prompt()
                if respond_tmpl:
                    prompt2 = respond_tmpl.format(
                        context=context, source_code=snippets, title=title,
                        body=body, files=files_summary, diff=diff,
                    )
            else:
                respond_tmpl = prompts.get_issue_respond_prompt()
                if respond_tmpl:
                    prompt2 = respond_tmpl.format(
                        context=context, source_code=snippets, title=title,
                        body=body, comments=comments_text,
                    )
            if respond_tmpl and 'prompt2' in locals():
                raw2 = bedrock.invoke(prompt2)
                if raw2:
                    parsed2 = _parse_response(raw2, is_pr)
                    parsed2["labels"] = parsed2.get("labels") or parsed.get("labels", [])
                    parsed = parsed2

    _write_artifact({
        "action": parsed["action"], "labels": parsed.get("labels", []),
        "response": parsed.get("response", ""),
        "inline_comments": parsed.get("inline_comments", []),
        "title": title, "html_url": html_url, "number": number,
        "is_pr": is_pr, "prompt_id": prompt_id, "model_id": cfg.bedrock_model_id,
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

    footer = "\n\n---\n*This response was generated using AI and may not be fully accurate. If this doesn't help, please reply and a maintainer will follow up.*"

    if action == "RESPOND":
        safe = sanitize(response)
        if safe is None:
            action = "ESCALATE"
            response = ""
        else:
            inline_comments = result.get("inline_comments", [])
            if is_pr and inline_comments:
                gh.post_pr_review(number, safe + footer, inline_comments)
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
            if response:
                ack = (
                    response + "\n\n"
                    "This has also been flagged for our maintainer team to review." + footer
                )
            else:
                ack = (
                    "Thank you for reporting this.\n\n"
                    "This has been flagged for review by our maintainer team. "
                    "We'll get back to you as soon as possible." + footer
                )
        gh.post_comment(number, ack)
        gh.add_labels(number, labels)
        slack.send_escalation(number, title, html_url, labels)
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


_HEADER_PREFIXES = ("ACTION:", "LABELS:", "READ_FILES:", "SEARCH:", "SEARCH_TERMS:")


def _parse_response(raw, is_pr):
    lines = raw.strip().split("\n")
    result = {"action": "ESCALATE", "labels": [], "response": "", "read_files": [], "inline_comments": []}
    response_lines = []

    for line in lines:
        upper = line.strip().upper()
        if upper.startswith("ACTION:"):
            val = line.split(":", 1)[1].strip().upper()
            if val in ("RESPOND", "ESCALATE", "CLOSE"):
                result["action"] = val
            continue
        elif upper.startswith("LABELS:"):
            raw_labels = line.split(":", 1)[1].strip()
            result["labels"] = [l.strip() for l in raw_labels.split(",") if l.strip().lower() not in ("none", "")]
            continue
        elif upper.startswith("READ_FILES:"):
            raw_files = line.split(":", 1)[1].strip()
            result["read_files"] = [f.strip() for f in raw_files.split(",") if f.strip().lower() not in ("none", "")]
            continue
        elif upper.startswith(("SEARCH:", "SEARCH_TERMS:")):
            continue
        response_lines.append(line)

    full_text = "\n".join(response_lines).strip()

    if is_pr and "INLINE:" in full_text and "FILE:" in full_text:
        result["response"], result["inline_comments"] = _parse_pr_review(full_text)
    else:
        result["response"] = _clean_response(full_text)

    if is_pr and result["action"] == "CLOSE":
        result["action"] = "ESCALATE"
    return result


def _parse_pr_review(text):
    """Split PR review into summary and inline comments."""
    summary_part = ""
    inline_comments = []

    parts = text.split("INLINE:")
    summary_part = parts[0].replace("SUMMARY:", "").strip()

    if len(parts) > 1:
        inline_text = parts[1].strip()
        if inline_text.lower() == "none":
            return _clean_response(summary_part), []

        current = {}
        for line in inline_text.split("\n"):
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("FILE:"):
                if current.get("file") and current.get("comment"):
                    inline_comments.append(current)
                current = {"file": stripped.split(":", 1)[1].strip()}
            elif upper.startswith("LINE:"):
                try:
                    current["line"] = int(stripped.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif upper.startswith("COMMENT:"):
                current["comment"] = stripped.split(":", 1)[1].strip()
            elif current.get("comment"):
                current["comment"] += "\n" + stripped

        if current.get("file") and current.get("comment"):
            inline_comments.append(current)

    return _clean_response(summary_part), inline_comments


def _clean_response(text):
    """Remove any leaked headers or internal thinking from the response."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        upper = line.strip().upper()
        if upper.startswith(_HEADER_PREFIXES):
            continue
        cleaned.append(line)
    result = "\n".join(cleaned).strip()
    # Remove leading preamble like "Let me request..." or "I'll analyze..."
    while result and result.split("\n")[0].strip().lower().startswith((
        "let me ", "i'll ", "i will ", "i need to ", "first,", "sure,",
        "since i don't", "since i do not",
    )):
        result = "\n".join(result.split("\n")[1:]).strip()
    return result


def _format_comments(comments):
    if not comments:
        return "(none)"
    return "\n".join(
        f"{c.get('user', {}).get('login', '?')}: {(c.get('body', '') or '')[:500]}"
        for c in comments[-5:]
    )


def _read_requested_files(gh, file_paths, cfg):
    snippets = []
    for path in file_paths[:cfg.max_github_search_results]:
        content = gh.read_local_file(path)
        if not content:
            content = gh.get_file_content(path, repo=cfg.upstream_repo)
        if content:
            snippets.append(f"### {path}\n```scala\n{content}\n```")
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
