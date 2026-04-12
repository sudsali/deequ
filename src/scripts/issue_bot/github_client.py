import logging
import requests

logger = logging.getLogger("issue_bot")


class GitHubClient:
    def __init__(self, cfg):
        self._token = cfg.github_token
        self._repo = cfg.repo
        self._timeout = cfg.github_api_timeout
        self._dry_run = cfg.dry_run
        self._headers = {
            "Authorization": f"token {self._token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def get_issue(self, number):
        return self._get(f"/repos/{self._repo}/issues/{number}")

    def get_comments(self, number, limit=10):
        comments = []
        page = 1
        while True:
            batch = self._get(f"/repos/{self._repo}/issues/{number}/comments?per_page=100&page={page}")
            if not batch:
                break
            comments.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return comments[-limit:]

    def get_pr(self, number):
        return self._get(f"/repos/{self._repo}/pulls/{number}")

    def get_pr_diff(self, number):
        headers = {**self._headers, "Accept": "application/vnd.github.v3.diff"}
        try:
            resp = requests.get(
                f"https://api.github.com/repos/{self._repo}/pulls/{number}",
                headers=headers, timeout=self._timeout,
            )
            return resp.text if resp.status_code == 200 else ""
        except Exception as e:
            logger.error(f"PR diff fetch failed: {e}")
            return ""

    def get_pr_files(self, number):
        return self._get(f"/repos/{self._repo}/pulls/{number}/files") or []

    def has_bot_commented(self, number):
        for c in self.get_comments(number, limit=50):
            if c.get("user", {}).get("login") == "github-actions[bot]":
                return True
        return False

    def get_codebase_map(self, src_dir="src/main/scala"):
        """List all Scala source files (excluding examples) as a simple path listing."""
        import subprocess
        try:
            proc = subprocess.run(
                ["find", src_dir, "-name", "*.scala", "-not", "-path", "*/examples/*"],
                capture_output=True, text=True, timeout=10,
            )
            paths = sorted(p for p in proc.stdout.strip().split("\n") if p)
            return "\n".join(paths)
        except Exception as e:
            logger.error(f"Codebase map failed: {e}")
            return ""

    def read_local_file(self, path, max_chars=15000):
        if not path.startswith("src/"):
            logger.error(f"Blocked read outside src/: {path}")
            return ""
        try:
            with open(path, "r", errors="replace") as f:
                content = f.read()
            if len(content) > max_chars:
                return content[:max_chars] + "\n... (truncated)"
            return content
        except Exception:
            return ""

    def get_file_content(self, path, repo=None, ref=None):
        target = repo or self._repo
        url = f"https://api.github.com/repos/{target}/contents/{path}"
        if ref:
            url += f"?ref={ref}"
        headers = {**self._headers, "Accept": "application/vnd.github.v3.raw"}
        try:
            resp = requests.get(url, headers=headers, timeout=self._timeout)
            return resp.text if resp.status_code == 200 else ""
        except Exception as e:
            logger.error(f"File fetch failed ({path}): {e}")
            return ""

    def post_comment(self, number, body):
        if self._dry_run:
            logger.info(f"[DRY RUN] Comment on #{number}: {body[:80]}...")
            return True
        return self._post(f"/repos/{self._repo}/issues/{number}/comments", {"body": body})

    def add_labels(self, number, labels):
        if not labels:
            return True
        if self._dry_run:
            logger.info(f"[DRY RUN] Labels on #{number}: {labels}")
            return True
        return self._post(f"/repos/{self._repo}/issues/{number}/labels", {"labels": labels})

    def close_issue(self, number):
        if self._dry_run:
            logger.info(f"[DRY RUN] Close #{number}")
            return True
        try:
            resp = requests.patch(
                f"https://api.github.com/repos/{self._repo}/issues/{number}",
                headers=self._headers, json={"state": "closed"}, timeout=self._timeout,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Close failed: {e}")
            return False

    def _get(self, path):
        try:
            resp = requests.get(f"https://api.github.com{path}", headers=self._headers, timeout=self._timeout)
            if resp.status_code == 200:
                return resp.json()
            logger.error(f"GET {path}: {resp.status_code}")
        except Exception as e:
            logger.error(f"GET {path}: {e}")
        return None

    def _post(self, path, payload):
        try:
            resp = requests.post(f"https://api.github.com{path}", headers=self._headers, json=payload, timeout=self._timeout)
            if resp.status_code in (200, 201):
                return True
            logger.error(f"POST {path}: {resp.status_code}")
        except Exception as e:
            logger.error(f"POST {path}: {e}")
        return False
