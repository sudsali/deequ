#!/usr/bin/env python3
import requests
import json
import os
import sys

class DeequIssueBot:
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.deequ_context = self.load_deequ_context()

    def load_deequ_context(self):
        """Load Deequ knowledge base"""
        try:
            with open('deequ-knowledge-base.md', 'r') as f:
                return f.read()[:3000]
        except:
            return "Deequ knowledge base not available"

    def fetch_issue(self, issue_number):
        """Fetch issue from repository"""
        url = f"https://api.github.com/repos/{os.getenv('GITHUB_REPOSITORY', 'awslabs/deequ')}/issues/{issue_number}"
        headers = {'Authorization': f'token {self.github_token}'} if self.github_token else {}

        response = requests.get(url, headers=headers)
        return response.json() if response.status_code == 200 else None

    def analyze_issue(self, issue_data):
        """Analyze issue with Deequ expertise"""
        title = issue_data['title'].lower()
        body = issue_data.get('body', '').lower() if issue_data.get('body') else ''

        if 'pom.xml' in title or 'maven' in title or 'publish' in title:
            return {
                "label": "enhancement",
                "category": "build_configuration",
                "response": "This appears to be a build configuration issue. This requires updating the pom.xml file and needs team review for repository access permissions.",
                "team_review": True
            }
        elif 'spark' in title or 'version' in title or 'compatibility' in body:
            return {
                "label": "bug",
                "category": "version_compatibility",
                "response": "This appears to be a version compatibility issue. Deequ 2.x requires Spark 3.1+. Please check your versions and use: `com.amazon.deequ:deequ:2.0.0-spark-3.1`",
                "team_review": False
            }
        elif 'dqdl' in title or 'dqdl' in body:
            return {
                "label": "question",
                "category": "dqdl_usage",
                "response": "This is about DQDL (Data Quality Definition Language). Please provide your DQDL rules and error details. Example: `Rules=[IsUnique \"column\", Completeness \"column\" > 0.8]`",
                "team_review": False
            }
        else:
            return {
                "label": "question",
                "category": "general",
                "response": "Thank you for your issue. Please provide more details about your use case, error messages, or code examples for better assistance.",
                "team_review": False
            }

    def post_comment(self, issue_number, analysis):
        """Post analysis comment to GitHub issue"""
        if not self.github_token:
            print("No GitHub token - would post:", analysis['response'])
            return

        url = f"https://api.github.com/repos/{os.getenv('GITHUB_REPOSITORY', 'awslabs/deequ')}/issues/{issue_number}/comments"
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }

        comment_body = f"""🤖 **Deequ Bot Analysis**

**Category**: {analysis['category']}
**Suggested Label**: `{analysis['label']}`

{analysis['response']}

{'⚠️ *This issue requires team review*' if analysis['team_review'] else '✅ *This can likely be resolved directly*'}

---
*This is an automated analysis. A team member will review shortly.*"""

        data = {"body": comment_body}
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 201:
            print(f"✅ Posted analysis comment to issue #{issue_number}")
        else:
            print(f"❌ Failed to post comment: {response.status_code}")

    def process_issue(self, issue_number):
        """Main processing function"""
        print(f"🔍 Analyzing issue #{issue_number}...")

        issue = self.fetch_issue(issue_number)
        if not issue:
            print("❌ Issue not found")
            return

        print(f"📝 Issue: {issue['title']}")
        analysis = self.analyze_issue(issue)

        self.post_comment(issue_number, analysis)

        return {
            "issue_number": issue_number,
            "title": issue['title'],
            "analysis": analysis
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python issue-bot.py <issue_number>")
        sys.exit(1)

    bot = DeequIssueBot()
    bot.process_issue(sys.argv[1])
