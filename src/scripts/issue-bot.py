#!/usr/bin/env python3
import requests
import json
import os
import sys
import boto3

class DeequIssueBot:
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.deequ_context = self.load_deequ_context()
        self.bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

    def load_deequ_context(self):
        """Load Deequ knowledge base"""
        try:
            with open('deequ-knowledge-base.md', 'r') as f:
                return f.read()[:8000]  # Limit context size
        except:
            return "Deequ knowledge base not available"

    def fetch_issue(self, issue_number):
        """Fetch issue from repository"""
        url = f"https://api.github.com/repos/{os.getenv('GITHUB_REPOSITORY', 'awslabs/deequ')}/issues/{issue_number}"
        headers = {'Authorization': f'token {self.github_token}'} if self.github_token else {}

        response = requests.get(url, headers=headers)
        return response.json() if response.status_code == 200 else None

    def analyze_with_bedrock(self, issue_data):
        """Use Amazon Bedrock to analyze the issue"""
        title = issue_data['title']
        body = issue_data.get('body', '') or ''
        
        prompt = f"""You are a Deequ expert analyzing GitHub issues. Use the knowledge base to provide helpful, specific solutions.

DEEQU KNOWLEDGE BASE:
{self.deequ_context}

ISSUE TO ANALYZE:
Title: {title}
Body: {body}

Analyze this issue and provide:
1. Issue type (bug/question/enhancement)  
2. Specific problem identification
3. Actionable solution with code examples if applicable
4. Whether team review is needed

Respond in JSON format:
{{
    "category": "bug|question|enhancement",
    "label": "bug|question|enhancement",
    "response": "Detailed solution with specific Deequ guidance and code examples",
    "team_review": true|false
}}

Focus on providing specific solutions for:
- Version compatibility (Spark versions)
- Code bugs with stack traces
- DQDL usage questions  
- API usage and constraints
- Build configuration issues"""

        try:
            response = self.bedrock.invoke_model(
                modelId='anthropic.claude-3-haiku-20240307-v1:0',
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 1000,
                    'temperature': 0.3
                })
            )
            
            result = json.loads(response['body'].read())
            content = result['content'][0]['text']
            
            # Parse JSON response
            try:
                return json.loads(content)
            except:
                # Fallback if JSON parsing fails
                return {
                    "category": "question",
                    "label": "question", 
                    "response": content,
                    "team_review": False
                }
                
        except Exception as e:
            print(f"Bedrock analysis failed: {e}")
            return self.fallback_analysis(issue_data)

    def fallback_analysis(self, issue_data):
        """Fallback analysis if Bedrock fails"""
        title = issue_data['title'].lower()
        body = issue_data.get('body', '').lower()
        
        if 'hasnumberofdistinctvalues' in body and 'exception' in body:
            return {
                "category": "bug",
                "label": "bug",
                "response": "This is a known issue with `hasNumberOfDistinctValues` when DataFrames contain reserved column names like 'count'. The histogram implementation creates column name conflicts. **Workaround**: Rename your 'count' column before applying the constraint. This needs a code fix from the team.",
                "team_review": True
            }
        elif any(x in body for x in ['spark 2.4', 'spark 2.3', 'compatibility']):
            return {
                "category": "question",
                "label": "question",
                "response": "**Version Compatibility**: Use Deequ 1.x for Spark 2.x:\n- Spark 2.4: `com.amazon.deequ:deequ:1.2.2-spark-2.4`\n- Spark 2.3: `com.amazon.deequ:deequ:1.2.2-spark-2.3`\n\nDeequ 2.x only supports Spark 3.1+.",
                "team_review": False
            }
        else:
            return {
                "category": "question", 
                "label": "question",
                "response": "Thank you for your issue. Please provide more details about your specific use case, error messages, or code examples for targeted assistance.",
                "team_review": False
            }

    def post_comment(self, issue_number, analysis):
        """Post analysis comment to GitHub issue"""
        if not self.github_token:
            print("No GitHub token - would post:", analysis['response'])
            return

        url = f"https://api.github.com/repos/{os.getenv('GITHUB_REPOSITORY')}/issues/{issue_number}/comments"
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }

        team_flag = "🔍 *Needs team review*" if analysis.get('team_review') else "✅ *Community can help resolve*"
        
        comment_body = f"""🤖 **Deequ AI Bot Analysis**

**Category**: {analysis['category']}
**Suggested Label**: `{analysis['label']}`

{analysis['response']}

{team_flag}

---
*Analyzed using Amazon Bedrock with Deequ knowledge base*"""

        response = requests.post(url, headers=headers, json={'body': comment_body})
        
        if response.status_code == 201:
            print(f"✅ Posted AI analysis to issue #{issue_number}")
        else:
            print(f"❌ Failed to post comment: {response.status_code}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python issue-bot.py <issue_number>")
        sys.exit(1)

    issue_number = sys.argv[1]
    bot = DeequIssueBot()
    
    issue_data = bot.fetch_issue(issue_number)
    if not issue_data:
        print(f"❌ Could not fetch issue #{issue_number}")
        sys.exit(1)

    print(f"🤖 Analyzing issue #{issue_number} with Amazon Bedrock")
    analysis = bot.analyze_with_bedrock(issue_data)
    bot.post_comment(issue_number, analysis)

if __name__ == "__main__":
    main()
