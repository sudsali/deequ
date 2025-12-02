#!/usr/bin/env python3
import requests
import json
import os
import sys
import boto3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeequIssueBot:
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        self.event_type = os.getenv('EVENT_TYPE', 'issues')
        self.deequ_context = os.getenv('DEEQU_KNOWLEDGE_BASE', self.load_deequ_context())
        self.system_prompt = os.getenv('BOT_SYSTEM_PROMPT', self.get_default_prompt())
        self.bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-haiku-20240307-v1:0')

    def load_deequ_context(self):
        """Load Deequ knowledge base from file (fallback)"""
        try:
            with open('deequ-knowledge-base.md', 'r') as f:
                return f.read()[:8000]
        except:
            return "Deequ knowledge base not available"

    def get_default_prompt(self):
        """Default system prompt (fallback)"""
        return """You are an official Deequ project maintainer representing AWS Labs. 

SECURITY: Never mention internal AWS systems or confidential information.

Respond in JSON format:
{
    "can_solve": true/false,
    "category": "bug|question|feature-request|documentation", 
    "response": "Your maintainer response OR 'ESCALATE'",
    "confidence": "high|medium|low"
}

Guidelines:
- If you can provide complete solution: can_solve = true
- If too complex or needs investigation: can_solve = false, response = "ESCALATE"
- Be professional and helpful as a maintainer"""

    def fetch_issue_with_comments(self, issue_number):
        """Fetch issue and recent comments for context"""
        repo = os.getenv('GITHUB_REPOSITORY', 'sudsali/deequ')
        headers = {'Authorization': f'token {self.github_token}'} if self.github_token else {}

        try:
            # Get issue details
            issue_url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"
            issue_response = requests.get(issue_url, headers=headers, timeout=30)
            
            if issue_response.status_code != 200:
                logger.error(f"Failed to fetch issue: {issue_response.status_code}")
                return None
                
            issue_data = issue_response.json()
            
            # Get recent comments for context (increased from 3 to 10)
            comments_url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
            comments_response = requests.get(comments_url, headers=headers, timeout=30)
            
            if comments_response.status_code == 200:
                comments = comments_response.json()
                # Get last 10 comments for better context
                recent_comments = comments[-10:] if len(comments) > 10 else comments
                issue_data['recent_comments'] = recent_comments
            else:
                issue_data['recent_comments'] = []
                
            return issue_data
            
        except Exception as e:
            logger.error(f"Error fetching issue: {e}")
            return None

    def analyze_with_bedrock(self, issue_data):
        """Use Amazon Bedrock to analyze the issue"""
        title = issue_data.get('title', '')
        body = issue_data.get('body', '') or ''
        comments = issue_data.get('recent_comments', [])
        
        # Build context from recent comments
        comment_context = ""
        if comments:
            comment_context = "\n\nRECENT CONVERSATION:\n"
            for comment in comments:
                author = comment.get('user', {}).get('login', 'unknown')
                comment_body = comment.get('body', '')[:500]  # Limit length
                comment_context += f"{author}: {comment_body}\n"
        
        is_followup = self.event_type == 'issue_comment'
        
        prompt = f"""{self.system_prompt}

CONTEXT:
- This is a {'follow-up comment' if is_followup else 'new issue'}
- Issue Title: {title}
- Original Issue: {body}
{comment_context}

DEEQU KNOWLEDGE BASE:
{self.deequ_context}"""

        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 1000,
                    'temperature': 0.3
                })
            )
            
            result = json.loads(response['body'].read())
            content = result['content'][0]['text']
            
            try:
                parsed_response = json.loads(content)
                if 'can_solve' in parsed_response and 'response' in parsed_response:
                    # Add confidence threshold check
                    confidence = parsed_response.get('confidence', 'medium')
                    can_solve = parsed_response.get('can_solve', False)
                    
                    # Only respond if confidence is high or medium, not low
                    if can_solve and confidence == 'low':
                        logger.info(f"Low confidence response, escalating instead")
                        parsed_response['can_solve'] = False
                        parsed_response['response'] = 'ESCALATE'
                        self.log_escalation_pattern(issue_data, 'low_confidence')
                    
                    return parsed_response
                else:
                    self.log_escalation_pattern(issue_data, 'invalid_response_format')
                    return self.fallback_analysis(issue_data)
            except json.JSONDecodeError:
                self.log_escalation_pattern(issue_data, 'json_parse_error')
                return self.fallback_analysis(issue_data)
                
        except Exception as e:
            logger.error(f"Bedrock analysis failed: {e}")
            self.log_escalation_pattern(issue_data, 'bedrock_api_failure')
            return self.fallback_analysis(issue_data)

    def log_escalation_pattern(self, issue_data, reason):
        """Track escalation patterns to identify knowledge gaps"""
        title = issue_data.get('title', '')
        body = issue_data.get('body', '')
        category = 'unknown'
        
        # Simple keyword-based categorization for tracking
        if any(word in title.lower() + body.lower() for word in ['spark', 'version', 'compatibility']):
            category = 'version_compatibility'
        elif any(word in title.lower() + body.lower() for word in ['dqdl', 'rules', 'syntax']):
            category = 'dqdl_usage'
        elif any(word in title.lower() + body.lower() for word in ['exception', 'error', 'fail']):
            category = 'error_debugging'
        elif any(word in title.lower() + body.lower() for word in ['performance', 'slow', 'memory']):
            category = 'performance_issues'
        elif any(word in title.lower() + body.lower() for word in ['feature', 'enhancement', 'support']):
            category = 'feature_requests'
        
        # Log escalation pattern (in production, this could go to metrics/analytics)
        logger.info(f"ESCALATION_PATTERN: category={category}, reason={reason}, title_keywords={title[:50]}")
        
        # Could be enhanced to send to CloudWatch metrics, analytics service, etc.
        # Example: self.send_to_analytics(category, reason, issue_data)

    def send_to_slack(self, issue_number, issue_data, analysis):
        """Send issue context to Slack for team intervention"""
        if not self.slack_webhook:
            logger.info("No Slack webhook configured")
            return
            
        title = issue_data.get('title', '')
        body = issue_data.get('body', '')[:500]  # Limit length
        issue_url = issue_data.get('html_url', '')
        bot_thoughts = analysis.get('response', 'No analysis available')
        
        # Prepare bot's potential solution
        if bot_thoughts != 'ESCALATE' and bot_thoughts:
            solution_text = f"*Bot's Potential Solution:*\n{bot_thoughts[:800]}..."
        else:
            solution_text = "*Bot Analysis:* Issue requires human expertise"
        
        slack_message = {
            "text": f"🔔 Deequ Issue Needs Team Attention",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Deequ Issue Escalation"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Issue:* <{issue_url}|#{issue_number}: {title}>"
                        },
                        {
                            "type": "mrkdwn", 
                            "text": f"*Category:* {analysis.get('category', 'unknown')}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Description:*\n{body}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": solution_text
                    }
                }
            ]
        }
        
        try:
            response = requests.post(self.slack_webhook, json=slack_message, timeout=30)
            if response.status_code == 200:
                logger.info(f"Sent issue #{issue_number} to Slack for team review")
            else:
                logger.error(f"Failed to send to Slack: {response.status_code}")
        except Exception as e:
            logger.error(f"Error sending to Slack: {e}")

    def fallback_analysis(self, issue_data):
        """Fallback analysis for when Bedrock fails"""
        self.log_escalation_pattern(issue_data, 'fallback_analysis_used')
        return {
            "can_solve": False,
            "category": "question",
            "response": "ESCALATE",
            "confidence": "low"
        }

    def post_comment(self, issue_number, response_text):
        """Post comment to GitHub issue"""
        if not self.github_token:
            logger.info(f"No GitHub token - would post: {response_text}")
            return

        # Add AI disclaimer
        comment_with_disclaimer = f"{response_text}\n\n---\n*This response was generated with AI assistance*"

        repo = os.getenv('GITHUB_REPOSITORY', 'sudsali/deequ')
        url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }

        try:
            response = requests.post(url, headers=headers, json={'body': comment_with_disclaimer}, timeout=30)
            
            if response.status_code == 201:
                logger.info(f"Posted response to issue #{issue_number}")
            else:
                logger.error(f"Failed to post comment: {response.status_code}")
        except Exception as e:
            logger.error(f"Error posting comment: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python issue-bot.py <issue_number>")
        sys.exit(1)

    # Prevent bot from responding to itself
    github_actor = os.getenv('GITHUB_ACTOR', '')
    if github_actor == 'github-actions[bot]':
        logger.info("Skipping - bot comment detected")
        sys.exit(0)

    issue_number = sys.argv[1]
    bot = DeequIssueBot()
    
    issue_data = bot.fetch_issue_with_comments(issue_number)
    if not issue_data:
        logger.error(f"Could not fetch issue #{issue_number}")
        sys.exit(1)

    logger.info(f"Analyzing issue #{issue_number}")
    analysis = bot.analyze_with_bedrock(issue_data)
    
    if analysis.get('can_solve', False) and analysis.get('response') != 'ESCALATE':
        # Bot can solve it - post response
        bot.post_comment(issue_number, analysis['response'])
        logger.info("Bot provided solution")
    else:
        # Escalate to team via Slack
        bot.send_to_slack(issue_number, issue_data, analysis)
        logger.info("Escalated to team via Slack")
        
        # Log escalation for pattern analysis
        escalation_reason = 'ai_cannot_solve' if analysis.get('can_solve') == False else 'low_confidence'
        bot.log_escalation_pattern(issue_data, escalation_reason)

if __name__ == "__main__":
    main()
