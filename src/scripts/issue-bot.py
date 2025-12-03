#!/usr/bin/env python3
import requests
import json
import os
import sys
import boto3
import logging
import base64
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeequIssueBot:
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        self.event_type = os.getenv('EVENT_TYPE', 'issues')
        self.deequ_context = self.load_deequ_context()
        self.system_prompt = os.getenv('BOT_SYSTEM_PROMPT', self.get_default_prompt())
        self.bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-haiku-20240307-v1:0')

    def load_deequ_context(self):
        """Load Deequ knowledge base from S3 or environment variable"""
        try:
            # Try S3 first
            s3 = boto3.client('s3')
            bucket = os.getenv('KB_S3_BUCKET', 'deequ-knowledge-base')
            key = 'deequ-kb.md'
            
            response = s3.get_object(Bucket=bucket, Key=key)
            return response['Body'].read().decode('utf-8')[:8000]
        except:
            # Fallback to environment variable (repository secret)
            kb_content = os.getenv('DEEQU_KNOWLEDGE_BASE')
            if kb_content:
                return kb_content[:8000]
            
            # Final fallback
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
    "confidence": "high|medium|low",
    "reasoning": "Brief explanation of your assessment"
}

Assessment Guidelines:
- can_solve = true for: usage questions, configuration issues, simple bugs with clear solutions
- can_solve = false for: regression bugs, complex code changes, performance issues, new features
- Use "ESCALATE" only when you cannot provide any helpful guidance
- Always provide reasoning for your decision
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
                    
                    # Log detailed analysis for debugging
                    logger.info(f"AI Analysis - can_solve: {can_solve}, confidence: {confidence}, category: {parsed_response.get('category')}")
                    logger.info(f"AI Reasoning: {parsed_response.get('reasoning', 'No reasoning provided')}")
                    
                    # Allow low confidence responses through, just log them
                    if can_solve and confidence == 'low':
                        logger.info(f"Low confidence response, but allowing it through")
                    
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

    def enhance_knowledge_base_with_validation(self, issue_data, analysis):
        """Enhance KB only after validating customer sentiment"""
        # First check if there's customer feedback
        feedback = self.analyze_customer_feedback(issue_data)
        
        # Only learn from positive feedback
        if feedback['sentiment'] == 'positive' and feedback['confidence'] > 0.4:
            logger.info(f"Positive feedback detected (confidence: {feedback['confidence']:.2f})")
            self.enhance_knowledge_base_if_needed(issue_data, analysis)
        elif feedback['sentiment'] == 'negative':
            logger.info(f"Negative feedback detected - not learning from this interaction")
            self.log_escalation_pattern(issue_data, 'negative_customer_feedback')
        else:
            # No clear feedback yet - learn tentatively from unsolved issues
            if not analysis.get('can_solve', False):
                logger.info("Learning tentatively from unsolved issue - will validate with future feedback")
                self.enhance_knowledge_base_if_needed(issue_data, analysis)
            else:
                logger.info("No feedback on solved issue - waiting for validation before learning")

    def enhance_knowledge_base_if_needed(self, issue_data, analysis):
        """Dynamically enhance KB if bot cannot solve the issue"""
        if analysis.get('can_solve', False):
            return  # No enhancement needed
        
        # Rate limiting - only enhance once per hour
        last_update_key = 'kb-last-update'
        try:
            s3 = boto3.client('s3')
            bucket = os.getenv('KB_S3_BUCKET', 'deequ-knowledge-base')
            
            # Check last update time
            try:
                response = s3.head_object(Bucket=bucket, Key=last_update_key)
                last_modified = response['LastModified']
                if (datetime.datetime.now(datetime.timezone.utc) - last_modified).total_seconds() < 3600:  # 1 hour
                    logger.info("KB enhancement rate limited")
                    return
            except:
                pass  # No previous update file
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return
        
        # Extract key terms from issue
        title = issue_data.get('title', '')
        body = issue_data.get('body', '')
        issue_content = f"{title}\n{body}"
        
        # Check if similar content already exists in KB
        if self.is_duplicate_content(issue_content):
            logger.info("Similar content already exists in KB")
            return
        
        # Search repository for relevant information
        key_terms = self.extract_key_terms(issue_content)
        if not key_terms:
            return
        
        try:
            # Search GitHub API for relevant files
            headers = {'Authorization': f'token {self.github_token}'} if self.github_token else {}
            search_url = f"https://api.github.com/search/code?q={'+'.join(key_terms)}+repo:{os.getenv('GITHUB_REPOSITORY', 'sudsali/deequ')}"
            
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code != 200:
                return
            
            search_results = response.json().get('items', [])[:3]  # Top 3 results
            if not search_results:
                return
            
            # Get file contents
            repo_context = ""
            for result in search_results:
                file_url = f"https://api.github.com/repos/{os.getenv('GITHUB_REPOSITORY', 'sudsali/deequ')}/contents/{result['path']}"
                file_response = requests.get(file_url, headers=headers, timeout=10)
                
                if file_response.status_code == 200:
                    content = base64.b64decode(file_response.json()['content']).decode('utf-8')
                    repo_context += f"\n## {result['path']}\n{content[:1000]}\n"  # Limit size
            
            if repo_context:
                self.update_knowledge_base(issue_content, repo_context)
                
                # Update rate limit marker
                s3.put_object(Bucket=bucket, Key=last_update_key, Body=b'updated')
                
        except Exception as e:
            logger.error(f"KB enhancement failed: {e}")
    
    def analyze_customer_feedback(self, issue_data):
        """Analyze follow-up comments using sentiment analysis for feedback on bot responses"""
        comments = issue_data.get('recent_comments', [])
        bot_comment_found = False
        feedback_scores = []
        
        for comment in comments:
            author = comment.get('user', {}).get('login', '')
            body = comment.get('body', '')
            
            # Track bot comments
            if author == 'github-actions[bot]':
                bot_comment_found = True
                continue
                
            # Only analyze comments after bot response
            if bot_comment_found and len(body.strip()) > 10:
                sentiment_score = self.get_sentiment_score(body)
                if sentiment_score is not None:
                    feedback_scores.append(sentiment_score)
        
        if not feedback_scores:
            return {'sentiment': 'neutral', 'confidence': 0}
        
        avg_sentiment = sum(feedback_scores) / len(feedback_scores)
        
        # Classify sentiment
        if avg_sentiment > 0.3:
            return {'sentiment': 'positive', 'confidence': avg_sentiment}
        elif avg_sentiment < -0.3:
            return {'sentiment': 'negative', 'confidence': abs(avg_sentiment)}
        else:
            return {'sentiment': 'neutral', 'confidence': abs(avg_sentiment)}
    
    def get_sentiment_score(self, text):
        """Use Bedrock to analyze sentiment of customer feedback"""
        try:
            sentiment_prompt = f"""Analyze the sentiment of this GitHub comment about a technical solution.
            
Comment: "{text}"

Respond with only a number between -1.0 (very negative) and 1.0 (very positive):
- 1.0: Very positive (solution worked perfectly)
- 0.5: Positive (solution helped)
- 0.0: Neutral (no clear sentiment)
- -0.5: Negative (solution didn't help)
- -1.0: Very negative (solution made things worse)

Number only:"""

            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'messages': [{'role': 'user', 'content': sentiment_prompt}],
                    'max_tokens': 10,
                    'temperature': 0.1
                })
            )
            
            result = json.loads(response['body'].read())
            sentiment_text = result['content'][0]['text'].strip()
            
            # Parse the sentiment score
            try:
                score = float(sentiment_text)
                return max(-1.0, min(1.0, score))  # Clamp between -1 and 1
            except ValueError:
                logger.warning(f"Could not parse sentiment score: {sentiment_text}")
                return None
                
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return None

    def is_duplicate_content(self, issue_content):
        """Use Bedrock to check if KB already covers this issue"""
        try:
            check_prompt = f"""Current Knowledge Base:
{self.deequ_context[:2000]}

New Issue:
{issue_content[:500]}

Does the knowledge base already contain sufficient information to help with this issue? 
Respond with only "YES" or "NO"."""

            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'messages': [{'role': 'user', 'content': check_prompt}],
                    'max_tokens': 10,
                    'temperature': 0.1
                })
            )
            
            result = json.loads(response['body'].read())
            answer = result['content'][0]['text'].strip().upper()
            
            return answer == "YES"
            
        except Exception as e:
            logger.error(f"Duplicate check failed: {e}")
            return False  # Default to allowing enhancement
    
    def extract_key_terms(self, issue_content):
        """Extract search terms from issue"""
        terms = []
        content_lower = issue_content.lower()
        
        if 'hasnumberofdistinctvalues' in content_lower:
            terms.extend(['hasNumberOfDistinctValues', 'Histogram'])
        if 'count column' in content_lower:
            terms.extend(['column', 'count', 'conflict'])
        if 'dqdl' in content_lower:
            terms.extend(['DQDL', 'EvaluateDataQuality'])
        if any(word in content_lower for word in ['error', 'exception', 'fail']):
            terms.extend(['error', 'exception'])
        
        return terms[:5]  # Limit terms
    
    def update_knowledge_base(self, issue_content, repo_context):
        """Update KB and save to S3"""
        try:
            enhancement_prompt = f"""Add missing information to help with this issue:

Issue: {issue_content[:500]}

Repository Context: {repo_context}

Provide ONLY the new section to append to the knowledge base. Be concise."""

            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'messages': [{'role': 'user', 'content': enhancement_prompt}],
                    'max_tokens': 800,
                    'temperature': 0.3
                })
            )
            
            result = json.loads(response['body'].read())
            new_section = result['content'][0]['text']
            
            # Update KB and save to S3
            enhanced_kb = f"{self.deequ_context}\n\n{new_section}"
            
            s3 = boto3.client('s3')
            bucket = os.getenv('KB_S3_BUCKET', 'deequ-knowledge-base')
            s3.put_object(
                Bucket=bucket,
                Key='deequ-kb.md',
                Body=enhanced_kb.encode('utf-8'),
                ServerSideEncryption='AES256'
            )
            
            self.deequ_context = enhanced_kb
            logger.info("Knowledge base enhanced and saved to S3")
            
        except Exception as e:
            logger.error(f"KB update failed: {e}")

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
        body = issue_data.get('body', '') or 'No description provided'
        issue_url = issue_data.get('html_url', '')
        bot_thoughts = analysis.get('response', 'No analysis available')
        
        # Truncate body smartly - keep complete sentences
        if len(body) > 1500:
            truncated_body = body[:1500]
            last_period = truncated_body.rfind('.')
            if last_period > 1000:  # Keep if reasonable length
                body = truncated_body[:last_period + 1] + f"\n\n*[Truncated - <{issue_url}|View Full Issue>]*"
            else:
                body = truncated_body + f"...\n\n*[Truncated - <{issue_url}|View Full Issue>]*"
        
        # Prepare bot's analysis with more detail
        if bot_thoughts != 'ESCALATE' and bot_thoughts:
            solution_text = f"*Bot's Assessment:*\n{bot_thoughts}"
        else:
            confidence = analysis.get('confidence', 'unknown')
            solution_text = f"*Bot Analysis:* Requires human expertise (Confidence: {confidence})"
        
        slack_message = {
            "text": f"🔔 Deequ Issue #{issue_number} Needs Team Attention",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"Deequ Issue #{issue_number} Escalation"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Issue:* <{issue_url}|{title}>"
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
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "View on GitHub"
                            },
                            "url": issue_url,
                            "style": "primary"
                        }
                    ]
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

        # Add better disclaimer encouraging feedback
        comment_with_disclaimer = f"""{response_text}

---
*This response was generated with AI assistance. If this doesn't solve your issue or you need clarification, please let us know and a human maintainer will help.*"""

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
    
    if analysis.get('can_solve', False):
        # Bot can solve it - post solution
        bot.post_comment(issue_number, analysis['response'])
        logger.info("Bot provided solution")
    else:
        # Always comment first, then escalate
        escalation_comment = """Thank you for reporting this issue.

This requires attention from our maintainer team. The team has been notified and will review this shortly.

We appreciate your patience and will get back to you as soon as possible."""
        
        bot.post_comment(issue_number, escalation_comment)
        
        # Try to enhance knowledge base with sentiment validation
        bot.enhance_knowledge_base_with_validation(issue_data, analysis)
        
        # Then escalate to team via Slack
        bot.send_to_slack(issue_number, issue_data, analysis)
        logger.info("Posted acknowledgment comment and escalated to team via Slack")
        
        # Log escalation for pattern analysis
        escalation_reason = 'ai_cannot_solve' if analysis.get('can_solve') == False else 'low_confidence'
        bot.log_escalation_pattern(issue_data, escalation_reason)

if __name__ == "__main__":
    main()
