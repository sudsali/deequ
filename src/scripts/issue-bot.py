#!/usr/bin/env python3
import requests
import json
import os
import sys
import boto3
import logging
import base64
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeequIssueBot:
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        self.event_type = os.getenv('EVENT_TYPE', 'issues')
        self.system_prompt = os.getenv('BOT_SYSTEM_PROMPT', self.get_default_prompt())
        self.bedrock = boto3.client('bedrock-runtime')
        self.model_id = os.getenv('BEDROCK_MODEL_ID')
        self.api_version = os.getenv('BEDROCK_API_VERSION')
        
        if not self.model_id:
            raise ValueError("BEDROCK_MODEL_ID environment variable is required")
        if not self.system_prompt or "not configured" in self.system_prompt:
            raise ValueError("BOT_SYSTEM_PROMPT environment variable is required")
        if not self.api_version:
            raise ValueError("BEDROCK_API_VERSION environment variable is required")
        
        try:
            self.deequ_context = self.load_deequ_context()
        except Exception as e:
            logger.error(f"Failed to load KB during init: {e}")
            self.deequ_context = "Deequ knowledge base not available"

    def load_deequ_context(self):
        """Load Deequ knowledge base from S3 or environment variable"""
        try:
            bucket = os.getenv('KB_S3_BUCKET')
            key = os.getenv('KB_S3_KEY')
            
            if not bucket or not key:
                raise ValueError("KB_S3_BUCKET and KB_S3_KEY must be configured")
            
            s3 = boto3.client('s3')
            try:
                s3.head_bucket(Bucket=bucket)
            except Exception as e:
                logger.warning(f"S3 bucket {bucket} not accessible: {e}")
                raise Exception("S3 bucket not accessible")
            
            response = s3.get_object(Bucket=bucket, Key=key)
            kb_content = response['Body'].read().decode('utf-8')
            logger.info(f"Loaded {len(kb_content)} chars from S3 KB")
            return kb_content
        except Exception as e:
            logger.warning(f"S3 KB load failed: {e}")
            kb_content = os.getenv('DEEQU_KNOWLEDGE_BASE')
            if kb_content:
                logger.info("Using fallback KB from environment")
                return kb_content
            
            logger.warning("No knowledge base available")
            return "Deequ knowledge base not available"

    def safe_github_request(self, url, headers):
        """Make GitHub API request with rate limit handling"""
        try:
            logger.info(f"Making GitHub API request to: {url}")
            response = requests.get(url, headers=headers, timeout=10)
            logger.info(f"GitHub API response: {response.status_code}")
            
            if response.status_code == 403:
                logger.error(f"GitHub API 403 error: {response.text}")
                if 'rate limit' in response.text.lower():
                    logger.warning("GitHub API rate limited - skipping repository search")
                    return None
            elif response.status_code == 401:
                logger.error("GitHub API authentication failed - check GITHUB_TOKEN")
            elif response.status_code != 200:
                logger.error(f"GitHub API error {response.status_code}: {response.text}")
                
            return response if response.status_code == 200 else None
        except Exception as e:
            logger.error(f"GitHub API request failed: {e}")
            return None

    def count_tokens_estimate(self, text):
        """Rough estimate: 1 token ≈ 4 characters for safety"""
        return len(text) // 4

    def should_search_repository(self, issue_data):
        """Use AI to determine if repository search is needed"""
        try:
            title = issue_data.get('title', '')
            body = issue_data.get('body', '')
            
            logger.info(f"Evaluating repository search for: '{title}'")
            logger.info(f"Body preview: '{body[:100]}...'")
            
            search_prompt = os.getenv('REPO_SEARCH_PROMPT')
            if not search_prompt:
                logger.warning("REPO_SEARCH_PROMPT not configured, skipping repository search")
                return False
            
            logger.info("REPO_SEARCH_PROMPT is configured, calling Bedrock for decision")
            
            prompt = f"""{search_prompt}

Issue: {title}
{body}"""

            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    'anthropic_version': self.api_version,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 50,
                    'temperature': 0.1
                })
            )
            
            result = json.loads(response['body'].read())
            answer = result['content'][0]['text'].strip()
            
            logger.info(f"Bedrock raw response: '{answer}'")
            
            lines = answer.split('\n')
            decision = lines[0].strip().upper()
            needs_search = "YES" in decision
            
            if needs_search and len(lines) > 1:
                self.current_search_terms = lines[1].strip().split()[:5]
                logger.info(f"Extracted search terms: {self.current_search_terms}")
            else:
                self.current_search_terms = []
                logger.info("No search terms extracted")
            
            logger.info(f"AI repository search decision: {decision} - {'Will search' if needs_search else 'Skip search'}")
            return needs_search
            
        except Exception as e:
            logger.error(f"Repository search decision failed: {e}")
            return False  # Default to no search on error

    def get_enhanced_context(self, issue_data):
        """Get enhanced context by combining KB with live repository search"""
        logger.info("Starting get_enhanced_context - checking if repository search needed")
        logger.info(f"Issue title: {issue_data.get('title', '')}")
        logger.info(f"Issue body preview: {issue_data.get('body', '')[:100]}...")
        base_context = self.deequ_context
        logger.info(f"Base context length: {len(base_context)} chars")
        
        if self.should_search_repository(issue_data):
            logger.info("AI determined repository search is needed")
            repo_context = self.search_repository_docs(issue_data)
            if repo_context:
                enhanced_context = f"{base_context}\n\n## Repository Context:\n{repo_context}"
                logger.info(f"Enhanced context with {len(repo_context)} chars from repository")
            else:
                logger.info("Repository search returned no results, using base KB only")
                enhanced_context = base_context
        else:
            logger.info("AI determined repository search not needed, using KB only")
            enhanced_context = base_context
        
        estimated_tokens = self.count_tokens_estimate(enhanced_context)
        if estimated_tokens > 8000:
            logger.info(f"Context too large ({estimated_tokens} tokens), applying smart truncation")
            enhanced_context = self.smart_truncate(enhanced_context, issue_data)
        
        logger.info(f"Final enhanced context length: {len(enhanced_context)} chars")
        return enhanced_context

    def smart_truncate(self, content, issue_data):
        """Keep most relevant sections when content is too large"""
        try:
            sections = content.split('\n## ')
            if len(sections) <= 1:
                target_chars = 7000 * 4
                return content[:target_chars]
            
            search_terms = self.extract_key_terms(f"{issue_data.get('title', '')} {issue_data.get('body', '')}")
            
            scored_sections = []
            for i, section in enumerate(sections):
                score = sum(1 for term in search_terms if term.lower() in section.lower())
                score += max(0, 10 - i)
                scored_sections.append((score, section))
            
            sorted_sections = sorted(scored_sections, key=lambda x: x[0], reverse=True)
            result = ""
            target_chars = 7000 * 4
            
            for score, section in sorted_sections:
                section_text = f"\n## {section}" if result else section
                if len(result + section_text) < target_chars:
                    result += section_text
                else:
                    break
            
            logger.info(f"Smart truncated context from {len(content)} to {len(result)} chars")
            return result
            
        except Exception as e:
            logger.error(f"Smart truncation failed: {e}")
            target_chars = 7000 * 4
            return content[:target_chars]

    def search_repository_docs(self, issue_data):
        """Search repository documentation and examples for relevant context"""
        try:
            search_terms = getattr(self, 'current_search_terms', [])
            
            if not search_terms:
                return ""
            
            headers = {'Authorization': f'token {self.github_token}'} if self.github_token else {}
            repo = os.getenv('GITHUB_REPOSITORY')
            repo_context = ""
            
            for term in search_terms[:3]:
                search_url = f"https://api.github.com/search/code?q={term}+OR+{term.capitalize()}+repo:{repo}+extension:scala"
                logger.info(f"Searching with URL: {search_url}")
                
                response = self.safe_github_request(search_url, headers)
                if response:
                    response_data = response.json()
                    results = response_data.get('items', [])[:2]
                    logger.info(f"Search response status: {response.status_code}, items found: {len(results)}")
                    
                    if response.status_code != 200:
                        logger.error(f"GitHub API error: {response.status_code} - {response_data}")
                    
                    for result in results:
                        file_content = self.get_file_content(result['url'], headers)
                        if file_content:
                            repo_context += f"\n### {result['name']}\n{file_content}\n"
                else:
                    logger.error(f"Failed to get response from GitHub API for term: {term}")
                
                if len(repo_context) > 5000:
                    break
            
            return repo_context
            
        except Exception as e:
            logger.error(f"Repository docs search failed: {e}")
            return ""

    def get_file_content(self, file_url, headers):
        """Get content of a specific file from GitHub API with rate limit handling"""
        try:
            response = self.safe_github_request(file_url, headers)
            if response:
                content = base64.b64decode(response.json()['content']).decode('utf-8')
                return content
        except Exception as e:
            logger.error(f"Failed to get file content: {e}")
        return ""

    def get_default_prompt(self):
        """Fallback when BOT_SYSTEM_PROMPT is not configured"""
        return "System prompt not configured - BOT_SYSTEM_PROMPT environment variable required"

    def fetch_issue_with_comments(self, issue_number):
        """Fetch issue and recent comments for context"""
        repo = os.getenv('GITHUB_REPOSITORY')
        headers = {'Authorization': f'token {self.github_token}'} if self.github_token else {}

        try:
            issue_url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"
            issue_response = self.safe_github_request(issue_url, headers)
            
            if not issue_response:
                logger.error(f"Failed to fetch issue: {issue_number}")
                return None
                
            issue_data = issue_response.json()
            
            comments_url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
            comments_response = self.safe_github_request(comments_url, headers)
            
            if comments_response:
                comments = comments_response.json()
                recent_comments = comments[-10:] if len(comments) > 10 else comments
                issue_data['recent_comments'] = recent_comments
            else:
                issue_data['recent_comments'] = []
                
            return issue_data
            
        except Exception as e:
            logger.error(f"Error fetching issue: {e}")
            return None

    def analyze_with_bedrock(self, issue_data):
        """Use Amazon Bedrock to get human-like response"""
        title = issue_data.get('title', '')
        body = issue_data.get('body', '') or ''
        comments = issue_data.get('recent_comments', [])
        
        comment_context = ""
        if comments:
            comment_context = "\n\nRECENT CONVERSATION:\n"
            for comment in comments:
                author = comment.get('user', {}).get('login', 'unknown')
                comment_body = comment.get('body', '')[:500]  # Limit length
                comment_context += f"{author}: {comment_body}\n"
        
        is_followup = self.event_type == 'issue_comment'
        
        logger.info("About to call get_enhanced_context")
        try:
            enhanced_context = self.get_enhanced_context(issue_data)
            logger.info("Completed get_enhanced_context call")
        except Exception as e:
            logger.error(f"get_enhanced_context failed: {e}")
            enhanced_context = self.deequ_context
        
        prompt = f"""{self.system_prompt}

CONTEXT:
- This is a {'follow-up comment' if is_followup else 'new issue'}
- Issue Title: {title}
- Original Issue: {body}
{comment_context}

DEEQU KNOWLEDGE BASE:
{enhanced_context}"""

        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    'anthropic_version': self.api_version,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 1000,
                    'temperature': 0.3
                })
            )
            
            result = json.loads(response['body'].read())
            
            if 'content' not in result or not result['content']:
                raise ValueError("Invalid Bedrock response structure")
            
            ai_response = result['content'][0]['text'].strip()
            
            should_escalate = "ESCALATE_TO_TEAM" in ai_response
            
            logger.info(f"AI Response - should_escalate: {should_escalate}")
            if should_escalate:
                logger.info("AI requested escalation to team")
            else:
                logger.info("AI provided direct solution")
            
            return {
                'response': ai_response,
                'should_escalate': should_escalate,
                'category': 'question'
            }
                
        except Exception as e:
            logger.error(f"Bedrock analysis failed: {e}")
            self.log_escalation_pattern(issue_data, 'bedrock_api_failure')
            return self.fallback_analysis(issue_data)

    def enhance_knowledge_base_with_validation(self, issue_data, analysis):
        """Enhance KB only after validating customer sentiment"""
        feedback = self.analyze_customer_feedback(issue_data)
        
        if feedback['sentiment'] == 'positive' and feedback['confidence'] > 0.4:
            logger.info(f"Positive feedback detected (confidence: {feedback['confidence']:.2f})")
            self.enhance_knowledge_base_if_needed(issue_data, analysis)
        elif feedback['sentiment'] == 'negative':
            logger.info(f"Negative feedback detected - not learning from this interaction")
            self.log_escalation_pattern(issue_data, 'negative_customer_feedback')
        else:
            if analysis.get('should_escalate', True):
                logger.info("Learning tentatively from unsolved issue - will validate with future feedback")
                self.enhance_knowledge_base_if_needed(issue_data, analysis)
            else:
                logger.info("No feedback on solved issue - waiting for validation before learning")

    def enhance_knowledge_base_if_needed(self, issue_data, analysis):
        """Dynamically enhance KB if bot cannot solve the issue"""
        if not analysis.get('should_escalate', True):
            return  # No enhancement needed
        
        last_update_key = 'kb-last-update'
        try:
            bucket = os.getenv('KB_S3_BUCKET')
            if not bucket:
                logger.warning("KB_S3_BUCKET not configured, skipping enhancement")
                return
                
            s3 = boto3.client('s3')
            
            try:
                response = s3.head_object(Bucket=bucket, Key=last_update_key)
                last_modified = response['LastModified']
                if (datetime.datetime.now(datetime.timezone.utc) - last_modified).total_seconds() < 3600:  # 1 hour
                    logger.info("KB enhancement rate limited")
                    return
            except Exception as e:
                logger.warning(f"No previous update file: {e}")  # No previous update file
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return
        
        title = issue_data.get('title', '')
        body = issue_data.get('body', '')
        issue_content = f"{title}\n{body}"
        
        if self.is_duplicate_content(issue_content):
            logger.info("Similar content already exists in KB")
            return
        
        key_terms = self.extract_key_terms(issue_content)
        if not key_terms:
            return
        
        repo_context = self.search_repository_docs(issue_data)
        if repo_context:
            self.update_knowledge_base(issue_content, repo_context)
            
            try:
                s3.put_object(Bucket=bucket, Key=last_update_key, Body=b'updated')
                logger.info("KB enhanced with repository context")
            except Exception as e:
                logger.error(f"Failed to update rate limit marker: {e}")

    def analyze_customer_feedback(self, issue_data):
        """Analyze follow-up comments using sentiment analysis for feedback on bot responses"""
        comments = issue_data.get('recent_comments', [])
        bot_has_responded = any(c.get('user', {}).get('login') == 'github-actions[bot]' for c in comments)
        
        if not bot_has_responded:
            return {'sentiment': 'neutral', 'confidence': 0}
        
        feedback_scores = []
        for comment in comments:
            author = comment.get('user', {}).get('login', '')
            body = comment.get('body', '')
            
            if author != 'github-actions[bot]' and len(body.strip()) > 10:
                sentiment_score = self.get_sentiment_score(body)
                if sentiment_score is not None:
                    feedback_scores.append(sentiment_score)
        
        if not feedback_scores:
            return {'sentiment': 'neutral', 'confidence': 0}
        
        avg_sentiment = sum(feedback_scores) / len(feedback_scores)
        
        if avg_sentiment > 0.3:
            return {'sentiment': 'positive', 'confidence': avg_sentiment}
        elif avg_sentiment < -0.3:
            return {'sentiment': 'negative', 'confidence': abs(avg_sentiment)}
        else:
            return {'sentiment': 'neutral', 'confidence': abs(avg_sentiment)}

    def has_negative_feedback_requiring_learning(self, issue_data):
        """Check if bot needs to learn from negative feedback"""
        feedback = self.analyze_customer_feedback(issue_data)
        return feedback['sentiment'] == 'negative' and feedback['confidence'] > 0.4
    
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
                    'anthropic_version': self.api_version,
                    'messages': [{'role': 'user', 'content': sentiment_prompt}],
                    'max_tokens': 10,
                    'temperature': 0.1
                })
            )
            
            result = json.loads(response['body'].read())
            sentiment_text = result['content'][0]['text'].strip()
            
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
                    'anthropic_version': self.api_version,
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
    
    def safe_s3_update(self, bucket, key, content):
        """Safely update S3 with conflict prevention"""
        try:
            s3 = boto3.client('s3')
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_key = f"{key}.tmp.{timestamp}"
            
            s3.put_object(
                Bucket=bucket,
                Key=temp_key,
                Body=content.encode('utf-8'),
                ServerSideEncryption='AES256'
            )
            
            s3.copy_object(
                Bucket=bucket,
                CopySource={'Bucket': bucket, 'Key': temp_key},
                Key=key,
                ServerSideEncryption='AES256'
            )
            
            s3.delete_object(Bucket=bucket, Key=temp_key)
            
            logger.info(f"Safely updated S3 object: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Safe S3 update failed: {e}")
            return False

    def update_knowledge_base(self, issue_content, repo_context):
        """Update knowledge base with new information using safe S3 operations"""
        try:
            if 'github-actions[bot]' in issue_content or 'AI assistance' in issue_content:
                logger.info("Skipping KB update - contains bot content")
                return
            
            enhancement_prompt = f"""Add missing information to help with this issue:

Issue: {issue_content[:500]}

Repository Context: {repo_context}

Provide ONLY the new section to append to the knowledge base. Be concise."""

            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    'anthropic_version': self.api_version,
                    'messages': [{'role': 'user', 'content': enhancement_prompt}],
                    'max_tokens': 800,
                    'temperature': 0.3
                })
            )
            
            result = json.loads(response['body'].read())
            new_section = result['content'][0]['text']
            
            enhanced_kb = f"{self.deequ_context}\n\n{new_section}"
            
            bucket = os.getenv('KB_S3_BUCKET')
            key = os.getenv('KB_S3_KEY')
            if bucket and key and self.safe_s3_update(bucket, key, enhanced_kb):
                self.deequ_context = enhanced_kb
                logger.info("Knowledge base updated successfully")
            
        except Exception as e:
            logger.error(f"KB update failed: {e}")

    def log_escalation_pattern(self, issue_data, reason):
        """Track escalation patterns to identify knowledge gaps"""
        title = issue_data.get('title', '')
        body = issue_data.get('body', '')
        category = 'unknown'
        
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
        
        logger.info(f"ESCALATION_PATTERN: category={category}, reason={reason}, title_keywords={title[:50]}")

    def send_to_slack(self, issue_number, issue_data, analysis):
        """Send issue context to Slack for team intervention"""
        if not self.slack_webhook:
            logger.info("No Slack webhook configured")
            return
            
        title = issue_data.get('title', '')
        issue_url = issue_data.get('html_url', '')
        
        bot_response = analysis.get('response', 'No analysis available')
        ai_category = analysis.get('category', 'unknown')
        should_escalate = analysis.get('should_escalate', True)
        
        if not should_escalate:
            analysis_text = f"**AI Analysis:** Bot provided direct solution\n\n**Category:** {ai_category}"
            solution_text = f"*Bot's Solution (Posted):*\n{bot_response}"
        else:
            analysis_text = f"**AI Analysis:** Issue requires human expertise\n\n**Category:** {ai_category}"
            solution_text = f"*Bot's Assessment:*\n{bot_response}\n\n*Escalated for human review*"
        
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
                            "text": f"*Category:* {ai_category}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": analysis_text
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
            "response": "ESCALATE_TO_TEAM",
            "should_escalate": True,
            "category": "question"
        }

    def post_comment(self, issue_number, response_text):
        """Post comment to GitHub issue"""
        if not self.github_token:
            logger.info("No GitHub token - skipping comment post")
            return

        comment_with_disclaimer = f"""{response_text}

---
*This response was generated with AI assistance. If this doesn't solve your issue or you need clarification, please let us know and a human maintainer will help.*"""

        repo = os.getenv('GITHUB_REPOSITORY')
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

    if bot.has_negative_feedback_requiring_learning(issue_data):
        logger.info("Negative feedback detected - entering learning mode")
        
        if bot.should_search_repository(issue_data):
            repo_context = bot.search_repository_docs(issue_data)
            
            if repo_context:
                title = issue_data.get('title', '')
                learning_prompt = f"""Based on this repository code, provide a correct and helpful answer to: {title}

Repository Code:
{repo_context}

Provide a clear, accurate explanation of what this code does and how to use it."""

                try:
                    response = bot.bedrock.invoke_model(
                        modelId=bot.model_id,
                        body=json.dumps({
                            'anthropic_version': bot.api_version,
                            'messages': [{'role': 'user', 'content': learning_prompt}],
                            'max_tokens': 1000,
                            'temperature': 0.3
                        })
                    )
                    
                    result = json.loads(response['body'].read())
                    learned_content = result['content'][0]['text'].strip()
                    
                    bot.update_knowledge_base(f"{title}\n{issue_data.get('body', '')}", repo_context)
                    
                    correction_comment = f"""Thank you for the feedback! I've learned from the repository code and here's the correct information:

{learned_content}

I apologize for the earlier incorrect response. I've updated my knowledge base to provide better answers in the future."""
                    
                    bot.post_comment(issue_number, correction_comment)
                    logger.info("Posted corrected response after learning")
                    return
                    
                except Exception as e:
                    logger.error(f"Failed to generate learned response: {e}")

    analysis = bot.analyze_with_bedrock(issue_data)
    
    if not analysis.get('should_escalate', True):
        bot.post_comment(issue_number, analysis['response'])
        logger.info("Bot provided solution")
    else:
        escalation_comment = """Thank you for reporting this issue.

This requires attention from our maintainer team. The team has been notified and will review this shortly.

We appreciate your patience and will get back to you as soon as possible."""
        
        bot.post_comment(issue_number, escalation_comment)
        bot.send_to_slack(issue_number, issue_data, analysis)
        logger.info("Posted acknowledgment comment and escalated to team via Slack")

if __name__ == "__main__":
    main()
