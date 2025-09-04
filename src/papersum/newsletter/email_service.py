import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import os
from dataclasses import dataclass

from ..database.models import User, Newsletter
from ..database.session import get_db_session_sync
from .generator import NewsletterService, NewsletterContent


@dataclass
class EmailConfig:
    """Email configuration settings"""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    from_email: str = ""
    from_name: str = "Research Paper Digest"
    use_tls: bool = True


class NewsletterFormatter:
    """Formats newsletter content for email delivery"""
    
    def format_html_email(self, content: Dict[str, Any]) -> str:
        """Format newsletter content as HTML email"""
        
        papers_html = ""
        for i, paper in enumerate(content.get('papers', []), 1):
            papers_html += f"""
            <div style="margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                <h3 style="color: #2c3e50; margin-top: 0;">{i}. {paper['title']}</h3>
                <p style="color: #7f8c8d; margin: 5px 0;"><strong>Authors:</strong> {', '.join(paper['authors'])}</p>
                <p style="margin: 10px 0;">{paper['summary']}</p>
                <p style="font-size: 14px; color: #27ae60; margin: 5px 0;"><strong>Why relevant:</strong> {paper['why_relevant']}</p>
            </div>
            """
        
        tips_html = ""
        for i, tip in enumerate(content.get('coding_tips', []), 1):
            tips_html += f"""
            <div style="margin: 20px 0; padding: 15px; background: #fff; border: 1px solid #e1e5e9; border-radius: 8px;">
                <h3 style="color: #2c3e50; margin-top: 0;">{i}. {tip['title']}</h3>
                <p style="color: #7f8c8d; font-size: 14px; margin: 5px 0;"><strong>Category:</strong> {tip['category']} | <strong>Level:</strong> {tip['difficulty']}</p>
                <p style="margin: 10px 0;">{tip['content']}</p>
            </div>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{content.get('title', 'Your Research Digest')}</title>
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 30px;">
                <h1 style="margin: 0; font-size: 24px;">📚 {content.get('title', 'Your Research Digest')}</h1>
                <p style="margin: 10px 0 0 0; opacity: 0.9;">📅 {datetime.utcnow().strftime('%B %d, %Y')} | ⏱️ {content.get('estimated_read_time', 5)} min read</p>
            </div>
            
            <div style="margin-bottom: 30px;">
                <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">📄 Research Papers ({len(content.get('papers', []))})</h2>
                {papers_html if papers_html else '<p style="color: #7f8c8d;">No new papers this week.</p>'}
            </div>
            
            <div style="margin-bottom: 30px;">
                <h2 style="color: #2c3e50; border-bottom: 2px solid #e74c3c; padding-bottom: 10px;">💡 Coding Tips ({len(content.get('coding_tips', []))})</h2>
                {tips_html if tips_html else '<p style="color: #7f8c8d;">No coding tips this week.</p>'}
            </div>
            
            <div style="text-align: center; margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 8px; color: #7f8c8d; font-size: 14px;">
                <p>🤖 This newsletter was automatically generated based on your research interests.</p>
                <p>Made with ❤️ by Research Paper Summarizer</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def format_text_email(self, content: Dict[str, Any]) -> str:
        """Format newsletter content as plain text email"""
        
        text = f"""
{content.get('title', 'Your Research Digest')}
{'=' * len(content.get('title', 'Your Research Digest'))}

Generated: {datetime.utcnow().strftime('%B %d, %Y')}
Estimated read time: {content.get('estimated_read_time', 5)} minutes

RESEARCH PAPERS ({len(content.get('papers', []))})
{'-' * 40}
"""
        
        for i, paper in enumerate(content.get('papers', []), 1):
            text += f"""
{i}. {paper['title']}
   Authors: {', '.join(paper['authors'])}
   
   {paper['summary']}
   
   Why relevant: {paper['why_relevant']}

"""
        
        text += f"""
CODING TIPS ({len(content.get('coding_tips', []))})
{'-' * 40}
"""
        
        for i, tip in enumerate(content.get('coding_tips', []), 1):
            text += f"""
{i}. {tip['title']}
   Category: {tip['category']} | Level: {tip['difficulty']}
   
   {tip['content']}

"""
        
        text += """
---
🤖 This newsletter was automatically generated based on your research interests.
Made with ❤️ by Research Paper Summarizer
"""
        
        return text


class EmailService:
    """Service for sending newsletter emails"""
    
    def __init__(self, config: Optional[EmailConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        self.formatter = NewsletterFormatter()
        self.newsletter_service = NewsletterService()
    
    def _get_default_config(self) -> EmailConfig:
        """Get email configuration from environment variables"""
        return EmailConfig(
            smtp_server=os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            smtp_port=int(os.getenv('SMTP_PORT', '587')),
            username=os.getenv('SMTP_USERNAME', ''),
            password=os.getenv('SMTP_PASSWORD', ''),
            from_email=os.getenv('FROM_EMAIL', ''),
            from_name=os.getenv('FROM_NAME', 'Research Paper Digest'),
            use_tls=os.getenv('SMTP_TLS', 'true').lower() == 'true'
        )
    
    def send_newsletter(self, user: User, force_send: bool = False) -> bool:
        """Send newsletter to a user"""
        
        try:
            # Check if user should receive newsletter
            if not self._should_send_newsletter(user, force_send):
                self.logger.info(f"Skipping newsletter for user {user.id} - not due yet")
                return False
            
            # Generate newsletter content
            content = self.newsletter_service.generate_newsletter_preview(user.id)
            
            if content.get('error'):
                self.logger.error(f"Failed to generate content for user {user.id}: {content['error']}")
                return False
            
            # Format email content
            html_content = self.formatter.format_html_email(content)
            text_content = self.formatter.format_text_email(content)
            
            # Send email
            success = self._send_email(
                to_email=user.email,
                to_name=user.name,
                subject=content['title'],
                html_content=html_content,
                text_content=text_content
            )
            
            if success:
                # Update user's last newsletter sent time
                with get_db_session_sync() as db:
                    db_user = db.query(User).filter(User.id == user.id).first()
                    if db_user:
                        db_user.last_newsletter_sent = datetime.utcnow()
                        db.commit()
                
                self.logger.info(f"Newsletter sent successfully to {user.email}")
                return True
            else:
                self.logger.error(f"Failed to send newsletter to {user.email}")
                return False
                
        except Exception as e:
            self.logger.error(f"Newsletter sending failed for user {user.id}: {e}")
            return False
    
    def _should_send_newsletter(self, user: User, force_send: bool = False) -> bool:
        """Check if user should receive a newsletter"""
        
        if force_send:
            return True
        
        if not user.last_newsletter_sent:
            return True  # First newsletter
        
        # Check frequency
        frequency = user.newsletter_frequency or 'weekly'
        now = datetime.utcnow()
        
        if frequency == 'daily':
            time_diff = timedelta(days=1)
        elif frequency == 'weekly':
            time_diff = timedelta(weeks=1)
        elif frequency == 'monthly':
            time_diff = timedelta(days=30)
        else:
            time_diff = timedelta(weeks=1)  # Default to weekly
        
        return (now - user.last_newsletter_sent) >= time_diff
    
    def _send_email(
        self, 
        to_email: str, 
        to_name: str, 
        subject: str, 
        html_content: str, 
        text_content: str
    ) -> bool:
        """Send an email with both HTML and text content"""
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.config.from_name} <{self.config.from_email}>"
            msg['To'] = f"{to_name} <{to_email}>"
            
            # Add plain text part
            text_part = MIMEText(text_content, 'plain', 'utf-8')
            msg.attach(text_part)
            
            # Add HTML part
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
            
            # Send email
            if not self.config.username or not self.config.password:
                self.logger.warning("Email credentials not configured - simulating send")
                self.logger.info(f"Would send email to {to_email} with subject '{subject}'")
                return True  # Simulate success for development
            
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            
            if self.config.use_tls:
                server.starttls()
            
            server.login(self.config.username, self.config.password)
            
            # Send email
            text = msg.as_string()
            server.sendmail(self.config.from_email, to_email, text)
            server.quit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email to {to_email}: {e}")
            return False
    
    def send_test_email(self, to_email: str) -> bool:
        """Send a test email to verify configuration"""
        
        test_content = {
            'title': 'Test Newsletter - Research Paper Digest',
            'papers': [{
                'title': 'Test Paper: Machine Learning Fundamentals',
                'authors': ['Test Author'],
                'summary': 'This is a test paper to verify that your newsletter system is working correctly.',
                'why_relevant': 'This is a test to ensure email delivery is functioning.'
            }],
            'coding_tips': [{
                'title': 'Test Tip: Python Best Practices',
                'content': 'This is a test coding tip to verify email formatting.',
                'category': 'best-practices',
                'difficulty': 'beginner'
            }],
            'estimated_read_time': 2
        }
        
        html_content = self.formatter.format_html_email(test_content)
        text_content = self.formatter.format_text_email(test_content)
        
        return self._send_email(
            to_email=to_email,
            to_name="Test User",
            subject="Test Newsletter - Research Paper Digest",
            html_content=html_content,
            text_content=text_content
        )


class NewsletterScheduler:
    """Handles scheduling and batch sending of newsletters"""
    
    def __init__(self, email_service: Optional[EmailService] = None):
        self.logger = logging.getLogger(__name__)
        self.email_service = email_service or EmailService()
    
    def send_weekly_newsletters(self, force_send: bool = False) -> Dict[str, Any]:
        """Send weekly newsletters to all eligible users"""
        
        results = {
            'sent_count': 0,
            'failed_count': 0,
            'skipped_count': 0,
            'errors': []
        }
        
        try:
            with get_db_session_sync() as db:
                # Get all users who should receive newsletters
                users = db.query(User).all()
                
                for user in users:
                    try:
                        success = self.email_service.send_newsletter(user, force_send)
                        
                        if success:
                            results['sent_count'] += 1
                        else:
                            # Check if it was skipped or failed
                            if self.email_service._should_send_newsletter(user, force_send):
                                results['failed_count'] += 1
                                results['errors'].append(f"Failed to send to {user.email}")
                            else:
                                results['skipped_count'] += 1
                                
                    except Exception as e:
                        results['failed_count'] += 1
                        results['errors'].append(f"Error sending to {user.email}: {str(e)}")
                        self.logger.error(f"Failed to send newsletter to user {user.id}: {e}")
                
                self.logger.info(f"Newsletter batch complete: {results['sent_count']} sent, "
                               f"{results['failed_count']} failed, {results['skipped_count']} skipped")
                
        except Exception as e:
            self.logger.error(f"Newsletter batch failed: {e}")
            results['errors'].append(f"Batch processing error: {str(e)}")
        
        return results
    
    def send_newsletter_to_user(self, user_id: int, force_send: bool = False) -> bool:
        """Send newsletter to a specific user"""
        
        try:
            with get_db_session_sync() as db:
                user = db.query(User).filter(User.id == user_id).first()
                
                if not user:
                    self.logger.error(f"User {user_id} not found")
                    return False
                
                return self.email_service.send_newsletter(user, force_send)
                
        except Exception as e:
            self.logger.error(f"Failed to send newsletter to user {user_id}: {e}")
            return False