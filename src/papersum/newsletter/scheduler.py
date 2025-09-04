import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import schedule
import time
import threading
from pathlib import Path
import json

from .email_service import EmailService, NewsletterScheduler


class WeeklyNewsletterScheduler:
    """Automated weekly newsletter scheduling system"""
    
    def __init__(self, email_service: Optional[EmailService] = None):
        self.logger = logging.getLogger(__name__)
        self.email_service = email_service or EmailService()
        self.newsletter_scheduler = NewsletterScheduler(self.email_service)
        self.is_running = False
        self.scheduler_thread = None
        
        # Default schedule: Mondays at 9 AM
        self.schedule_day = "monday"
        self.schedule_time = "09:00"
    
    def start_scheduler(self):
        """Start the automated newsletter scheduler"""
        
        if self.is_running:
            self.logger.warning("Newsletter scheduler is already running")
            return
        
        self.logger.info(f"Starting newsletter scheduler - weekly on {self.schedule_day} at {self.schedule_time}")
        
        # Clear any existing jobs
        schedule.clear()
        
        # Schedule weekly newsletter
        getattr(schedule.every(), self.schedule_day).at(self.schedule_time).do(
            self._send_weekly_newsletters
        )
        
        # Start scheduler in background thread
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("Newsletter scheduler started successfully")
    
    def stop_scheduler(self):
        """Stop the automated newsletter scheduler"""
        
        if not self.is_running:
            return
        
        self.is_running = False
        schedule.clear()
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        self.logger.info("Newsletter scheduler stopped")
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _send_weekly_newsletters(self):
        """Send weekly newsletters to all users"""
        
        self.logger.info("Starting weekly newsletter batch send")
        
        try:
            results = self.newsletter_scheduler.send_weekly_newsletters()
            
            self.logger.info(f"Weekly newsletter batch completed: "
                           f"{results['sent_count']} sent, "
                           f"{results['failed_count']} failed, "
                           f"{results['skipped_count']} skipped")
            
            # Log errors if any
            if results['errors']:
                for error in results['errors']:
                    self.logger.error(f"Newsletter error: {error}")
                    
        except Exception as e:
            self.logger.error(f"Weekly newsletter batch failed: {e}")
    
    def send_test_newsletters(self, force_send: bool = True) -> Dict[str, Any]:
        """Send test newsletters to all users (for testing)"""
        
        self.logger.info("Sending test newsletters to all users")
        return self.newsletter_scheduler.send_weekly_newsletters(force_send)
    
    def update_schedule(self, day: str, time: str):
        """Update the newsletter schedule"""
        
        valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        
        if day.lower() not in valid_days:
            raise ValueError(f"Invalid day: {day}. Must be one of {valid_days}")
        
        try:
            # Validate time format (HH:MM)
            datetime.strptime(time, "%H:%M")
        except ValueError:
            raise ValueError(f"Invalid time format: {time}. Must be HH:MM format (e.g., '09:00')")
        
        old_day, old_time = self.schedule_day, self.schedule_time
        self.schedule_day = day.lower()
        self.schedule_time = time
        
        # Restart scheduler if running
        if self.is_running:
            self.stop_scheduler()
            self.start_scheduler()
        
        self.logger.info(f"Newsletter schedule updated from {old_day} {old_time} to {day} {time}")
    
    def get_schedule_info(self) -> Dict[str, Any]:
        """Get current schedule information"""
        
        return {
            'is_running': self.is_running,
            'schedule_day': self.schedule_day,
            'schedule_time': self.schedule_time,
            'next_run': self._get_next_run_time()
        }
    
    def _get_next_run_time(self) -> Optional[str]:
        """Get the next scheduled run time"""
        
        if not self.is_running:
            return None
        
        jobs = schedule.jobs
        if not jobs:
            return None
        
        next_run = min(job.next_run for job in jobs if job.next_run)
        return next_run.isoformat() if next_run else None


# Global scheduler instance
_global_scheduler = None


def get_newsletter_scheduler() -> WeeklyNewsletterScheduler:
    """Get or create the global newsletter scheduler instance"""
    global _global_scheduler
    
    if _global_scheduler is None:
        _global_scheduler = WeeklyNewsletterScheduler()
    
    return _global_scheduler


def start_newsletter_automation():
    """Start automated newsletter system"""
    scheduler = get_newsletter_scheduler()
    scheduler.start_scheduler()
    return scheduler


def stop_newsletter_automation():
    """Stop automated newsletter system"""
    global _global_scheduler
    
    if _global_scheduler:
        _global_scheduler.stop_scheduler()


# CLI commands for manual operation
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python -m papersum.newsletter.scheduler [command]")
        print("Commands:")
        print("  start    - Start automated scheduler")
        print("  send     - Send newsletters now (force)")
        print("  test     - Send test newsletter")
        print("  status   - Show scheduler status")
        sys.exit(1)
    
    command = sys.argv[1]
    scheduler = get_newsletter_scheduler()
    
    if command == "start":
        print("Starting newsletter scheduler...")
        scheduler.start_scheduler()
        print(f"Scheduler started - weekly newsletters on {scheduler.schedule_day} at {scheduler.schedule_time}")
        
        # Keep running
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nStopping scheduler...")
            scheduler.stop_scheduler()
    
    elif command == "send":
        print("Sending newsletters now...")
        results = scheduler.send_test_newsletters(force_send=True)
        print(f"Results: {results}")
    
    elif command == "test":
        email = input("Enter email address for test: ")
        if email:
            service = EmailService()
            success = service.send_test_email(email)
            print(f"Test email {'sent' if success else 'failed'}")
    
    elif command == "status":
        info = scheduler.get_schedule_info()
        print(f"Scheduler status: {info}")
    
    else:
        print(f"Unknown command: {command}")