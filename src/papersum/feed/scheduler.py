"""Background scheduler for generating daily and weekly feed content."""

from __future__ import annotations

import logging
import threading
import time
from datetime import date
from typing import Optional

import schedule
from sqlalchemy.orm import Session

from ..database.models import User
from ..database.session import get_db_session_sync
from .aggregator import FeedAggregator


class FeedScheduler:
    """Schedules daily headline and weekly highlight generation."""

    def __init__(self, aggregator: Optional[FeedAggregator] = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.aggregator = aggregator or FeedAggregator()
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None

        # Defaults: refresh headlines every morning, highlights on Monday
        self.daily_time = "08:00"
        self.weekly_day = "monday"
        self.weekly_time = "09:00"

    # ------------------------------------------------------------------
    # Control methods
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self.is_running:
            self.logger.warning("Feed scheduler already running")
            return

        schedule.clear('feed')
        schedule.every().day.at(self.daily_time).do(self._run_daily_job).tag('feed')
        getattr(schedule.every(), self.weekly_day).at(self.weekly_time).do(self._run_weekly_job).tag('feed')

        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._loop, daemon=True)
        self.scheduler_thread.start()
        self.logger.info(
            "Feed scheduler started (daily %s, weekly %s at %s)",
            self.daily_time,
            self.weekly_day,
            self.weekly_time,
        )

    def stop(self) -> None:
        if not self.is_running:
            return

        self.is_running = False
        schedule.clear('feed')
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.logger.info("Feed scheduler stopped")

    def update_schedule(self, daily_time: str, weekly_day: str, weekly_time: str) -> None:
        self.daily_time = daily_time
        self.weekly_day = weekly_day.lower()
        self.weekly_time = weekly_time

        if self.is_running:
            self.stop()
            self.start()

    def status(self) -> dict:
        return {
            "is_running": self.is_running,
            "daily_time": self.daily_time,
            "weekly_day": self.weekly_day,
            "weekly_time": self.weekly_time,
        }

    # ------------------------------------------------------------------
    # Immediate triggers
    # ------------------------------------------------------------------

    def run_daily_now(self, target_date: Optional[date] = None, refresh: bool = True) -> dict:
        return self._generate_headlines_for_all(target_date or date.today(), refresh)

    def run_weekly_now(self, week_start: Optional[date] = None, refresh: bool = True) -> dict:
        return self._generate_weekly_for_all(week_start, refresh)

    # ------------------------------------------------------------------
    # Internal scheduling helpers
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while self.is_running:
            try:
                schedule.run_pending()
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.error(f"Feed scheduler error: {exc}")
            time.sleep(30)

    def _run_daily_job(self) -> None:
        self.logger.info("Running scheduled daily feed generation")
        self._generate_headlines_for_all(date.today(), refresh=True)

    def _run_weekly_job(self) -> None:
        self.logger.info("Running scheduled weekly highlights generation")
        self._generate_weekly_for_all(None, refresh=True)

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def _generate_headlines_for_all(self, target_date: date, refresh: bool) -> dict:
        results = {"generated": 0, "errors": []}
        with get_db_session_sync() as session:
            users = session.query(User).all()
            for user in users:
                try:
                    self.aggregator.generate_daily_headlines(
                        session,
                        user.id,
                        headline_date=target_date,
                        refresh=refresh,
                    )
                    results["generated"] += 1
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.error(f"Daily feed failed for user {user.id}: {exc}")
                    results["errors"].append({"user_id": user.id, "error": str(exc)})
        return results

    def _generate_weekly_for_all(self, week_start: Optional[date], refresh: bool) -> dict:
        results = {"generated": 0, "errors": []}
        with get_db_session_sync() as session:
            users = session.query(User).all()
            for user in users:
                try:
                    self.aggregator.generate_weekly_report(
                        session,
                        user.id,
                        week_start=week_start,
                        refresh=refresh,
                    )
                    results["generated"] += 1
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.error(f"Weekly feed failed for user {user.id}: {exc}")
                    results["errors"].append({"user_id": user.id, "error": str(exc)})
        return results


_global_feed_scheduler: Optional[FeedScheduler] = None


def get_feed_scheduler() -> FeedScheduler:
    global _global_feed_scheduler
    if _global_feed_scheduler is None:
        _global_feed_scheduler = FeedScheduler()
    return _global_feed_scheduler
