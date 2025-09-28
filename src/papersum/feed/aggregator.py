"""Service for generating personalized daily headlines and weekly reports."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy.orm import Session

from ..config.settings import FeedConfig, settings
from ..database.models import (
    DailyHeadline,
    Paper,
    User,
    UserInterest as DBUserInterest,
    WeeklyReport,
    WeeklyReportPaper,
)
from ..intelligence.preference_learner import PreferenceLearner, UserInterest
from ..models.hybrid import HybridSummarizer
from ..parse.pdf_extractor import ExtractedPaper
from ..scraping.arxiv_monitor import ArxivMonitoringService
from ..utils.service_registry import (
    get_arxiv_service,
    get_hybrid_summarizer,
    get_preference_learner,
)


@dataclass
class GeneratedHeadline:
    """Convenience structure describing a generated headline."""

    headline: DailyHeadline
    paper: Optional[Paper]


@dataclass
class GeneratedWeeklyReport:
    """Convenience structure describing a generated weekly report."""

    report: WeeklyReport
    papers: List[WeeklyReportPaper]


class FeedAggregator:
    """Creates daily TL;DR items and weekly highlight reports for users."""

    def __init__(
        self,
        arxiv_service: Optional[ArxivMonitoringService] = None,
        summarizer: Optional[HybridSummarizer] = None,
        preference_learner: Optional[PreferenceLearner] = None,
        feed_config: Optional[FeedConfig] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.feed_config = feed_config or settings.feed
        self.offline_mode = self.feed_config.offline_mode

        if arxiv_service is not None:
            self.arxiv_service = arxiv_service
        elif not self.offline_mode:
            self.arxiv_service = get_arxiv_service()
        else:
            self.arxiv_service = None

        self.summarizer = summarizer or get_hybrid_summarizer(
            self.feed_config.use_lightweight_models
        )
        self.preference_learner = preference_learner or get_preference_learner()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_daily_headlines(
        self,
        session: Session,
        user_id: int,
        headline_date: Optional[date] = None,
        max_items: Optional[int] = None,
        refresh: bool = False,
    ) -> List[GeneratedHeadline]:
        """Generate personalized TL;DR headlines for the given user.

        Args:
            session: Active SQLAlchemy session.
            user_id: ID of the user to generate content for.
            headline_date: Date to associate with the headlines (defaults to today).
            max_items: Number of items to return.
            refresh: If False and headlines already exist, the existing
                records are returned.
        """

        headline_date = headline_date or date.today()
        max_items = max_items or self.feed_config.max_daily_items
        user = self._get_user(session, user_id)

        if not refresh:
            existing = (
                session.query(DailyHeadline)
                .filter(
                    DailyHeadline.user_id == user.id,
                    DailyHeadline.headline_date == headline_date,
                )
                .order_by(DailyHeadline.rank.asc())
                .all()
            )
            if existing:
                self.logger.info(
                    "Returning cached headlines for user %s (%s)",
                    user.id,
                    headline_date,
                )
                return [GeneratedHeadline(headline=h, paper=h.paper) for h in existing]

        user_interests = self._build_user_interests(user.learned_interests)

        extracted_papers = self._discover_papers(
            session=session,
            user_interests=user_interests,
            max_papers=max_items * 2,
            days_back=self.feed_config.discovery_days_back,
        )

        if not extracted_papers:
            self.logger.warning("No papers discovered for user %s", user.id)
            return []

        sorted_papers = sorted(
            extracted_papers,
            key=lambda paper: paper.metadata.get("relevance_score", 0.0),
            reverse=True,
        )[:max_items]

        generated: List[GeneratedHeadline] = []
        for rank, extracted in enumerate(sorted_papers, start=1):
            summary = self.summarizer.summarize_paper_full(
                extracted,
                target_length=150,
            )

            paper_record = self._ensure_paper_record(session, extracted)

            headline = DailyHeadline(
                user_id=user.id,
                paper_id=paper_record.id if paper_record else None,
                headline_date=headline_date,
                title=extracted.title,
                summary=summary.hybrid_summary,
                rank=rank,
                extra_metadata=self._build_headline_metadata(extracted, summary),
            )

            session.add(headline)
            generated.append(GeneratedHeadline(headline=headline, paper=paper_record))

        session.commit()
        session.flush()

        return generated

    def generate_weekly_report(
        self,
        session: Session,
        user_id: int,
        week_start: Optional[date] = None,
        max_papers: Optional[int] = None,
        refresh: bool = False,
    ) -> Optional[GeneratedWeeklyReport]:
        """Generate a weekly highlight report for the given user."""

        user = self._get_user(session, user_id)
        week_start = week_start or self._default_week_start()
        week_end = week_start + timedelta(days=6)

        if not refresh:
            existing = (
                session.query(WeeklyReport)
                .filter(
                    WeeklyReport.user_id == user.id,
                    WeeklyReport.week_start == week_start,
                )
                .first()
            )
            if existing:
                self.logger.info(
                    "Returning cached weekly report for user %s (%s)",
                    user.id,
                    week_start,
                )
                return GeneratedWeeklyReport(
                    report=existing,
                    papers=list(existing.included_papers),
                )

        user_interests = self._build_user_interests(user.learned_interests)

        base_max = max_papers or self.feed_config.max_weekly_items

        extracted_papers = self._discover_papers(
            session=session,
            user_interests=user_interests,
            max_papers=base_max * 3,
            days_back=self.feed_config.weekly_days_back,
        )

        if not extracted_papers:
            self.logger.warning("No weekly papers discovered for user %s", user.id)
            return None

        sorted_papers = sorted(
            extracted_papers,
            key=lambda paper: paper.metadata.get("relevance_score", 0.0),
            reverse=True,
        )[:base_max]

        weekly_report = WeeklyReport(
            user_id=user.id,
            week_start=week_start,
            week_end=week_end,
            generated_at=datetime.utcnow(),
        )
        session.add(weekly_report)
        session.flush()  # ensure ID available for children

        combined_sections: List[str] = []
        report_entries: List[WeeklyReportPaper] = []

        for order, extracted in enumerate(sorted_papers, start=1):
            summary = self.summarizer.summarize_paper_full(
                extracted,
                target_length=240,
            )

            paper_record = self._ensure_paper_record(session, extracted)

            entry = WeeklyReportPaper(
                weekly_report_id=weekly_report.id,
                paper_id=paper_record.id if paper_record else None,
                display_order=order,
                headline=extracted.title,
                summary=summary.hybrid_summary,
                extra_metadata=self._build_weekly_metadata(extracted, summary),
            )

            session.add(entry)
            report_entries.append(entry)
            combined_sections.append(summary.hybrid_summary)

        weekly_report.summary = "\n\n".join(combined_sections)
        weekly_report.extra_metadata = {
            "paper_count": len(report_entries),
            "generated_at": datetime.utcnow().isoformat(),
        }

        session.commit()
        session.flush()

        weekly_report.included_papers[:] = report_entries
        return GeneratedWeeklyReport(report=weekly_report, papers=report_entries)

    def discover_candidates(
        self,
        session: Session,
        user_id: int,
        max_papers: Optional[int] = None,
        days_back: Optional[int] = None,
    ) -> List[ExtractedPaper]:
        """Surface candidate papers without writing feed records."""

        user = self._get_user(session, user_id)
        user_interests = self._build_user_interests(user.learned_interests)

        max_results = max_papers or self.feed_config.max_daily_items
        window = days_back or self.feed_config.discovery_days_back

        return self._discover_papers(
            session=session,
            user_interests=user_interests,
            max_papers=max_results,
            days_back=window,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_user(self, session: Session, user_id: int) -> User:
        user = session.get(User, user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        return user

    def _build_user_interests(
        self,
        db_interests: List[DBUserInterest],
    ) -> List[UserInterest]:
        interests: List[UserInterest] = []
        encoder = self.preference_learner.encoder
        embedding_dim = getattr(encoder, "get_sentence_embedding_dimension", lambda: 384)()

        for db_interest in db_interests:
            keywords = db_interest.keywords or []
            seed_text = " ".join(keywords) or db_interest.topic
            embedding = encoder.encode([seed_text])[0] if seed_text else np.zeros(embedding_dim, dtype=float)
            interests.append(
                UserInterest(
                    topic=db_interest.topic,
                    keywords=keywords,
                    embedding=embedding,
                    confidence_score=db_interest.confidence_score or 0.5,
                    paper_count=db_interest.paper_count or 0,
                    last_seen=db_interest.last_seen or datetime.utcnow(),
                )
            )
        return interests

    def _ensure_paper_record(
        self,
        session: Session,
        extracted_paper,
    ) -> Optional[Paper]:
        metadata = extracted_paper.metadata or {}
        arxiv_id = metadata.get("arxiv_id")

        paper: Optional[Paper] = None
        if arxiv_id:
            paper = session.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()

        if paper:
            # Update relevance score to keep ranking fresh
            paper.relevance_score = metadata.get("relevance_score", paper.relevance_score)
            return paper

        paper = Paper(
            title=extracted_paper.title,
            authors=extracted_paper.authors,
            abstract=extracted_paper.abstract,
            full_text=extracted_paper.full_text,
            arxiv_id=arxiv_id,
            pdf_url=metadata.get("pdf_url"),
            source=metadata.get("source") or ("arxiv" if arxiv_id else "local"),
            source_url=metadata.get("source_url"),
            relevance_score=metadata.get("relevance_score", 0.0),
            discovered_at=datetime.utcnow(),
            processed_at=datetime.utcnow(),
            is_processed=True,
        )

        session.add(paper)
        session.flush()
        return paper

    def _build_headline_metadata(self, extracted_paper, summary) -> Dict[str, object]:
        metadata = dict(extracted_paper.metadata or {})
        metadata.update(
            {
                "summary_strategy": summary.strategy_used,
                "extractive_preview": summary.extractive_summary,
                "abstractive_preview": summary.abstractive_summary,
            }
        )
        return metadata

    def _build_weekly_metadata(self, extracted_paper, summary) -> Dict[str, object]:
        metadata = dict(extracted_paper.metadata or {})
        metadata.update(
            {
                "summary_strategy": summary.strategy_used,
                "section_summaries": summary.section_summaries,
            }
        )
        return metadata

    def _default_week_start(self) -> date:
        today = date.today()
        # Week starts on Monday; generate for previous week by default.
        start_of_week = today - timedelta(days=today.weekday())
        previous_week_start = start_of_week - timedelta(days=7)
        return previous_week_start

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def _discover_papers(
        self,
        session: Session,
        user_interests: List[UserInterest],
        max_papers: int,
        days_back: int,
    ) -> List[ExtractedPaper]:
        """Discover candidate papers based on config and mode."""

        if self.offline_mode:
            return self._discover_local_papers(session, max_papers)

        if not self.arxiv_service:
            self.logger.warning("Arxiv service unavailable; returning no papers")
            return []

        return self.arxiv_service.discover_relevant_papers(
            user_interests=user_interests,
            max_papers=max_papers,
            days_back=days_back,
            include_trending=True,
        )

    def _discover_local_papers(
        self,
        session: Session,
        limit: int,
    ) -> List[ExtractedPaper]:
        """Fallback discovery that reuses locally stored papers."""

        papers = (
            session.query(Paper)
            .filter(Paper.is_processed.is_(True))
            .order_by(Paper.discovered_at.desc())
            .limit(min(limit, self.feed_config.local_fallback_limit))
            .all()
        )

        extracted: List[ExtractedPaper] = []
        for paper in papers:
            metadata = {
                "relevance_score": paper.relevance_score or 0.0,
                "arxiv_id": paper.arxiv_id,
                "pdf_url": paper.pdf_url or paper.source_url,
                "source": paper.source,
            }
            extracted.append(
                ExtractedPaper(
                    title=paper.title,
                    abstract=paper.abstract or "",
                    authors=paper.authors or [],
                    sections=[],
                    full_text=paper.full_text or paper.abstract or "",
                    metadata=metadata,
                )
            )

        if not extracted:
            self.logger.info(
                "Offline mode enabled but no local papers available; returning empty list"
            )

        return extracted
