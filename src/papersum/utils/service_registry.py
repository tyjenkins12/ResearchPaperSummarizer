"""Singleton-style factories for heavyweight services."""

from __future__ import annotations

from functools import lru_cache

from ..config.settings import settings
from ..intelligence.preference_learner import PreferenceLearner
from ..models.hybrid import HybridSummarizer
from ..scraping.arxiv_monitor import ArxivMonitoringService


@lru_cache(maxsize=1)
def get_preference_learner() -> PreferenceLearner:
    """Return a cached instance of the preference learner."""

    return PreferenceLearner()


@lru_cache(maxsize=2)
def get_hybrid_summarizer(use_lightweight: bool | None = None) -> HybridSummarizer:
    """Return a cached hybrid summarizer."""

    if use_lightweight is None:
        use_lightweight = settings.feed.use_lightweight_models
    # lru_cache key includes flag, so identical configs reuse the same instance.
    return HybridSummarizer(use_lightweight=use_lightweight)


@lru_cache(maxsize=1)
def get_arxiv_service() -> ArxivMonitoringService:
    """Return a cached arXiv monitoring service."""

    return ArxivMonitoringService()

