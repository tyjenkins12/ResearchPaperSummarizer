from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import json

from ..database.models import User, Newsletter, NewsletterPaper, NewsletterTip, UserInterest as DBUserInterest
from ..database.session import get_db_session_sync
from ..intelligence.preference_learner import UserInterest, PreferenceLearner
from ..scraping.arxiv_monitor import ArxivMonitoringService
from ..newsletter.tip_generator import AutoTipGenerator
from ..newsletter.coding_tips import CodingTipsManager
from ..models.hybrid import HybridSummarizer
from ..parse.pdf_extractor import ExtractedPaper


@dataclass
class NewsletterContent:
    """Structure for newsletter content"""
    title: str
    papers: List[Dict]
    coding_tips: List[Dict]
    user_profile: Dict
    generated_at: datetime
    estimated_read_time: int


@dataclass
class NewsletterConfig:
    """Configuration for newsletter generation"""
    max_papers: int = 5
    max_tips: int = 3
    days_back: int = 7
    include_trending: bool = True
    summary_length: int = 150
    min_relevance_score: float = 0.3


class NewsletterGenerator:
    """Generates personalized newsletters with papers and coding tips"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.preference_learner = PreferenceLearner()
        self.arxiv_service = ArxivMonitoringService()
        self.tip_generator = AutoTipGenerator()
        self.tips_manager = CodingTipsManager()
        self.summarizer = HybridSummarizer(use_lightweight=True)


class NewsletterService:
    """High-level service for newsletter operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.generator = NewsletterGenerator()
    
    def generate_newsletter_preview(self, user_id: int = 1) -> Dict:
        """Generate a simple newsletter preview for demo"""
        
        try:
            # Mock newsletter content for demo
            papers = [
                {
                    'title': 'Attention Is All You Need: A Modern Perspective',
                    'authors': ['John Smith', 'Jane Doe'],
                    'summary': 'This paper revisits the transformer architecture with modern optimization techniques, showing significant improvements in training efficiency and model performance.',
                    'relevance_score': 0.85,
                    'why_relevant': 'Highly relevant to your research interests'
                }
            ]
            
            tips = [
                {
                    'title': 'Apple Silicon GPU Acceleration with PyTorch MPS',
                    'content': 'Apple Silicon Macs offer significant GPU acceleration through the Metal Performance Shaders (MPS) backend.',
                    'category': 'performance',
                    'difficulty': 'advanced'
                }
            ]
            
            return {
                'title': f'Your AI Research Digest - Week of {datetime.utcnow().strftime("%B %d, %Y")}',
                'papers': papers,
                'coding_tips': tips,
                'generated_at': datetime.utcnow().isoformat(),
                'estimated_read_time': 5
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate preview: {e}")
            return {'error': str(e)}