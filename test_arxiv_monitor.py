#!/usr/bin/env python3
"""
Test script for arXiv monitoring and paper discovery system
"""
import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from papersum.scraping.arxiv_monitor import (
    ArxivAPI, 
    PaperMonitor, 
    ArxivMonitoringService,
    PaperRelevanceScorer
)
from papersum.intelligence.preference_learner import UserInterest
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_arxiv_api():
    """Test basic arXiv API functionality"""
    logger.info("=== Testing arXiv API ===")
    
    try:
        api = ArxivAPI()
        
        # Test 1: Get recent ML papers
        recent_papers = api.get_recent_papers(days_back=30, max_results=5)
        logger.info(f"✓ Found {len(recent_papers)} recent papers")
        
        if recent_papers:
            sample_paper = recent_papers[0]
            logger.info(f"✓ Sample paper: {sample_paper.title}")
            logger.info(f"✓ Categories: {sample_paper.categories}")
            logger.info(f"✓ Published: {sample_paper.published}")
        
        # Test 2: Keyword search
        keyword_papers = api.search_by_keywords(
            keywords=["transformer", "attention"], 
            max_results=3,
            days_back=30
        )
        logger.info(f"✓ Found {len(keyword_papers)} papers for 'transformer' and 'attention'")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ arXiv API test failed: {e}")
        return False


def test_paper_relevance_scoring():
    """Test paper relevance scoring system"""
    logger.info("\n=== Testing Paper Relevance Scoring ===")
    
    try:
        scorer = PaperRelevanceScorer()
        
        # Create mock user interests
        mock_interests = [
            UserInterest(
                topic="natural_language_processing: transformer, attention, bert",
                keywords=["transformer", "attention", "bert", "language model"],
                embedding=np.random.rand(384),  # Mock embedding
                confidence_score=0.8,
                paper_count=5,
                last_seen=datetime.utcnow()
            ),
            UserInterest(
                topic="computer_vision: cnn, detection, segmentation", 
                keywords=["cnn", "detection", "segmentation", "vision"],
                embedding=np.random.rand(384),
                confidence_score=0.6,
                paper_count=3,
                last_seen=datetime.utcnow() - timedelta(days=10)
            )
        ]
        
        # Get some real papers to score
        api = ArxivAPI()
        papers = api.get_recent_papers(days_back=30, max_results=5)
        
        if papers:
            scored_papers = scorer.score_papers(papers, mock_interests)
            logger.info(f"✓ Scored {len(scored_papers)} papers")
            
            # Show top scoring paper
            if scored_papers:
                top_paper = scored_papers[0]
                logger.info(f"✓ Top paper: {top_paper.title}")
                logger.info(f"✓ Relevance score: {top_paper.relevance_score:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Relevance scoring test failed: {e}")
        return False


def test_full_monitoring_service():
    """Test complete monitoring service"""
    logger.info("\n=== Testing Full Monitoring Service ===")
    
    try:
        service = ArxivMonitoringService()
        
        # Create mock user interests
        mock_interests = [
            UserInterest(
                topic="machine_learning: deep learning, neural networks",
                keywords=["deep learning", "neural networks", "optimization"],
                embedding=np.random.rand(384),
                confidence_score=0.9,
                paper_count=8,
                last_seen=datetime.utcnow()
            )
        ]
        
        # Test paper discovery
        discovered_papers = service.discover_relevant_papers(
            user_interests=mock_interests,
            max_papers=3,
            days_back=30,
            include_trending=True
        )
        
        logger.info(f"✓ Discovered {len(discovered_papers)} relevant papers")
        
        if discovered_papers:
            for i, paper in enumerate(discovered_papers, 1):
                logger.info(f"  {i}. {paper.title}")
                logger.info(f"     Relevance from metadata: {paper.metadata.get('relevance_score', 'N/A')}")
        
        # Test weekly batch
        weekly_papers = service.get_weekly_paper_batch(
            user_interests=mock_interests,
            target_count=2
        )
        logger.info(f"✓ Weekly batch contains {len(weekly_papers)} papers")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Monitoring service test failed: {e}")
        return False


def test_paper_conversion():
    """Test arXiv to ExtractedPaper conversion"""
    logger.info("\n=== Testing Paper Conversion ===")
    
    try:
        # Get a sample arXiv paper
        api = ArxivAPI()
        papers = api.get_recent_papers(days_back=30, max_results=1)
        
        if not papers:
            logger.warning("No papers found for conversion test")
            return False
        
        arxiv_paper = papers[0]
        logger.info(f"✓ Got arXiv paper: {arxiv_paper.title}")
        
        # Convert to ExtractedPaper
        from papersum.scraping.arxiv_monitor import ArxivToPaperConverter
        converter = ArxivToPaperConverter()
        extracted_paper = converter.convert_arxiv_paper(arxiv_paper)
        
        logger.info(f"✓ Converted to ExtractedPaper")
        logger.info(f"  Title: {extracted_paper.title}")
        logger.info(f"  Authors: {len(extracted_paper.authors)}")
        logger.info(f"  Sections: {len(extracted_paper.sections)}")
        logger.info(f"  Metadata keys: {list(extracted_paper.metadata.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Paper conversion test failed: {e}")
        return False


def main():
    """Run all arXiv monitoring tests"""
    logger.info("Starting arXiv monitoring system tests...")
    
    tests = [
        ("arXiv API", test_arxiv_api),
        ("Paper Relevance Scoring", test_paper_relevance_scoring), 
        ("Full Monitoring Service", test_full_monitoring_service),
        ("Paper Conversion", test_paper_conversion)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n=== Test Results ===")
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        logger.info("\n🎉 All arXiv monitoring tests passed!")
    else:
        logger.warning("\n⚠️  Some tests failed. Check logs above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)