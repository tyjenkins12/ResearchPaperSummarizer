#!/usr/bin/env python3
"""
Test script to verify summarization functionality with existing PDFs
"""
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from papersum.parse.pdf_extractor import PDFExtractor
from papersum.models.extractive import ExtractiveSummarizer
from papersum.models.abstractive import AbstractiveSummarizer, LightweightSummarizer
from papersum.models.hybrid import HybridSummarizer
from papersum.config.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_pdf_extraction():
    """Test PDF extraction with existing files"""
    logger.info("=== Testing PDF Extraction ===")
    
    extractor = PDFExtractor()
    pdf_dir = Path("data/raw_pdfs")
    
    if not pdf_dir.exists():
        logger.error("No PDF directory found at data/raw_pdfs")
        return False
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found in data/raw_pdfs")
        return False
    
    # Test with first PDF
    test_pdf = pdf_files[0]
    logger.info(f"Testing extraction with: {test_pdf.name}")
    
    try:
        paper = extractor.extract_paper(test_pdf)
        logger.info(f"✓ Extracted: {paper.title}")
        logger.info(f"✓ Authors: {len(paper.authors)} found")
        logger.info(f"✓ Sections: {len(paper.sections)} found")
        logger.info(f"✓ Abstract length: {len(paper.abstract)} chars")
        logger.info(f"✓ Full text length: {len(paper.full_text)} chars")
        return True
    except Exception as e:
        logger.error(f"✗ Extraction failed: {e}")
        return False


def test_extractive_summarization():
    """Test extractive summarization"""
    logger.info("\n=== Testing Extractive Summarization ===")
    
    try:
        summarizer = ExtractiveSummarizer()
        test_text = "This is a test sentence. This sentence is about testing. Testing is important for quality. Quality ensures good software."
        
        summary = summarizer.summarize_text(test_text, num_sentences=2)
        logger.info(f"✓ Extractive summary: {summary[:100]}...")
        return True
    except Exception as e:
        logger.error(f"✗ Extractive summarization failed: {e}")
        return False


def test_abstractive_summarization():
    """Test abstractive summarization"""
    logger.info("\n=== Testing Abstractive Summarization ===")
    
    try:
        # Test lightweight model first (better for Apple Silicon)
        summarizer = LightweightSummarizer()
        test_text = "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on that data."
        
        summary = summarizer.summarize_text(test_text, target_length=50)
        logger.info(f"✓ Abstractive summary: {summary}")
        return True
    except Exception as e:
        logger.error(f"✗ Abstractive summarization failed: {e}")
        logger.info("This might be due to model download - try running again")
        return False


def test_full_pipeline():
    """Test complete pipeline with real PDF"""
    logger.info("\n=== Testing Full Pipeline ===")
    
    pdf_dir = Path("data/raw_pdfs")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.error("No PDFs available for testing")
        return False
    
    try:
        # Initialize components
        extractor = PDFExtractor()
        hybrid_summarizer = HybridSummarizer(use_lightweight=True)
        
        # Process first PDF
        test_pdf = pdf_files[0]
        logger.info(f"Processing: {test_pdf.name}")
        
        # Extract paper
        paper = extractor.extract_paper(test_pdf)
        
        # Generate summary
        result = hybrid_summarizer.summarize_paper_full(paper, target_length=150)
        
        logger.info("✓ Full pipeline successful!")
        logger.info(f"Strategy used: {result.strategy_used}")
        logger.info(f"Hybrid summary: {result.hybrid_summary[:200]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Full pipeline failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("Starting summarization system tests...")
    logger.info(f"Using device: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")
    
    tests = [
        ("PDF Extraction", test_pdf_extraction),
        ("Extractive Summarization", test_extractive_summarization),
        ("Abstractive Summarization", test_abstractive_summarization),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n=== Test Results ===")
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("\n🎉 All tests passed! System is ready.")
    else:
        logger.warning("\n⚠️  Some tests failed. Check logs above.")
    
    return all_passed


if __name__ == "__main__":
    import torch
    success = main()
    sys.exit(0 if success else 1)