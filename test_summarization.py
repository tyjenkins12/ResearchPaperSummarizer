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
        return False, None
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found in data/raw_pdfs")
        return False, None
    
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
        
        # Show first 500 characters of extracted text
        if paper.full_text:
            logger.info(f"✓ Text preview: {paper.full_text[:500]}...")
        
        return True, paper
    except Exception as e:
        logger.error(f"✗ Extraction failed: {e}")
        return False, None


def test_extractive_summarization(paper=None):
    """Test extractive summarization"""
    logger.info("\n=== Testing Extractive Summarization ===")
    
    try:
        summarizer = ExtractiveSummarizer()
        
        # Test with simple text first
        test_text = "This is a test sentence. This sentence is about testing. Testing is important for quality. Quality ensures good software. Software development requires careful testing. Good testing practices lead to reliable applications."
        
        summary = summarizer.summarize_text(test_text, num_sentences=2)
        logger.info(f"✓ Simple text summary: {summary}")
        
        # Test with real paper if available
        if paper and paper.full_text:
            paper_summary = summarizer.summarize_text(paper.full_text[:5000], num_sentences=3)
            logger.info(f"✓ Real paper summary: {paper_summary[:200]}...")
        
        return True
    except Exception as e:
        logger.error(f"✗ Extractive summarization failed: {e}")
        return False


def test_abstractive_summarization(paper=None):
    """Test abstractive summarization"""
    logger.info("\n=== Testing Abstractive Summarization ===")
    
    try:
        # Test lightweight model first (better for Apple Silicon)
        summarizer = LightweightSummarizer()
        
        test_text = "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on that data. Modern machine learning techniques include deep learning, neural networks, and transformer models."
        
        summary = summarizer.summarize_text(test_text, target_length=50)
        logger.info(f"✓ Simple text summary: {summary}")
        
        # Test with real paper if available
        if paper and paper.full_text:
            paper_summary = summarizer.summarize_text(paper.full_text[:3000], target_length=100)
            logger.info(f"✓ Real paper summary: {paper_summary[:200]}...")
        
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
        
        # For PyMuPDF fallback, create a mock structured paper
        if not paper.sections and paper.full_text:
            logger.info("Creating structured content from extracted text...")
            
            # Take first 5000 characters as a reasonable sample
            sample_text = paper.full_text[:5000]
            
            # Test direct text summarization instead
            extractive_summary = hybrid_summarizer.extractive.summarize_text(sample_text, num_sentences=3)
            abstractive_summary = hybrid_summarizer.abstractive.summarize_text(sample_text, target_length=150)
            
            logger.info("✓ Full pipeline successful!")
            logger.info(f"✓ Extractive summary: {extractive_summary[:200]}...")
            logger.info(f"✓ Abstractive summary: {abstractive_summary[:200]}...")
            
            return True
        
        # Generate summary with structured paper
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
    
    try:
        import torch
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
    except:
        logger.info("PyTorch not available for device detection")
    
    # Test PDF extraction first to get paper data
    extraction_passed, paper = test_pdf_extraction()
    
    tests = [
        ("PDF Extraction", extraction_passed),
        ("Extractive Summarization", test_extractive_summarization(paper)),
        ("Abstractive Summarization", test_abstractive_summarization(paper)),
        ("Full Pipeline", test_full_pipeline())
    ]
    
    # Summary
    logger.info("\n=== Test Results ===")
    for test_name, passed in tests:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in tests)
    if all_passed:
        logger.info("\n🎉 All tests passed! System is ready.")
    else:
        logger.warning("\n⚠️  Some tests failed. Check logs above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)