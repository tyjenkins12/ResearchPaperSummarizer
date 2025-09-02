# Research Paper Summarizer (2025 Edition)

A modernized research paper summarization system that leverages state-of-the-art transformer models for both extractive and abstractive summarization of academic papers.

## About This Project

This project is an updated implementation inspired by the original [Research Paper Summarization](https://github.com/jananiarunachalam/Research-Paper-Summarization) repository by Janani Arunachalam. The purpose of this modernization is to address the limitations of the original 2019 implementation by incorporating recent advances in natural language processing and transformer models.

### Original Project Challenges
The original implementation faced several challenges:
- Grammatically coherent but sometimes irrelevant summaries
- Repetitive text generation in abstractive models
- Limited semantic understanding with TF-IDF based approaches

### Modern Solutions
This updated version addresses these issues using:
- **Transformer-based Models**: BART, T5, and Sentence-BERT for better semantic understanding
- **Hybrid Approach**: Combines extractive and abstractive methods for optimal results
- **Apple Silicon Optimization**: Efficient inference using MPS backend
- **Local Processing**: No external API dependencies - everything runs locally

## Features

### Multiple Summarization Methods
- **Extractive**: Identifies and extracts the most important sentences using semantic similarity
- **Abstractive**: Generates new summary text using local transformer models
- **Hybrid**: Intelligently combines both approaches for best results

### Advanced Text Processing
- PDF parsing with Grobid integration for structured text extraction
- Section-aware processing (abstract, introduction, methods, results, conclusion)
- Automatic strategy selection based on paper characteristics

### Modern Architecture
- FastAPI web interface for easy interaction
- Structured data models with Pydantic validation
- Configurable model selection and parameters
- Apple Silicon MPS acceleration support

## Requirements

- Python 3.10+
- Docker (for Grobid PDF processing)
- Apple Silicon Mac (for MPS optimization) or CUDA GPU (optional)

## Setup

1. **Start Grobid Server** (if not already running):
   ```bash
   docker run -p 8070:8070 lfoppiano/grobid:0.8.2
   ```

2. **Install Dependencies**:
   ```bash
   pip install -e .
   pip install -e ".[dev]"  # For development tools
   ```

3. **Download Required Models** (first run will download automatically):
   - Sentence-BERT: `all-MiniLM-L6-v2` (~90MB)
   - BART: `facebook/bart-large-cnn` (~1.6GB)
   - T5-small: `t5-small` (~240MB, lightweight option)

## Usage

### Command Line Interface
```bash
# Process a single PDF
parse-pdfs summarize paper.pdf

# Batch process multiple PDFs
parse-pdfs batch-summarize data/raw_pdfs/

# Use specific summarization method
parse-pdfs summarize paper.pdf --method hybrid --length 300
```

### Python API
```python
from papersum import HybridSummarizer, PDFExtractor

# Initialize components
extractor = PDFExtractor()
summarizer = HybridSummarizer(use_lightweight=True)  # For Apple Silicon

# Process a paper
paper = extractor.extract_paper("path/to/paper.pdf")
summary = summarizer.summarize_paper_full(paper, target_length=250)

print(f"Hybrid Summary: {summary.hybrid_summary}")
```

### Web Interface
```bash
# Start web server
uvicorn papersum.api.main:app --reload

# Access at http://localhost:8000
```

## Project Structure

```
src/papersum/
├── config/          # Configuration management
├── models/          # Summarization models (extractive, abstractive, hybrid)
├── parse/           # PDF processing and text extraction
├── api/             # FastAPI web interface
└── utils/           # Logging and utilities
```

## Performance Notes

- **Apple Silicon**: Uses MPS backend for GPU acceleration
- **Memory Usage**: Models require 2-4GB RAM depending on configuration
- **Speed**: ~10-30 seconds per paper depending on length and model choice
- **Local Processing**: No internet connection required after initial model download

## Model Options

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| T5-small | 240MB | Fast | Good | Quick summaries, Apple Silicon |
| BART-large-cnn | 1.6GB | Medium | Excellent | High-quality summaries |
| Sentence-BERT | 90MB | Very Fast | N/A | Extractive similarity |

## Credits

Original implementation by [Janani Arunachalam](https://github.com/jananiarunachalam/Research-Paper-Summarization).

This modernized version incorporates advances in transformer models and addresses the limitations identified in the original work through improved architectures and local model deployment.

## License

MIT License - See original repository for details.