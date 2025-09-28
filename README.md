# Research Paper Summarizer (2025 Edition)

AI-powered research paper summarization with a personalized in-app research feed, daily TL;DR headlines, and automated preference learning.

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

### Personalized Research Feed
- **AI-Powered Paper Analysis**: Local processing with Apple Silicon optimization
- **Preference Learning**: Automatically learns your research interests from uploaded papers and in-app feedback
- **arXiv Integration**: Discovers relevant papers based on your preferences
- **Daily TL;DR Headlines**: Curated short summaries published directly in the app every morning
- **Weekly Highlight Report**: Every Monday, get deeper summaries of the most notable papers from the previous week
- **Feedback-Driven Experience**: Inline like/dislike controls and preference requests shape future recommendations

### Multiple Summarization Methods
- **Extractive**: Identifies and extracts the most important sentences using semantic similarity
- **Abstractive**: Generates new summary text using local transformer models
- **Hybrid**: Intelligently combines both approaches for best results and powers both daily TL;DR cards and weekly deep dives

### Advanced Text Processing
- PDF parsing with Grobid integration for structured text extraction
- Section-aware processing (abstract, introduction, methods, results, conclusion)
- Automatic strategy selection based on paper characteristics

### Modern Architecture
- FastAPI web interface that now serves a personalized research feed
- Docker deployment with production-ready setup
- Background jobs for daily headlines and weekly highlight reports
- Comprehensive admin dashboard

## Requirements

- Python 3.10+
- Docker (for Grobid PDF processing)
- Apple Silicon Mac (for MPS optimization) or CUDA GPU (optional)

## Quick Start

**For complete deployment instructions, see [docker/DEPLOYMENT.md](docker/DEPLOYMENT.md)**

### Development Setup

```bash
# Start development environment
docker/deploy.sh development
```

### Production Setup

```bash
# Configure environment
cp .env.example .env
# Edit .env with your email settings

# Deploy production environment
docker/deploy.sh production
```

The application will be available at:
- Main app: http://localhost:8000
- API docs: http://localhost:8000/docs  
- Admin dashboard: http://localhost:8000/admin

## Usage

### App Experience
- Open `http://localhost:8000` to see your personalized dashboard
- Review the **Daily TL;DR** feed for short-form headlines tuned to your interests
- Every Monday the **Weekly Highlights** tab shows deeper write-ups of the most noteworthy papers from the last 7 days
- Provide quick feedback with like/dislike buttons or add specific topic requests to refine future recommendations
- (Optional) Enable legacy newsletters if you still want email digests for archival purposes

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
# Start web server (from project root)
uvicorn --app-dir src papersum.web.main:app --reload

# Access the app-style dashboard at http://localhost:8000
```

### Feed Configuration
- Defaults favor lightweight summarization models and live arXiv discovery.
- To run entirely offline (reuse locally processed papers and cached models), set `PAPERSUM_FEED__OFFLINE_MODE=true` before launching the server.
- Adjust `PAPERSUM_FEED__USE_LIGHTWEIGHT_MODELS`, `PAPERSUM_FEED__MAX_DAILY_ITEMS`, and related env vars to tune generation behavior.

## Project Structure

```
├── src/papersum/          # Main application code
│   ├── config/            # Configuration management
│   ├── models/            # Summarization models (extractive, abstractive, hybrid)
│   ├── parse/             # PDF processing and text extraction
│   ├── web/               # FastAPI web interface
│   ├── feed/              # Daily TL;DR and weekly highlight generation (in progress)
│   ├── newsletter/        # Legacy newsletter and email system (optional)
│   ├── intelligence/      # Preference learning and recommendation
│   └── database/          # Database models and operations
├── tests/                 # Test files
├── docker/                # Docker deployment files
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── deploy.sh
│   └── DEPLOYMENT.md
├── data/                  # Data files and uploads
└── .env.example           # Environment configuration template
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
