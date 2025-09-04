from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from ..database.models import User, Paper, UserInterest as DBUserInterest
from ..database.session import get_db_session
from ..parse.pdf_extractor import PDFExtractor
from ..intelligence.preference_learner import PreferenceLearner, UserInterest
from ..scraping.arxiv_monitor import ArxivMonitoringService
from ..newsletter.tip_generator import AutoTipGenerator
from ..newsletter.coding_tips import CodingTipsManager
from .schemas import (
    UserCreate, UserResponse, PaperResponse, 
    InterestResponse, FeedbackRequest, PreferencesResponse
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Research Paper Summarizer",
    description="AI-powered research paper summarization with personalized newsletters",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
static_dir = Path(__file__).parent / "static"
template_dir = Path(__file__).parent / "templates"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates = None
if template_dir.exists() and (template_dir / "index.html").exists():
    templates = Jinja2Templates(directory=str(template_dir))

# Global services (in production, use dependency injection)
pdf_extractor = PDFExtractor()
preference_learner = PreferenceLearner()
arxiv_service = ArxivMonitoringService()
tip_generator = AutoTipGenerator()


# Dependency to get current user (simplified - no auth for now)
async def get_current_user(db = Depends(get_db_session)) -> User:
    """Get current user (simplified - creates default user if none exists)"""
    user = db.query(User).first()
    if not user:
        user = User(
            email="user@example.com",
            name="Default User",
            created_at=datetime.utcnow()
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        return HTMLResponse("""
        <html>
        <head>
            <title>Research Paper Summarizer</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .nav { background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .nav a { margin-right: 15px; text-decoration: none; color: #007bff; }
                .nav a:hover { text-decoration: underline; }
                .feature { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>🔬 Research Paper Summarizer</h1>
            <div class="nav">
                <a href="/upload">📄 Upload Papers</a>
                <a href="/preferences">⚙️ My Preferences</a>
                <a href="/discover">🔍 Discover Papers</a>
                <a href="/api/tips">💡 Get Tips</a>
                <a href="/docs">📚 API Docs</a>
            </div>
            
            <div class="feature">
                <h3>📤 Upload Research Papers</h3>
                <p>Upload PDF papers to learn your research interests and get personalized recommendations.</p>
                <a href="/upload">Start Uploading →</a>
            </div>
            
            <div class="feature">
                <h3>🤖 AI-Powered Discovery</h3>
                <p>Automatically discover relevant papers from arXiv based on your learned preferences.</p>
                <a href="/discover">Discover Papers →</a>
            </div>
            
            <div class="feature">
                <h3>📊 Personal Dashboard</h3>
                <p>View your learned interests, uploaded papers, and preference evolution.</p>
                <a href="/preferences">View Dashboard →</a>
            </div>
            
            <div class="feature">
                <h3>🔧 Coding Tips</h3>
                <p>Get personalized coding tips based on your research interests and trending topics.</p>
                <a href="/api/tips">Get Tips →</a>
            </div>
            
            <p style="margin-top: 30px; color: #666;">
                <strong>Features:</strong> Local AI processing, Apple Silicon optimization, 
                automatic preference learning, arXiv integration, personalized newsletters
            </p>
        </body>
        </html>
        """)


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Paper upload page"""
    if templates:
        return templates.TemplateResponse("upload.html", {"request": request})
    else:
        return HTMLResponse("""
        <html>
        <head>
            <title>Upload Papers - Research Paper Summarizer</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
                .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; }
                .upload-area:hover { border-color: #007bff; background: #f8f9fa; }
                input[type="file"] { margin: 10px 0; }
                button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
                button:hover { background: #0056b3; }
                .nav a { margin-right: 15px; text-decoration: none; color: #007bff; }
            </style>
        </head>
        <body>
            <div class="nav">
                <a href="/">🏠 Home</a>
                <a href="/preferences">⚙️ Preferences</a>
                <a href="/discover">🔍 Discover</a>
            </div>
            
            <h1>📄 Upload Research Papers</h1>
            <p>Upload PDF files to learn your research interests. The system will analyze your papers to understand your preferences and recommend similar research.</p>
            
            <form action="/upload" method="post" enctype="multipart/form-data">
                <div class="upload-area">
                    <h3>📂 Select PDF Files</h3>
                    <input type="file" name="files" multiple accept=".pdf" required>
                    <p style="color: #666; font-size: 14px;">You can select multiple PDF files at once</p>
                </div>
                <button type="submit">🚀 Process Papers</button>
            </form>
            
            <div style="margin-top: 30px; padding: 15px; background: #f8f9fa; border-radius: 5px;">
                <h4>What happens next?</h4>
                <ol>
                    <li>Papers are processed to extract text and metadata</li>
                    <li>Your research interests are automatically learned</li>
                    <li>Similar papers will be recommended in the future</li>
                    <li>Personalized coding tips will be generated</li>
                </ol>
            </div>
        </body>
        </html>
        """)


@app.post("/upload")
async def upload_papers(
    files: List[UploadFile] = File(...),
    user: User = Depends(get_current_user),
    db = Depends(get_db_session)
):
    """Upload and process research papers"""
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    processed_papers = []
    errors = []
    
    for file in files:
        if not file.filename.endswith('.pdf'):
            errors.append(f"{file.filename}: Only PDF files are supported")
            continue
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                tmp_path = Path(tmp_file.name)
            
            # Extract paper content
            extracted_paper = pdf_extractor.extract_paper(tmp_path)
            
            # Save to database
            db_paper = Paper(
                title=extracted_paper.title,
                authors=extracted_paper.authors,
                abstract=extracted_paper.abstract,
                full_text=extracted_paper.full_text,
                metadata=extracted_paper.metadata,
                uploaded_by_user_id=user.id,
                discovered_at=datetime.utcnow(),
                processed_at=datetime.utcnow()
            )
            
            db.add(db_paper)
            db.commit()
            
            processed_papers.append({
                "filename": file.filename,
                "title": extracted_paper.title,
                "authors": extracted_paper.authors,
                "status": "success"
            })
            
            # Clean up temp file
            tmp_path.unlink()
            
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")
            logger.error(f"Failed to process {file.filename}: {e}")
    
    # Update user preferences based on uploaded papers
    if processed_papers:
        await update_user_preferences(user.id, db)
    
    return {
        "processed_papers": processed_papers,
        "errors": errors,
        "success_count": len(processed_papers),
        "error_count": len(errors)
    }


async def update_user_preferences(user_id: int, db):
    """Update user preferences based on uploaded papers"""
    
    try:
        # Get user's uploaded papers
        user_papers = db.query(Paper).filter(Paper.uploaded_by_user_id == user_id).all()
        
        if len(user_papers) >= 2:  # Need at least 2 papers for meaningful learning
            # Convert to ExtractedPaper format
            from ..parse.pdf_extractor import ExtractedPaper
            extracted_papers = []
            
            for paper in user_papers:
                extracted_paper = ExtractedPaper(
                    title=paper.title,
                    authors=paper.authors or [],
                    abstract=paper.abstract or "",
                    sections=[],
                    full_text=paper.full_text or "",
                    metadata=paper.metadata or {}
                )
                extracted_papers.append(extracted_paper)
            
            # Learn preferences
            learned_interests = preference_learner.learn_from_papers(extracted_papers)
            
            # Save interests to database (simplified - replace existing)
            db.query(DBUserInterest).filter(DBUserInterest.user_id == user_id).delete()
            
            for interest in learned_interests:
                db_interest = DBUserInterest(
                    user_id=user_id,
                    topic=interest.topic,
                    keywords=interest.keywords,
                    confidence_score=interest.confidence_score,
                    paper_count=interest.paper_count,
                    last_seen=interest.last_seen
                )
                db.add(db_interest)
            
            db.commit()
            logger.info(f"Updated preferences for user {user_id}: {len(learned_interests)} interests")
            
    except Exception as e:
        logger.error(f"Failed to update user preferences: {e}")


@app.get("/preferences", response_class=HTMLResponse)
async def preferences_page(
    request: Request,
    user: User = Depends(get_current_user),
    db = Depends(get_db_session)
):
    """User preferences page"""
    
    # Get user's current interests
    interests = db.query(DBUserInterest).filter(DBUserInterest.user_id == user.id).all()
    
    # Get user's papers
    papers = db.query(Paper).filter(Paper.uploaded_by_user_id == user.id).order_by(Paper.discovered_at.desc()).limit(10).all()
    
    context = {
        "request": request,
        "user": user,
        "interests": interests,
        "papers": papers,
        "paper_count": len(papers)
    }
    
    if templates:
        return templates.TemplateResponse("preferences.html", context)
    else:
        # Simple HTML fallback
        interests_html = ""
        for interest in interests:
            interests_html += f"""
            <div style="border: 1px solid #ccc; margin: 10px; padding: 10px;">
                <h3>{interest.topic}</h3>
                <p>Confidence: {interest.confidence_score:.2f}</p>
                <p>Keywords: {', '.join(interest.keywords)}</p>
                <p>Papers: {interest.paper_count}</p>
            </div>
            """
        
        papers_html = ""
        for paper in papers:
            papers_html += f"""
            <div style="border: 1px solid #eee; margin: 5px; padding: 8px;">
                <h4>{paper.title}</h4>
                <p>Authors: {', '.join(paper.authors) if paper.authors else 'Unknown'}</p>
                <p>Uploaded: {paper.discovered_at.strftime('%Y-%m-%d')}</p>
            </div>
            """
        
        return HTMLResponse(f"""
        <html>
        <head><title>Your Preferences</title></head>
        <body>
            <h1>Your Research Preferences</h1>
            
            <h2>Learned Interests ({len(interests)})</h2>
            {interests_html or '<p>No interests learned yet. Upload more papers!</p>'}
            
            <h2>Uploaded Papers ({len(papers)})</h2>
            {papers_html or '<p>No papers uploaded yet.</p>'}
            
            <br>
            <a href="/upload">Upload More Papers</a> | 
            <a href="/discover">Discover New Papers</a> | 
            <a href="/">Home</a>
        </body>
        </html>
        """)


@app.get("/discover", response_class=HTMLResponse)
async def discover_page(
    request: Request,
    user: User = Depends(get_current_user),
    db = Depends(get_db_session)
):
    """Paper discovery page"""
    
    # Get user interests
    db_interests = db.query(DBUserInterest).filter(DBUserInterest.user_id == user.id).all()
    
    # Convert to UserInterest format
    user_interests = []
    for db_interest in db_interests:
        user_interest = UserInterest(
            topic=db_interest.topic,
            keywords=db_interest.keywords,
            embedding=None,  # Will be regenerated if needed
            confidence_score=db_interest.confidence_score,
            paper_count=db_interest.paper_count,
            last_seen=db_interest.last_seen
        )
        user_interests.append(user_interest)
    
    # Discover relevant papers
    try:
        discovered_papers = arxiv_service.discover_relevant_papers(
            user_interests=user_interests,
            max_papers=10,
            days_back=14,
            include_trending=True
        )
        
        papers_html = ""
        for paper in discovered_papers:
            relevance = paper.metadata.get('relevance_score', 0)
            papers_html += f"""
            <div style="border: 1px solid #ddd; margin: 10px; padding: 15px; border-radius: 5px;">
                <h3>{paper.title}</h3>
                <p><strong>Authors:</strong> {', '.join(paper.authors)}</p>
                <p><strong>Abstract:</strong> {paper.abstract[:300]}...</p>
                <p><strong>Relevance Score:</strong> {relevance:.3f}</p>
                <p><strong>ArXiv ID:</strong> {paper.metadata.get('arxiv_id', 'N/A')}</p>
                <a href="{paper.metadata.get('pdf_url', '#')}" target="_blank">View PDF</a>
                
                <form style="margin-top: 10px; display: inline;" action="/feedback" method="post">
                    <input type="hidden" name="paper_id" value="{paper.metadata.get('arxiv_id', '')}">
                    <label>Rate this paper:</label>
                    <select name="rating">
                        <option value="1">1 - Not relevant</option>
                        <option value="2">2 - Slightly relevant</option>
                        <option value="3" selected>3 - Moderately relevant</option>
                        <option value="4">4 - Very relevant</option>
                        <option value="5">5 - Extremely relevant</option>
                    </select>
                    <button type="submit">Submit Rating</button>
                </form>
            </div>
            """
        
        context = {
            "request": request,
            "user": user,
            "discovered_papers": discovered_papers,
            "papers_html": papers_html,
            "interest_count": len(user_interests)
        }
        
    except Exception as e:
        logger.error(f"Failed to discover papers: {e}")
        context = {
            "request": request,
            "user": user,
            "error": str(e),
            "papers_html": f"<p>Error discovering papers: {e}</p>",
            "interest_count": len(user_interests)
        }
    
    if templates:
        return templates.TemplateResponse("discover.html", context)
    else:
        return HTMLResponse(f"""
        <html>
        <head><title>Discover Papers</title></head>
        <body>
            <h1>Discover Relevant Papers</h1>
            <p>Based on your {context['interest_count']} learned interests</p>
            
            {context.get('papers_html', '<p>No papers found</p>')}
            
            <br>
            <a href="/preferences">View Preferences</a> | 
            <a href="/upload">Upload Papers</a> | 
            <a href="/">Home</a>
        </body>
        </html>
        """)


@app.post("/feedback")
async def submit_feedback(
    paper_id: str = Form(...),
    rating: int = Form(...),
    user: User = Depends(get_current_user),
    db = Depends(get_db_session)
):
    """Submit feedback on recommended papers"""
    
    if not (1 <= rating <= 5):
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    
    try:
        # Get user's current interests
        db_interests = db.query(DBUserInterest).filter(DBUserInterest.user_id == user.id).all()
        
        # Convert to UserInterest format for preference learner
        user_interests = []
        for db_interest in db_interests:
            user_interest = UserInterest(
                topic=db_interest.topic,
                keywords=db_interest.keywords,
                embedding=None,
                confidence_score=db_interest.confidence_score,
                paper_count=db_interest.paper_count,
                last_seen=db_interest.last_seen
            )
            user_interests.append(user_interest)
        
        # Create feedback data (simplified - you'd normally fetch the actual paper)
        paper_ratings = {paper_id: rating}
        
        # Mock paper data for feedback (in real system, fetch from arXiv or cache)
        mock_paper = ExtractedPaper(
            title=f"Paper {paper_id}",
            authors=[],
            abstract="Mock paper for feedback",
            sections=[],
            full_text="",
            metadata={'arxiv_id': paper_id}
        )
        papers_db = {paper_id: mock_paper}
        
        # Update interests based on feedback
        updated_interests = preference_learner.update_interests_from_feedback(
            interests=user_interests,
            paper_ratings=paper_ratings,
            papers_db=papers_db
        )
        
        # Save updated interests back to database
        for i, updated_interest in enumerate(updated_interests):
            db_interest = db_interests[i]
            db_interest.confidence_score = updated_interest.confidence_score
            db_interest.last_seen = updated_interest.last_seen
        
        db.commit()
        
        logger.info(f"Updated preferences for user {user.id} based on rating {rating} for paper {paper_id}")
        
        return RedirectResponse(url="/discover", status_code=303)
        
    except Exception as e:
        logger.error(f"Failed to process feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to process feedback")


@app.get("/api/interests", response_model=List[InterestResponse])
async def get_user_interests(
    user: User = Depends(get_current_user),
    db = Depends(get_db_session)
):
    """Get user's learned interests via API"""
    
    interests = db.query(DBUserInterest).filter(DBUserInterest.user_id == user.id).all()
    
    return [
        InterestResponse(
            id=interest.id,
            topic=interest.topic,
            keywords=interest.keywords,
            confidence_score=interest.confidence_score,
            paper_count=interest.paper_count,
            last_seen=interest.last_seen
        )
        for interest in interests
    ]


@app.get("/api/papers", response_model=List[PaperResponse])
async def get_user_papers(
    limit: int = 20,
    user: User = Depends(get_current_user),
    db = Depends(get_db_session)
):
    """Get user's uploaded papers via API"""
    
    papers = db.query(Paper).filter(
        Paper.uploaded_by_user_id == user.id
    ).order_by(Paper.discovered_at.desc()).limit(limit).all()
    
    return [
        PaperResponse(
            id=paper.id,
            title=paper.title,
            authors=paper.authors or [],
            abstract=paper.abstract,
            relevance_score=paper.relevance_score,
            discovered_at=paper.discovered_at,
            metadata=paper.metadata or {}
        )
        for paper in papers
    ]


@app.post("/api/discover")
async def discover_papers_api(
    max_papers: int = 10,
    days_back: int = 7,
    include_trending: bool = True,
    user: User = Depends(get_current_user),
    db = Depends(get_db_session)
) -> Dict[str, Any]:
    """Discover relevant papers via API"""
    
    try:
        # Get user interests
        db_interests = db.query(DBUserInterest).filter(DBUserInterest.user_id == user.id).all()
        
        user_interests = []
        for db_interest in db_interests:
            user_interest = UserInterest(
                topic=db_interest.topic,
                keywords=db_interest.keywords,
                embedding=None,
                confidence_score=db_interest.confidence_score,
                paper_count=db_interest.paper_count,
                last_seen=db_interest.last_seen
            )
            user_interests.append(user_interest)
        
        # Discover papers
        discovered_papers = arxiv_service.discover_relevant_papers(
            user_interests=user_interests,
            max_papers=max_papers,
            days_back=days_back,
            include_trending=include_trending
        )
        
        return {
            "discovered_papers": len(discovered_papers),
            "papers": [
                {
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract,
                    "relevance_score": paper.metadata.get('relevance_score', 0),
                    "arxiv_id": paper.metadata.get('arxiv_id'),
                    "pdf_url": paper.metadata.get('pdf_url')
                }
                for paper in discovered_papers
            ],
            "user_interests": len(user_interests)
        }
        
    except Exception as e:
        logger.error(f"Paper discovery API failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tips")
async def get_coding_tips(
    count: int = 3,
    user: User = Depends(get_current_user),
    db = Depends(get_db_session)
):
    """Get personalized coding tips"""
    
    try:
        # Get user interests
        db_interests = db.query(DBUserInterest).filter(DBUserInterest.user_id == user.id).all()
        
        user_interests = []
        for db_interest in db_interests:
            user_interest = UserInterest(
                topic=db_interest.topic,
                keywords=db_interest.keywords,
                embedding=None,
                confidence_score=db_interest.confidence_score,
                paper_count=db_interest.paper_count,
                last_seen=db_interest.last_seen
            )
            user_interests.append(user_interest)
        
        # Get recent papers for tip generation
        recent_papers = db.query(Paper).filter(
            Paper.uploaded_by_user_id == user.id
        ).order_by(Paper.discovered_at.desc()).limit(5).all()
        
        extracted_papers = []
        for paper in recent_papers:
            extracted_paper = ExtractedPaper(
                title=paper.title,
                authors=paper.authors or [],
                abstract=paper.abstract or "",
                sections=[],
                full_text=paper.full_text or "",
                metadata=paper.metadata or {}
            )
            extracted_papers.append(extracted_paper)
        
        # Get trending topics
        tips_manager = CodingTipsManager()
        trending_topics = tips_manager.get_trending_topics()
        
        # Generate personalized tips
        personalized_tips = tip_generator.generate_personalized_tips(
            user_interests=user_interests,
            recent_papers=extracted_papers,
            trending_topics=trending_topics,
            total_tips=count
        )
        
        return {
            "tips": [
                {
                    "title": tip.title,
                    "content": tip.content,
                    "category": tip.category.value,
                    "difficulty": tip.difficulty.value,
                    "code_example": tip.code_example,
                    "tags": tip.tags or []
                }
                for tip in personalized_tips
            ]
        }
        
    except Exception as e:
        logger.error(f"Coding tips API failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/profile")
async def get_user_profile(
    user: User = Depends(get_current_user),
    db = Depends(get_db_session)
):
    """Get comprehensive user profile"""
    
    try:
        # Get user interests
        db_interests = db.query(DBUserInterest).filter(DBUserInterest.user_id == user.id).all()
        
        user_interests = []
        for db_interest in db_interests:
            user_interest = UserInterest(
                topic=db_interest.topic,
                keywords=db_interest.keywords,
                embedding=None,
                confidence_score=db_interest.confidence_score,
                paper_count=db_interest.paper_count,
                last_seen=db_interest.last_seen
            )
            user_interests.append(user_interest)
        
        # Generate profile summary
        profile_summary = preference_learner.create_user_profile_summary(user_interests)
        
        # Get paper stats
        paper_count = db.query(Paper).filter(Paper.uploaded_by_user_id == user.id).count()
        
        return {
            "user": {
                "username": user.username,
                "email": user.email,
                "created_at": user.created_at,
                "paper_count": paper_count
            },
            "profile_summary": profile_summary,
            "interests": [
                {
                    "topic": interest.topic,
                    "confidence": interest.confidence_score,
                    "keywords": interest.keywords[:5],
                    "paper_count": interest.paper_count
                }
                for interest in user_interests
            ]
        }
        
    except Exception as e:
        logger.error(f"Profile API failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "pdf_extractor": "ready",
            "preference_learner": "ready", 
            "arxiv_service": "ready",
            "tip_generator": "ready"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)