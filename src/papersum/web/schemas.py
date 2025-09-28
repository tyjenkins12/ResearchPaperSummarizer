from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import date, datetime
from uuid import UUID


class UserCreate(BaseModel):
    """Schema for creating a new user"""
    username: str = Field(min_length=3, max_length=50)
    email: str = Field(pattern=r'^[^@]+@[^@]+\.[^@]+$')
    newsletter_frequency: str = Field(default="weekly")


class UserResponse(BaseModel):
    """Schema for user response"""
    id: int
    username: str
    email: str
    created_at: datetime
    newsletter_enabled: bool
    newsletter_frequency: str


class PaperResponse(BaseModel):
    """Schema for paper response"""
    id: UUID
    title: str
    authors: List[str]
    abstract: Optional[str]
    relevance_score: float
    discovered_at: datetime
    metadata: Dict[str, Any]


class InterestResponse(BaseModel):
    """Schema for user interest response"""
    id: int
    topic: str
    keywords: List[str]
    confidence_score: float
    paper_count: int
    last_seen: Optional[datetime]


class FeedbackRequest(BaseModel):
    """Schema for paper feedback"""
    paper_id: str
    rating: int = Field(ge=1, le=5)
    notes: Optional[str] = None


class PreferencesResponse(BaseModel):
    """Schema for user preferences overview"""
    interests_count: int
    papers_uploaded: int
    profile_summary: Dict[str, Any]
    top_interests: List[InterestResponse]


class DiscoverRequest(BaseModel):
    """Schema for paper discovery request"""
    max_papers: int = Field(default=10, ge=1, le=50)
    days_back: int = Field(default=7, ge=1, le=90)
    include_trending: bool = True
    categories: Optional[List[str]] = None


class TipResponse(BaseModel):
    """Schema for coding tip response"""
    title: str
    content: str
    category: str
    difficulty: str
    code_example: Optional[str]
    tags: List[str]


class NewsletterPreview(BaseModel):
    """Schema for newsletter preview"""
    papers: List[PaperResponse]
    coding_tips: List[TipResponse]
    generated_at: datetime
    estimated_read_time: int  # minutes


class UploadResponse(BaseModel):
    """Schema for paper upload response"""
    processed_papers: List[Dict[str, str]]
    errors: List[str]
    success_count: int
    error_count: int
    preferences_updated: bool


class FeedHeadline(BaseModel):
    """Schema for a daily headline entry"""
    id: int
    title: str
    summary: str
    rank: int
    metadata: Dict[str, Any]
    paper_id: Optional[UUID]
    paper_title: Optional[str]
    paper_url: Optional[str]


class DailyFeedResponse(BaseModel):
    """Schema for daily feed response"""
    date: date
    items: List[FeedHeadline]


class WeeklyReportItem(BaseModel):
    """Schema for an item inside the weekly highlight report"""
    id: int
    headline: str
    summary: str
    metadata: Dict[str, Any]
    paper_id: Optional[UUID]


class WeeklyFeedResponse(BaseModel):
    """Schema for weekly feed response"""
    id: UUID
    week_start: date
    week_end: date
    summary: Optional[str]
    metadata: Dict[str, Any]
    items: List[WeeklyReportItem]


class FeedFeedbackRequest(BaseModel):
    """Schema for capturing feed interactions"""

    target_type: Literal["headline", "weekly_item", "weekly_report", "paper"]
    target_id: str
    interaction_type: Literal["like", "dislike", "save", "request", "bookmark"]
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    note: Optional[str] = None
