from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, Table, JSON, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Mapped
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

paper_topics = Table(
    'paper_topics',
    Base.metadata,
    Column('paper_id', UUID(as_uuid = True), ForeignKey('papers.id')),
    Column('topic_id', Integer, ForeignKey('research_topics.id'))
)

user_interests = Table(
    'user_interests',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('topic_id', Integer, ForeignKey('research_topics.id')),
    Column('interest_score', Float, default = 1.0)
)


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key = True)
    email = Column(String(255), unique = True, nullable = False)
    name = Column(String(255), nullable = False)

    newsletter_frequency = Column(String(50), default = 'weekly')
    max_papers_per_newsletter = Column(Integer, default = 5)
    include_coding_tips = Column(Boolean, default = True)
    coding_experience_level = Column(String(50), default = 'intermediate')

    created_at = Column(DateTime, default = datetime.utcnow)
    last_newsletter_sent = Column(DateTime)

    uploaded_papers = relationship("Paper", back_populates = "uploaded_by")
    newsletters = relationship("Newsletter", back_populates = "user")
    interests = relationship("ResearchTopic", secondary = user_interests, back_populates = "interested_users")


class ResearchTopic(Base):
    __tablename__ = 'research_topics'

    id = Column(Integer, primary_key = True)
    name = Column(String(255), unique = True, nullable = Flase)
    description = Column(Text)
    category = Column(String(100))

    paper_count = Column(Integer, default = 0)
    trending_score = Column(Float, default = 0.0)
    last_updated = Column(DateTime, default = datetime.utcnow)
    
    papers = relationship("Paper", seconday = paper_topics, back_populates = "topics")
    interested_users = relationship("User", secondary = user_interests, back_populates = "interests")


class Paper(Base):
    __tablename__ = 'papers'

    id = Column(UUID(as_uuid = True), primary_key = True, default = uuid.uuid4)

    title = Column(String(500), nullable = False)
    authors = Column(JSON)
    abstract = Column(Text)
    arxiv_id = Column(String(50), unique = True)
    doi = Column(String(255))
    publication_date = Column(DateTime)

    full_text = Column(Text)
    pdf_url = Column(String(500))
    pdf_path = Column(String(500))

    extractive_summary = Column(Text)
    abstractive_summary = Column(Text)
    hybrid_summary = Column(Text)

    relevance_score = Column(Float, default = 0.0)
    industry_impact_score = Column(Float, default = 0.0)
    personal_relevance_score = Column(Float, default = 0.0)

    is_processed = Column(Boolean, default = False)
    processing_error = Column(Text)

    source = Column(String(100))
    source_url = Column(String(500))

    discovered_at = Column(DateTime, default = datetime.utcnow)
    processed_at = Column(DateTime)
    uploaded_by_user_id = Column(Integer, ForeignKey('users.id'))

    topics = relationship("ResearchTopic", seconday = paper_topics, back_populates = "papers")
    uploaded_by = relationship("User", back_populates = "uploaded_papers")
    newsletter_inclusions = relationship("NewsletterPaper", back_populates = "paper")

    __table_args__ = (
        Index('idx_paper_relevance', 'relevance_score'),
        Index('idx_paper_data', 'publication_date'),
        Index('idx_paper_source', 'source')
    )


class CodingTip(Base):
    __tablename__ = 'coding_tips'

    id = Column(Integer, primary_key = True)
    title = Column(String(255), nullable = False)
    content = Column(Text, nullable = False)

    categoy = Column(String(100))
    difficulty_level = Column(String(50))
    tags = Column(JSON)

    estimated_read_time = Column(Integer)
    code_example = Column(Text)
    external_links = Column(JSON)

    times_included = Column(Integer, default = 0)
    last_used = Column(DateTime)

    is_active = Column(Boolean, default = True)
    created_at = Column(DateTime, default = datetime.utcnow)

    newsletter_inclusions = relationship("NewsletterTip", back_populates = "tip")


class Newsletter(Base):
    __tablename__ = 'newsletter_tips'

    id = Column(Integer, primary_key = True)
    newsletter_id = Column(UUID(as_uuid = True), ForeignKey('newsletters.id'))
    tip_id = Column(Integer, ForeignKey('coding_tips.id'))

    display_order = Column(Integer)
    custom_intro = Column(Text)

    newsletter = relationship("Newsletter", back_populates = "included_tips")
    tip = relationship("CodingTip", back_populates = "newsletter_inclusions")


class UserFeedback(Base):
    __tablename__ = 'user_feedback'

    id = Column(Integer, primary_key = True)
    user_id = Column(Integer, ForeignKey('users.id'))

    paper_id = Column(UUID(as_uuid = True), ForeignKey('papers.id'), nullable = True)
    tip_id = Column(Integer, ForeignKey('coding_tips.id'), nullable = True)

    rating = Column(Integer) # 1-5 Scale
    feedback_type = Column(String(50)) # relevant, not_relevant, too_basic, too_advanced
    comments = Column(Text)

    created_at = Column(DateTime, default = datetime.utcnow)


class ProcessingJob(Base):
    __tablename__ = 'processing_jobs'

    id = Column(UUID(as_uuid = True), primary_key = True, default = uuid.uuid4)
    job_type = Column(String(100), nullable = False)
    status = Column(String(50), default = 'pending')

    user_id = Column(Integer, ForeignKey('users.id'), nullable = True)
    parameters = Column(JSON)

    progress_percentage = Column(Float, default = 0.0)
    current_step = Column(String(255))

    result = Column(JSON)
    error_message = Column(Text)

    created_at = Column(DateTime, default = datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)