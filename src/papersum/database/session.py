from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator
import logging

from .models import Base

logger = logging.getLogger(__name__)

# Database configuration (using SQLite for simplicity)
DATABASE_URL = "sqlite:///./papers.db"

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite specific
    poolclass=StaticPool,
    echo=False  # Set to True for SQL debugging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


def get_db_session() -> Generator[Session, None, None]:
    """Dependency function to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session_sync() -> Session:
    """Get database session for synchronous use"""
    return SessionLocal()


# Initialize database on import
try:
    create_tables()
except Exception as e:
    logger.warning(f"Database initialization failed: {e}")