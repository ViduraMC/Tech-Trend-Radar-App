from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.sqlite import JSON
from datetime import datetime
import json

from config import settings

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Database models
class Technology(Base):
    __tablename__ = "technologies"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    category = Column(String(100), nullable=False)
    description = Column(Text)
    keywords = Column(JSON)
    first_detected = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    trends = relationship("TrendData", back_populates="technology")
    predictions = relationship("Prediction", back_populates="technology")
    metrics = relationship("TechnologyMetric", back_populates="technology")

class TrendData(Base):
    __tablename__ = "trend_data"
    
    id = Column(Integer, primary_key=True, index=True)
    technology_id = Column(Integer, ForeignKey("technologies.id"), nullable=False)
    source = Column(String(50), nullable=False)  # github, arxiv, patents, etc.
    date = Column(DateTime, nullable=False)
    
    # Metrics
    github_stars = Column(Integer, default=0)
    github_forks = Column(Integer, default=0)
    github_issues = Column(Integer, default=0)
    arxiv_papers = Column(Integer, default=0)
    patent_filings = Column(Integer, default=0)
    job_postings = Column(Integer, default=0)
    social_mentions = Column(Integer, default=0)
    
    # Calculated scores
    trend_score = Column(Float, default=0.0)
    momentum_score = Column(Float, default=0.0)
    adoption_score = Column(Float, default=0.0)
    
    # Relationships
    technology = relationship("Technology", back_populates="trends")

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    technology_id = Column(Integer, ForeignKey("technologies.id"), nullable=False)
    prediction_date = Column(DateTime, default=datetime.utcnow)
    target_date = Column(DateTime, nullable=False)
    
    # Prediction metrics
    adoption_probability = Column(Float, nullable=False)
    market_impact_score = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    confidence_interval = Column(Float, nullable=False)
    
    # Prediction details
    model_used = Column(String(100), nullable=False)
    features_used = Column(JSON)
    prediction_reasoning = Column(Text)
    
    # Validation
    is_validated = Column(Boolean, default=False)
    actual_outcome = Column(Float)
    accuracy_score = Column(Float)
    
    # Relationships
    technology = relationship("Technology", back_populates="predictions")

class TechnologyMetric(Base):
    __tablename__ = "technology_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    technology_id = Column(Integer, ForeignKey("technologies.id"), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    date = Column(DateTime, default=datetime.utcnow)
    source = Column(String(50), nullable=False)
    
    # Relationships
    technology = relationship("Technology", back_populates="metrics")

class DataSource(Base):
    __tablename__ = "data_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    url = Column(String(255))
    is_active = Column(Boolean, default=True)
    last_sync = Column(DateTime)
    sync_frequency = Column(Integer, default=3600)  # seconds
    
    # Rate limiting
    rate_limit = Column(Integer, default=1000)
    requests_made = Column(Integer, default=0)
    last_request = Column(DateTime)
    
    # Configuration
    config = Column(JSON)
    api_key_required = Column(Boolean, default=False)
    
class CollectionLog(Base):
    __tablename__ = "collection_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    source_name = Column(String(100), nullable=False)
    collection_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), nullable=False)  # success, error, partial
    records_collected = Column(Integer, default=0)
    errors_encountered = Column(Integer, default=0)
    execution_time = Column(Float)  # seconds
    
    # Details
    details = Column(JSON)
    error_message = Column(Text)

# Database utility functions
async def init_database():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_sync():
    """Get database session (synchronous)"""
    return SessionLocal()

# Helper functions for database operations
def create_technology(db, name: str, category: str, description: str = None, keywords: list = None):
    """Create a new technology entry"""
    tech = Technology(
        name=name,
        category=category,
        description=description,
        keywords=keywords or []
    )
    db.add(tech)
    db.commit()
    db.refresh(tech)
    return tech

def get_technology_by_name(db, name: str):
    """Get technology by name"""
    return db.query(Technology).filter(Technology.name == name).first()

def get_technologies_by_category(db, category: str):
    """Get all technologies in a category"""
    return db.query(Technology).filter(Technology.category == category).all()

def add_trend_data(db, technology_id: int, source: str, **metrics):
    """Add trend data for a technology"""
    trend = TrendData(
        technology_id=technology_id,
        source=source,
        date=datetime.utcnow(),
        **metrics
    )
    db.add(trend)
    db.commit()
    db.refresh(trend)
    return trend

def create_prediction(db, technology_id: int, target_date: datetime, **prediction_data):
    """Create a new prediction"""
    prediction = Prediction(
        technology_id=technology_id,
        target_date=target_date,
        **prediction_data
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction

def log_collection(db, source_name: str, status: str, **details):
    """Log data collection activity"""
    log = CollectionLog(
        source_name=source_name,
        status=status,
        **details
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log 