from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from models.database import get_db, Technology, TrendData
from config import settings

router = APIRouter()

# Pydantic models for API responses
class TechnologyResponse(BaseModel):
    id: int
    name: str
    category: str
    description: Optional[str] = None
    keywords: List[str] = []
    first_detected: datetime
    last_updated: datetime
    
    class Config:
        from_attributes = True

class TrendDataResponse(BaseModel):
    id: int
    technology_id: int
    source: str
    date: datetime
    github_stars: int = 0
    github_forks: int = 0
    github_issues: int = 0
    arxiv_papers: int = 0
    patent_filings: int = 0
    job_postings: int = 0
    social_mentions: int = 0
    trend_score: float = 0.0
    momentum_score: float = 0.0
    adoption_score: float = 0.0
    
    class Config:
        from_attributes = True

class TrendSummaryResponse(BaseModel):
    technology: TechnologyResponse
    latest_trend: TrendDataResponse
    trend_change: float  # Percentage change from previous period
    ranking: int  # Position in category rankings
    
class CategoryTrendsResponse(BaseModel):
    category: str
    total_technologies: int
    average_trend_score: float
    top_technologies: List[TrendSummaryResponse]
    
class TrendsOverviewResponse(BaseModel):
    total_technologies: int
    active_trends: int
    categories: List[str]
    top_trending: List[TrendSummaryResponse]
    category_breakdown: List[CategoryTrendsResponse]
    last_updated: datetime

class TrendsListResponse(BaseModel):
    trends: List[TrendDataResponse]
    total: int

# Simple endpoint for frontend
@router.get("/", response_model=TrendsListResponse)
async def get_trends_list(
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get list of trend data for frontend"""
    try:
        # Get recent trend data
        recent_date = datetime.utcnow() - timedelta(days=30)
        trends_query = db.query(TrendData).filter(
            TrendData.date >= recent_date
        ).order_by(TrendData.date.desc()).limit(limit)
        
        trends = trends_query.all()
        
        # Convert to response format
        trend_responses = []
        for trend in trends:
            trend_responses.append(TrendDataResponse(
                id=trend.id,
                technology_id=trend.technology_id,
                source=trend.source,
                date=trend.date,
                github_stars=trend.github_stars,
                github_forks=trend.github_forks,
                github_issues=trend.github_issues,
                arxiv_papers=trend.arxiv_papers,
                patent_filings=trend.patent_filings,
                job_postings=trend.job_postings,
                social_mentions=trend.social_mentions,
                trend_score=trend.trend_score,
                momentum_score=trend.momentum_score,
                adoption_score=trend.adoption_score
            ))
        
        return TrendsListResponse(
            trends=trend_responses,
            total=len(trend_responses)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching trends: {str(e)}")

@router.get("/technology/{tech_id}", response_model=TrendSummaryResponse)
async def get_technology_trend(
    tech_id: int,
    db: Session = Depends(get_db)
):
    """Get trend data for a specific technology"""
    try:
        # Get technology
        technology = db.query(Technology).filter(Technology.id == tech_id).first()
        if not technology:
            raise HTTPException(status_code=404, detail="Technology not found")
        
        # Get latest trend data
        latest_trend = db.query(TrendData).filter(
            TrendData.technology_id == tech_id
        ).order_by(TrendData.date.desc()).first()
        
        if not latest_trend:
            raise HTTPException(status_code=404, detail="No trend data found for this technology")
        
        # Calculate trend change (simplified)
        trend_change = 0.0
        
        # Get ranking in category
        ranking_query = db.query(Technology, TrendData).join(TrendData).filter(
            Technology.category == technology.category,
            TrendData.date >= datetime.utcnow() - timedelta(days=7)
        ).order_by(TrendData.trend_score.desc()).all()
        
        ranking = 1
        for i, (tech, _) in enumerate(ranking_query):
            if tech.id == tech_id:
                ranking = i + 1
                break
        
        return TrendSummaryResponse(
            technology=TechnologyResponse.from_orm(technology),
            latest_trend=TrendDataResponse.from_orm(latest_trend),
            trend_change=trend_change,
            ranking=ranking
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching technology trend: {str(e)}")

@router.get("/category/{category}", response_model=CategoryTrendsResponse)
async def get_category_trends(
    category: str,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get trends for a specific category"""
    try:
        # Check if category exists
        total_technologies = db.query(Technology).filter(Technology.category == category).count()
        if total_technologies == 0:
            raise HTTPException(status_code=404, detail="Category not found")
        
        # Get recent date for filtering
        recent_date = datetime.utcnow() - timedelta(days=7)
        
        # Get average trend score for category
        avg_score_result = db.query(db.func.avg(TrendData.trend_score)).join(Technology).filter(
            Technology.category == category,
            TrendData.date >= recent_date
        ).scalar()
        
        average_trend_score = float(avg_score_result) if avg_score_result else 0.0
        
        # Get top technologies in category
        top_query = db.query(Technology, TrendData).join(TrendData).filter(
            Technology.category == category,
            TrendData.date >= recent_date
        ).order_by(TrendData.trend_score.desc()).limit(limit)
        
        top_data = top_query.all()
        top_technologies = []
        
        for tech, trend_data in top_data:
            top_technologies.append(TrendSummaryResponse(
                technology=TechnologyResponse.from_orm(tech),
                latest_trend=TrendDataResponse.from_orm(trend_data),
                trend_change=0.0,  # Would be calculated with historical data
                ranking=len(top_technologies) + 1
            ))
        
        return CategoryTrendsResponse(
            category=category,
            total_technologies=total_technologies,
            average_trend_score=average_trend_score,
            top_technologies=top_technologies
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching category trends: {str(e)}")

@router.get("/historical/{tech_id}", response_model=List[TrendDataResponse])
async def get_historical_trends(
    tech_id: int,
    days: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get historical trend data for a technology"""
    try:
        # Check if technology exists
        technology = db.query(Technology).filter(Technology.id == tech_id).first()
        if not technology:
            raise HTTPException(status_code=404, detail="Technology not found")
        
        # Get historical data
        start_date = datetime.utcnow() - timedelta(days=days)
        historical_data = db.query(TrendData).filter(
            TrendData.technology_id == tech_id,
            TrendData.date >= start_date
        ).order_by(TrendData.date.asc()).all()
        
        return [TrendDataResponse.from_orm(data) for data in historical_data]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching historical trends: {str(e)}")

@router.get("/search", response_model=List[TrendSummaryResponse])
async def search_trends(
    query: str = Query(..., min_length=1),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Search for technologies and their trends"""
    try:
        # Search in technology names, descriptions, and keywords
        search_query = f"%{query}%"
        
        technologies = db.query(Technology).filter(
            db.or_(
                Technology.name.ilike(search_query),
                Technology.description.ilike(search_query),
                Technology.keywords.contains([query])
            )
        ).limit(limit).all()
        
        results = []
        recent_date = datetime.utcnow() - timedelta(days=7)
        
        for tech in technologies:
            # Get latest trend data
            latest_trend = db.query(TrendData).filter(
                TrendData.technology_id == tech.id,
                TrendData.date >= recent_date
            ).order_by(TrendData.date.desc()).first()
            
            if latest_trend:
                results.append(TrendSummaryResponse(
                    technology=TechnologyResponse.from_orm(tech),
                    latest_trend=TrendDataResponse.from_orm(latest_trend),
                    trend_change=0.0,
                    ranking=0  # Not applicable for search results
                ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching trends: {str(e)}")

@router.get("/stats", response_model=Dict[str, Any])
async def get_trend_stats(db: Session = Depends(get_db)):
    """Get overall trend statistics"""
    try:
        recent_date = datetime.utcnow() - timedelta(days=7)
        
        # Total technologies
        total_technologies = db.query(Technology).count()
        
        # Active trends
        active_trends = db.query(Technology).join(TrendData).filter(
            TrendData.date >= recent_date
        ).distinct().count()
        
        # Average trend score
        avg_score = db.query(db.func.avg(TrendData.trend_score)).filter(
            TrendData.date >= recent_date
        ).scalar()
        
        # Top categories by activity
        category_stats = db.query(
            Technology.category,
            db.func.count(TrendData.id).label('trend_count'),
            db.func.avg(TrendData.trend_score).label('avg_score')
        ).join(TrendData).filter(
            TrendData.date >= recent_date
        ).group_by(Technology.category).order_by(
            db.func.count(TrendData.id).desc()
        ).limit(10).all()
        
        # Most active sources
        source_stats = db.query(
            TrendData.source,
            db.func.count(TrendData.id).label('count')
        ).filter(
            TrendData.date >= recent_date
        ).group_by(TrendData.source).order_by(
            db.func.count(TrendData.id).desc()
        ).all()
        
        return {
            "total_technologies": total_technologies,
            "active_trends": active_trends,
            "average_trend_score": float(avg_score) if avg_score else 0.0,
            "category_stats": [
                {
                    "category": stat[0],
                    "trend_count": stat[1],
                    "average_score": float(stat[2])
                }
                for stat in category_stats
            ],
            "source_stats": [
                {
                    "source": stat[0],
                    "count": stat[1]
                }
                for stat in source_stats
            ],
            "last_updated": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching trend stats: {str(e)}") 