from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from models.database import get_db, Technology, TrendData, Prediction, create_technology
from config import settings, TECHNOLOGY_CATEGORIES

router = APIRouter()

# Pydantic models
class TechnologyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    category: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    keywords: List[str] = []

class TechnologyUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    category: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    keywords: Optional[List[str]] = None

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

class TechnologyDetailResponse(BaseModel):
    technology: TechnologyResponse
    latest_trend_score: float
    total_predictions: int
    recent_activity: List[Dict[str, Any]]
    related_technologies: List[TechnologyResponse]
    category_ranking: int

@router.get("/", response_model=List[TechnologyResponse])
async def get_technologies(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    category: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
    db: Session = Depends(get_db)
):
    """Get all technologies with optional filtering"""
    try:
        query = db.query(Technology)
        
        # Apply category filter
        if category:
            query = query.filter(Technology.category == category)
        
        # Apply search filter
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                db.or_(
                    Technology.name.ilike(search_pattern),
                    Technology.description.ilike(search_pattern)
                )
            )
        
        # Order by last updated
        query = query.order_by(Technology.last_updated.desc())
        
        # Apply pagination
        technologies = query.offset(skip).limit(limit).all()
        
        return [TechnologyResponse.from_orm(tech) for tech in technologies]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching technologies: {str(e)}")

@router.get("/{tech_id}", response_model=TechnologyDetailResponse)
async def get_technology_detail(
    tech_id: int,
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific technology"""
    try:
        # Get technology
        technology = db.query(Technology).filter(Technology.id == tech_id).first()
        if not technology:
            raise HTTPException(status_code=404, detail="Technology not found")
        
        # Get latest trend score
        latest_trend = db.query(TrendData).filter(
            TrendData.technology_id == tech_id
        ).order_by(TrendData.date.desc()).first()
        
        latest_trend_score = latest_trend.trend_score if latest_trend else 0.0
        
        # Get total predictions
        total_predictions = db.query(Prediction).filter(
            Prediction.technology_id == tech_id
        ).count()
        
        # Get recent activity (last 10 trend data points)
        recent_trends = db.query(TrendData).filter(
            TrendData.technology_id == tech_id
        ).order_by(TrendData.date.desc()).limit(10).all()
        
        recent_activity = []
        for trend in recent_trends:
            recent_activity.append({
                "date": trend.date.isoformat(),
                "source": trend.source,
                "trend_score": trend.trend_score,
                "github_stars": trend.github_stars,
                "github_forks": trend.github_forks
            })
        
        # Get related technologies (same category, similar trend scores)
        related_query = db.query(Technology).filter(
            Technology.category == technology.category,
            Technology.id != tech_id
        ).limit(5)
        
        related_technologies = [TechnologyResponse.from_orm(tech) for tech in related_query.all()]
        
        # Get category ranking
        category_ranking_query = db.query(Technology, TrendData).join(TrendData).filter(
            Technology.category == technology.category,
            TrendData.date >= datetime.utcnow() - timedelta(days=7)
        ).order_by(TrendData.trend_score.desc()).all()
        
        category_ranking = 1
        for i, (tech, _) in enumerate(category_ranking_query):
            if tech.id == tech_id:
                category_ranking = i + 1
                break
        
        return TechnologyDetailResponse(
            technology=TechnologyResponse.from_orm(technology),
            latest_trend_score=latest_trend_score,
            total_predictions=total_predictions,
            recent_activity=recent_activity,
            related_technologies=related_technologies,
            category_ranking=category_ranking
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching technology detail: {str(e)}")

@router.post("/", response_model=TechnologyResponse)
async def create_technology_endpoint(
    technology_data: TechnologyCreateRequest,
    db: Session = Depends(get_db)
):
    """Create a new technology"""
    try:
        # Check if technology already exists
        existing_tech = db.query(Technology).filter(
            Technology.name == technology_data.name
        ).first()
        
        if existing_tech:
            raise HTTPException(status_code=400, detail="Technology with this name already exists")
        
        # Validate category
        if technology_data.category not in TECHNOLOGY_CATEGORIES:
            raise HTTPException(status_code=400, detail="Invalid category")
        
        # Create technology
        technology = create_technology(
            db=db,
            name=technology_data.name,
            category=technology_data.category,
            description=technology_data.description,
            keywords=technology_data.keywords
        )
        
        return TechnologyResponse.from_orm(technology)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating technology: {str(e)}")

@router.put("/{tech_id}", response_model=TechnologyResponse)
async def update_technology(
    tech_id: int,
    technology_data: TechnologyUpdateRequest,
    db: Session = Depends(get_db)
):
    """Update an existing technology"""
    try:
        # Get technology
        technology = db.query(Technology).filter(Technology.id == tech_id).first()
        if not technology:
            raise HTTPException(status_code=404, detail="Technology not found")
        
        # Update fields
        if technology_data.name is not None:
            # Check if new name is unique
            existing_tech = db.query(Technology).filter(
                Technology.name == technology_data.name,
                Technology.id != tech_id
            ).first()
            if existing_tech:
                raise HTTPException(status_code=400, detail="Technology with this name already exists")
            technology.name = technology_data.name
        
        if technology_data.category is not None:
            if technology_data.category not in TECHNOLOGY_CATEGORIES:
                raise HTTPException(status_code=400, detail="Invalid category")
            technology.category = technology_data.category
        
        if technology_data.description is not None:
            technology.description = technology_data.description
        
        if technology_data.keywords is not None:
            technology.keywords = technology_data.keywords
        
        technology.last_updated = datetime.utcnow()
        
        db.commit()
        db.refresh(technology)
        
        return TechnologyResponse.from_orm(technology)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating technology: {str(e)}")

@router.delete("/{tech_id}")
async def delete_technology(
    tech_id: int,
    db: Session = Depends(get_db)
):
    """Delete a technology"""
    try:
        # Get technology
        technology = db.query(Technology).filter(Technology.id == tech_id).first()
        if not technology:
            raise HTTPException(status_code=404, detail="Technology not found")
        
        # Delete related data first
        db.query(TrendData).filter(TrendData.technology_id == tech_id).delete()
        db.query(Prediction).filter(Prediction.technology_id == tech_id).delete()
        
        # Delete technology
        db.delete(technology)
        db.commit()
        
        return {"message": "Technology deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting technology: {str(e)}")

@router.get("/categories/list", response_model=List[str])
async def get_technology_categories():
    """Get all available technology categories"""
    return list(TECHNOLOGY_CATEGORIES.keys())

@router.get("/categories/{category}/keywords", response_model=List[str])
async def get_category_keywords(category: str):
    """Get keywords for a specific category"""
    if category not in TECHNOLOGY_CATEGORIES:
        raise HTTPException(status_code=404, detail="Category not found")
    
    return TECHNOLOGY_CATEGORIES[category]

@router.get("/search/suggest", response_model=List[str])
async def get_search_suggestions(
    query: str = Query(..., min_length=1),
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Get search suggestions for technologies"""
    try:
        search_pattern = f"%{query}%"
        
        # Search in technology names
        suggestions = db.query(Technology.name).filter(
            Technology.name.ilike(search_pattern)
        ).limit(limit).all()
        
        return [suggestion[0] for suggestion in suggestions]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching search suggestions: {str(e)}")

@router.get("/stats/overview", response_model=Dict[str, Any])
async def get_technology_stats(db: Session = Depends(get_db)):
    """Get overall technology statistics"""
    try:
        # Total technologies
        total_technologies = db.query(Technology).count()
        
        # Technologies by category
        category_stats = db.query(
            Technology.category,
            db.func.count(Technology.id).label('count')
        ).group_by(Technology.category).all()
        
        category_breakdown = {stat[0]: stat[1] for stat in category_stats}
        
        # Recent additions (last 30 days)
        recent_date = datetime.utcnow() - timedelta(days=30)
        recent_additions = db.query(Technology).filter(
            Technology.first_detected >= recent_date
        ).count()
        
        # Most active technologies (by trend data)
        most_active = db.query(
            Technology.name,
            db.func.count(TrendData.id).label('activity_count')
        ).join(TrendData).filter(
            TrendData.date >= datetime.utcnow() - timedelta(days=7)
        ).group_by(Technology.name).order_by(
            db.func.count(TrendData.id).desc()
        ).limit(10).all()
        
        most_active_list = [
            {"name": stat[0], "activity_count": stat[1]}
            for stat in most_active
        ]
        
        # Average trend scores by category
        avg_scores = db.query(
            Technology.category,
            db.func.avg(TrendData.trend_score).label('avg_score')
        ).join(TrendData).filter(
            TrendData.date >= datetime.utcnow() - timedelta(days=7)
        ).group_by(Technology.category).all()
        
        avg_scores_dict = {
            stat[0]: float(stat[1]) for stat in avg_scores
        }
        
        return {
            "total_technologies": total_technologies,
            "category_breakdown": category_breakdown,
            "recent_additions": recent_additions,
            "most_active_technologies": most_active_list,
            "average_scores_by_category": avg_scores_dict,
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching technology stats: {str(e)}")

@router.get("/trending/rising", response_model=List[TechnologyResponse])
async def get_rising_technologies(
    limit: int = Query(default=20, ge=1, le=100),
    days: int = Query(default=7, ge=1, le=30),
    db: Session = Depends(get_db)
):
    """Get technologies with rising trend scores"""
    try:
        # Get technologies with positive trend momentum
        recent_date = datetime.utcnow() - timedelta(days=days)
        older_date = datetime.utcnow() - timedelta(days=days*2)
        
        # Subquery for recent average scores
        recent_scores = db.query(
            TrendData.technology_id,
            db.func.avg(TrendData.trend_score).label('recent_avg')
        ).filter(
            TrendData.date >= recent_date
        ).group_by(TrendData.technology_id).subquery()
        
        # Subquery for older average scores
        older_scores = db.query(
            TrendData.technology_id,
            db.func.avg(TrendData.trend_score).label('older_avg')
        ).filter(
            TrendData.date >= older_date,
            TrendData.date < recent_date
        ).group_by(TrendData.technology_id).subquery()
        
        # Find technologies with increasing scores
        rising_tech_ids = db.query(
            Technology.id,
            (recent_scores.c.recent_avg - older_scores.c.older_avg).label('momentum')
        ).join(recent_scores, Technology.id == recent_scores.c.technology_id).join(
            older_scores, Technology.id == older_scores.c.technology_id
        ).filter(
            recent_scores.c.recent_avg > older_scores.c.older_avg
        ).order_by(
            db.desc('momentum')
        ).limit(limit).all()
        
        # Get the actual technology objects
        tech_ids = [tech[0] for tech in rising_tech_ids]
        rising_technologies = db.query(Technology).filter(
            Technology.id.in_(tech_ids)
        ).all()
        
        return [TechnologyResponse.from_orm(tech) for tech in rising_technologies]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching rising technologies: {str(e)}")

@router.get("/trending/declining", response_model=List[TechnologyResponse])
async def get_declining_technologies(
    limit: int = Query(default=20, ge=1, le=100),
    days: int = Query(default=7, ge=1, le=30),
    db: Session = Depends(get_db)
):
    """Get technologies with declining trend scores"""
    try:
        # Get technologies with negative trend momentum
        recent_date = datetime.utcnow() - timedelta(days=days)
        older_date = datetime.utcnow() - timedelta(days=days*2)
        
        # Subquery for recent average scores
        recent_scores = db.query(
            TrendData.technology_id,
            db.func.avg(TrendData.trend_score).label('recent_avg')
        ).filter(
            TrendData.date >= recent_date
        ).group_by(TrendData.technology_id).subquery()
        
        # Subquery for older average scores
        older_scores = db.query(
            TrendData.technology_id,
            db.func.avg(TrendData.trend_score).label('older_avg')
        ).filter(
            TrendData.date >= older_date,
            TrendData.date < recent_date
        ).group_by(TrendData.technology_id).subquery()
        
        # Find technologies with decreasing scores
        declining_tech_ids = db.query(
            Technology.id,
            (older_scores.c.older_avg - recent_scores.c.recent_avg).label('decline')
        ).join(recent_scores, Technology.id == recent_scores.c.technology_id).join(
            older_scores, Technology.id == older_scores.c.technology_id
        ).filter(
            recent_scores.c.recent_avg < older_scores.c.older_avg
        ).order_by(
            db.desc('decline')
        ).limit(limit).all()
        
        # Get the actual technology objects
        tech_ids = [tech[0] for tech in declining_tech_ids]
        declining_technologies = db.query(Technology).filter(
            Technology.id.in_(tech_ids)
        ).all()
        
        return [TechnologyResponse.from_orm(tech) for tech in declining_technologies]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching declining technologies: {str(e)}") 