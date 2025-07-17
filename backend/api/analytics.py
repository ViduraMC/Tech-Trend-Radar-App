from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from models.database import get_db, Technology, TrendData, Prediction, CollectionLog
from config import settings, TECHNOLOGY_CATEGORIES

router = APIRouter()

# Pydantic models for analytics responses
class TrendAnalyticsResponse(BaseModel):
    technology_name: str
    category: str
    trend_score: float
    momentum: float
    volatility: float
    percentile_rank: float

class CategoryAnalyticsResponse(BaseModel):
    category: str
    total_technologies: int
    average_trend_score: float
    growth_rate: float
    volatility: float
    top_performers: List[str]

class TimeSeriesDataPoint(BaseModel):
    date: datetime
    value: float
    technology_count: int = 0

class TimeSeriesResponse(BaseModel):
    label: str
    data: List[TimeSeriesDataPoint]
    trend_direction: str  # "up", "down", "stable"
    growth_rate: float

class ComparisonResponse(BaseModel):
    technology_a: str
    technology_b: str
    comparison_metrics: Dict[str, Any]
    recommendation: str

class InsightResponse(BaseModel):
    type: str  # "trend", "prediction", "anomaly", "opportunity"
    title: str
    description: str
    confidence: float
    data: Dict[str, Any]

class AnalyticsOverviewResponse(BaseModel):
    summary_stats: Dict[str, Any]
    top_trending: List[TrendAnalyticsResponse]
    category_performance: List[CategoryAnalyticsResponse]
    recent_insights: List[InsightResponse]
    time_series: List[TimeSeriesResponse]

@router.get("/overview", response_model=AnalyticsOverviewResponse)
async def get_analytics_overview(
    days: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get comprehensive analytics overview"""
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Summary statistics
        total_technologies = db.query(Technology).count()
        active_technologies = db.query(Technology).join(TrendData).filter(
            TrendData.date >= start_date
        ).distinct().count()
        
        avg_trend_score = db.query(func.avg(TrendData.trend_score)).filter(
            TrendData.date >= start_date
        ).scalar() or 0.0
        
        total_predictions = db.query(Prediction).count()
        
        summary_stats = {
            "total_technologies": total_technologies,
            "active_technologies": active_technologies,
            "average_trend_score": float(avg_trend_score),
            "total_predictions": total_predictions,
            "data_freshness": (datetime.utcnow() - start_date).days
        }
        
        # Top trending technologies
        top_trending_query = db.query(
            Technology.name,
            Technology.category,
            func.avg(TrendData.trend_score).label('avg_score'),
            func.count(TrendData.id).label('data_points')
        ).join(TrendData).filter(
            TrendData.date >= start_date
        ).group_by(Technology.id, Technology.name, Technology.category).order_by(
            func.avg(TrendData.trend_score).desc()
        ).limit(10).all()
        
        top_trending = []
        for tech_name, category, avg_score, data_points in top_trending_query:
            # Calculate momentum and volatility (simplified)
            momentum = await _calculate_momentum(db, tech_name, days)
            volatility = await _calculate_volatility(db, tech_name, days)
            percentile_rank = await _calculate_percentile_rank(db, tech_name, float(avg_score))
            
            top_trending.append(TrendAnalyticsResponse(
                technology_name=tech_name,
                category=category,
                trend_score=float(avg_score),
                momentum=momentum,
                volatility=volatility,
                percentile_rank=percentile_rank
            ))
        
        # Category performance
        category_performance = []
        for category in TECHNOLOGY_CATEGORIES.keys():
            cat_stats = await _get_category_analytics(db, category, start_date)
            if cat_stats:
                category_performance.append(cat_stats)
        
        # Recent insights
        recent_insights = await _generate_insights(db, start_date)
        
        # Time series data
        time_series = await _get_time_series_data(db, start_date)
        
        return AnalyticsOverviewResponse(
            summary_stats=summary_stats,
            top_trending=top_trending,
            category_performance=category_performance,
            recent_insights=recent_insights,
            time_series=time_series
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching analytics overview: {str(e)}")

@router.get("/technology/{tech_id}/analysis", response_model=Dict[str, Any])
async def get_technology_analysis(
    tech_id: int,
    days: int = Query(default=90, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get detailed analysis for a specific technology"""
    try:
        # Get technology
        technology = db.query(Technology).filter(Technology.id == tech_id).first()
        if not technology:
            raise HTTPException(status_code=404, detail="Technology not found")
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get trend data
        trend_data = db.query(TrendData).filter(
            TrendData.technology_id == tech_id,
            TrendData.date >= start_date
        ).order_by(TrendData.date.asc()).all()
        
        if not trend_data:
            raise HTTPException(status_code=404, detail="No trend data found")
        
        # Calculate metrics
        scores = [data.trend_score for data in trend_data]
        github_stars = [data.github_stars for data in trend_data]
        
        analysis = {
            "technology": {
                "id": technology.id,
                "name": technology.name,
                "category": technology.category,
                "description": technology.description
            },
            "trend_metrics": {
                "current_score": scores[-1] if scores else 0.0,
                "average_score": sum(scores) / len(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
                "volatility": _calculate_volatility_from_scores(scores),
                "momentum": _calculate_momentum_from_scores(scores),
                "trend_direction": _determine_trend_direction(scores)
            },
            "github_metrics": {
                "current_stars": github_stars[-1] if github_stars else 0,
                "star_growth": github_stars[-1] - github_stars[0] if len(github_stars) > 1 else 0,
                "average_stars": sum(github_stars) / len(github_stars) if github_stars else 0.0
            },
            "comparative_analysis": {
                "category_ranking": await _get_category_ranking(db, tech_id, technology.category),
                "percentile_rank": await _calculate_percentile_rank(db, technology.name, scores[-1] if scores else 0.0),
                "similar_technologies": await _find_similar_technologies(db, tech_id, technology.category)
            },
            "time_series": [
                {
                    "date": data.date.isoformat(),
                    "trend_score": data.trend_score,
                    "github_stars": data.github_stars,
                    "github_forks": data.github_forks,
                    "source": data.source
                }
                for data in trend_data
            ]
        }
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching technology analysis: {str(e)}")

@router.get("/comparison", response_model=ComparisonResponse)
async def compare_technologies(
    tech_a: str = Query(..., description="First technology name"),
    tech_b: str = Query(..., description="Second technology name"),
    days: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Compare two technologies"""
    try:
        # Get technologies
        tech_a_obj = db.query(Technology).filter(Technology.name == tech_a).first()
        tech_b_obj = db.query(Technology).filter(Technology.name == tech_b).first()
        
        if not tech_a_obj or not tech_b_obj:
            raise HTTPException(status_code=404, detail="One or both technologies not found")
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get trend data for both technologies
        trend_a = db.query(TrendData).filter(
            TrendData.technology_id == tech_a_obj.id,
            TrendData.date >= start_date
        ).order_by(TrendData.date.asc()).all()
        
        trend_b = db.query(TrendData).filter(
            TrendData.technology_id == tech_b_obj.id,
            TrendData.date >= start_date
        ).order_by(TrendData.date.asc()).all()
        
        # Calculate comparison metrics
        scores_a = [data.trend_score for data in trend_a]
        scores_b = [data.trend_score for data in trend_b]
        
        stars_a = [data.github_stars for data in trend_a]
        stars_b = [data.github_stars for data in trend_b]
        
        comparison_metrics = {
            "trend_scores": {
                "tech_a_avg": sum(scores_a) / len(scores_a) if scores_a else 0.0,
                "tech_b_avg": sum(scores_b) / len(scores_b) if scores_b else 0.0,
                "winner": tech_a if (sum(scores_a) / len(scores_a) if scores_a else 0.0) > (sum(scores_b) / len(scores_b) if scores_b else 0.0) else tech_b
            },
            "github_stars": {
                "tech_a_latest": stars_a[-1] if stars_a else 0,
                "tech_b_latest": stars_b[-1] if stars_b else 0,
                "winner": tech_a if (stars_a[-1] if stars_a else 0) > (stars_b[-1] if stars_b else 0) else tech_b
            },
            "momentum": {
                "tech_a_momentum": _calculate_momentum_from_scores(scores_a),
                "tech_b_momentum": _calculate_momentum_from_scores(scores_b),
                "winner": tech_a if _calculate_momentum_from_scores(scores_a) > _calculate_momentum_from_scores(scores_b) else tech_b
            },
            "volatility": {
                "tech_a_volatility": _calculate_volatility_from_scores(scores_a),
                "tech_b_volatility": _calculate_volatility_from_scores(scores_b),
                "more_stable": tech_a if _calculate_volatility_from_scores(scores_a) < _calculate_volatility_from_scores(scores_b) else tech_b
            }
        }
        
        # Generate recommendation
        recommendation = _generate_comparison_recommendation(comparison_metrics, tech_a, tech_b)
        
        return ComparisonResponse(
            technology_a=tech_a,
            technology_b=tech_b,
            comparison_metrics=comparison_metrics,
            recommendation=recommendation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing technologies: {str(e)}")

@router.get("/insights", response_model=List[InsightResponse])
async def get_insights(
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Get AI-generated insights about technology trends"""
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        insights = await _generate_insights(db, start_date, limit)
        return insights
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching insights: {str(e)}")

@router.get("/category/{category}/analytics", response_model=CategoryAnalyticsResponse)
async def get_category_analytics(
    category: str,
    days: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get detailed analytics for a specific category"""
    try:
        if category not in TECHNOLOGY_CATEGORIES:
            raise HTTPException(status_code=404, detail="Category not found")
        
        start_date = datetime.utcnow() - timedelta(days=days)
        analytics = await _get_category_analytics(db, category, start_date)
        
        if not analytics:
            raise HTTPException(status_code=404, detail="No data found for this category")
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching category analytics: {str(e)}")

# Helper functions
async def _calculate_momentum(db: Session, tech_name: str, days: int) -> float:
    """Calculate momentum for a technology"""
    try:
        tech = db.query(Technology).filter(Technology.name == tech_name).first()
        if not tech:
            return 0.0
        
        # Get recent and older data
        recent_date = datetime.utcnow() - timedelta(days=days//2)
        older_date = datetime.utcnow() - timedelta(days=days)
        
        recent_avg = db.query(func.avg(TrendData.trend_score)).filter(
            TrendData.technology_id == tech.id,
            TrendData.date >= recent_date
        ).scalar()
        
        older_avg = db.query(func.avg(TrendData.trend_score)).filter(
            TrendData.technology_id == tech.id,
            TrendData.date >= older_date,
            TrendData.date < recent_date
        ).scalar()
        
        if recent_avg and older_avg:
            return float(recent_avg - older_avg)
        return 0.0
        
    except Exception:
        return 0.0

async def _calculate_volatility(db: Session, tech_name: str, days: int) -> float:
    """Calculate volatility for a technology"""
    try:
        tech = db.query(Technology).filter(Technology.name == tech_name).first()
        if not tech:
            return 0.0
        
        start_date = datetime.utcnow() - timedelta(days=days)
        scores = db.query(TrendData.trend_score).filter(
            TrendData.technology_id == tech.id,
            TrendData.date >= start_date
        ).all()
        
        if len(scores) < 2:
            return 0.0
        
        score_values = [float(score[0]) for score in scores]
        return _calculate_volatility_from_scores(score_values)
        
    except Exception:
        return 0.0

def _calculate_volatility_from_scores(scores: List[float]) -> float:
    """Calculate volatility from a list of scores"""
    if len(scores) < 2:
        return 0.0
    
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    return variance ** 0.5

def _calculate_momentum_from_scores(scores: List[float]) -> float:
    """Calculate momentum from a list of scores"""
    if len(scores) < 2:
        return 0.0
    
    # Simple momentum: difference between last and first score
    return scores[-1] - scores[0]

def _determine_trend_direction(scores: List[float]) -> str:
    """Determine trend direction from scores"""
    if len(scores) < 2:
        return "stable"
    
    momentum = _calculate_momentum_from_scores(scores)
    
    if momentum > 0.1:
        return "up"
    elif momentum < -0.1:
        return "down"
    else:
        return "stable"

async def _calculate_percentile_rank(db: Session, tech_name: str, score: float) -> float:
    """Calculate percentile rank for a technology"""
    try:
        # Get all scores for comparison
        all_scores = db.query(TrendData.trend_score).filter(
            TrendData.date >= datetime.utcnow() - timedelta(days=7)
        ).all()
        
        if not all_scores:
            return 0.0
        
        score_values = [float(s[0]) for s in all_scores]
        score_values.sort()
        
        # Find position of the score
        count_below = sum(1 for s in score_values if s < score)
        percentile = (count_below / len(score_values)) * 100
        
        return percentile
        
    except Exception:
        return 0.0

async def _get_category_ranking(db: Session, tech_id: int, category: str) -> int:
    """Get ranking of technology within its category"""
    try:
        rankings = db.query(Technology.id, func.avg(TrendData.trend_score)).join(TrendData).filter(
            Technology.category == category,
            TrendData.date >= datetime.utcnow() - timedelta(days=7)
        ).group_by(Technology.id).order_by(
            func.avg(TrendData.trend_score).desc()
        ).all()
        
        for i, (tid, _) in enumerate(rankings):
            if tid == tech_id:
                return i + 1
        
        return len(rankings)
        
    except Exception:
        return 0

async def _find_similar_technologies(db: Session, tech_id: int, category: str) -> List[str]:
    """Find similar technologies in the same category"""
    try:
        similar = db.query(Technology.name).filter(
            Technology.category == category,
            Technology.id != tech_id
        ).limit(5).all()
        
        return [tech[0] for tech in similar]
        
    except Exception:
        return []

async def _get_category_analytics(db: Session, category: str, start_date: datetime) -> Optional[CategoryAnalyticsResponse]:
    """Get analytics for a specific category"""
    try:
        # Get technologies in category
        technologies = db.query(Technology).filter(Technology.category == category).all()
        
        if not technologies:
            return None
        
        total_technologies = len(technologies)
        tech_ids = [tech.id for tech in technologies]
        
        # Average trend score
        avg_score = db.query(func.avg(TrendData.trend_score)).filter(
            TrendData.technology_id.in_(tech_ids),
            TrendData.date >= start_date
        ).scalar()
        
        # Get top performers
        top_performers = db.query(Technology.name, func.avg(TrendData.trend_score)).join(TrendData).filter(
            Technology.category == category,
            TrendData.date >= start_date
        ).group_by(Technology.name).order_by(
            func.avg(TrendData.trend_score).desc()
        ).limit(5).all()
        
        top_performer_names = [perf[0] for perf in top_performers]
        
        # Simple growth rate calculation
        growth_rate = 0.0  # This would need historical comparison
        volatility = 0.0  # This would need proper calculation
        
        return CategoryAnalyticsResponse(
            category=category,
            total_technologies=total_technologies,
            average_trend_score=float(avg_score) if avg_score else 0.0,
            growth_rate=growth_rate,
            volatility=volatility,
            top_performers=top_performer_names
        )
        
    except Exception:
        return None

async def _generate_insights(db: Session, start_date: datetime, limit: int = 10) -> List[InsightResponse]:
    """Generate AI insights about technology trends"""
    insights = []
    
    try:
        # Insight 1: Rising stars
        rising_technologies = db.query(
            Technology.name,
            func.avg(TrendData.trend_score).label('avg_score')
        ).join(TrendData).filter(
            TrendData.date >= start_date
        ).group_by(Technology.name).order_by(
            func.avg(TrendData.trend_score).desc()
        ).limit(3).all()
        
        if rising_technologies:
            top_tech = rising_technologies[0]
            insights.append(InsightResponse(
                type="trend",
                title=f"Rising Star: {top_tech[0]}",
                description=f"{top_tech[0]} shows exceptional growth with an average trend score of {top_tech[1]:.3f}",
                confidence=0.85,
                data={"technology": top_tech[0], "score": float(top_tech[1])}
            ))
        
        # Insight 2: Category performance
        category_performance = db.query(
            Technology.category,
            func.avg(TrendData.trend_score).label('avg_score'),
            func.count(TrendData.id).label('activity')
        ).join(TrendData).filter(
            TrendData.date >= start_date
        ).group_by(Technology.category).order_by(
            func.avg(TrendData.trend_score).desc()
        ).first()
        
        if category_performance:
            insights.append(InsightResponse(
                type="trend",
                title=f"Leading Category: {category_performance[0]}",
                description=f"The {category_performance[0]} category leads with highest average trend score of {category_performance[1]:.3f}",
                confidence=0.75,
                data={"category": category_performance[0], "score": float(category_performance[1])}
            ))
        
        # Insight 3: Data collection status
        recent_collections = db.query(CollectionLog).filter(
            CollectionLog.collection_date >= start_date
        ).count()
        
        insights.append(InsightResponse(
            type="system",
            title="Data Collection Health",
            description=f"System has performed {recent_collections} data collection runs in the last period",
            confidence=0.95,
            data={"collection_count": recent_collections}
        ))
        
        return insights[:limit]
        
    except Exception as e:
        return [InsightResponse(
            type="error",
            title="Insight Generation Error",
            description=f"Error generating insights: {str(e)}",
            confidence=0.0,
            data={}
        )]

async def _get_time_series_data(db: Session, start_date: datetime) -> List[TimeSeriesResponse]:
    """Get time series data for analytics"""
    try:
        # Overall trend score over time
        daily_scores = db.query(
            func.date(TrendData.date).label('date'),
            func.avg(TrendData.trend_score).label('avg_score'),
            func.count(TrendData.id).label('tech_count')
        ).filter(
            TrendData.date >= start_date
        ).group_by(func.date(TrendData.date)).order_by('date').all()
        
        if not daily_scores:
            return []
        
        data_points = []
        for date_str, avg_score, tech_count in daily_scores:
            data_points.append(TimeSeriesDataPoint(
                date=datetime.strptime(str(date_str), '%Y-%m-%d'),
                value=float(avg_score),
                technology_count=tech_count
            ))
        
        # Determine trend direction
        values = [dp.value for dp in data_points]
        trend_direction = _determine_trend_direction(values)
        growth_rate = _calculate_momentum_from_scores(values)
        
        return [TimeSeriesResponse(
            label="Overall Trend Score",
            data=data_points,
            trend_direction=trend_direction,
            growth_rate=growth_rate
        )]
        
    except Exception:
        return []

def _generate_comparison_recommendation(comparison_metrics: Dict[str, Any], tech_a: str, tech_b: str) -> str:
    """Generate recommendation based on comparison metrics"""
    try:
        trend_winner = comparison_metrics['trend_scores']['winner']
        momentum_winner = comparison_metrics['momentum']['winner']
        stability_winner = comparison_metrics['volatility']['more_stable']
        
        if trend_winner == momentum_winner:
            return f"Recommendation: {trend_winner} shows both higher trend scores and better momentum, making it a strong choice for investment or adoption."
        elif trend_winner == stability_winner:
            return f"Recommendation: {trend_winner} combines high trend scores with stability, offering a balanced opportunity."
        else:
            return f"Recommendation: Mixed signals - {trend_winner} leads in trends, {momentum_winner} in momentum, and {stability_winner} in stability. Consider your specific priorities."
            
    except Exception:
        return "Unable to generate recommendation due to insufficient data." 