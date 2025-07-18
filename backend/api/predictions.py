from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from models.database import get_db, Technology, Prediction, TrendData
from config import settings
from ml_models.model_manager import ModelManager

router = APIRouter()

# Initialize model manager
model_manager = ModelManager()

# Pydantic models
class PredictionResponse(BaseModel):
    id: int
    technology_id: int
    prediction_date: datetime
    target_date: datetime
    adoption_probability: float
    market_impact_score: float
    risk_score: float
    confidence_interval: float
    model_used: str
    features_used: List[str] = []
    prediction_reasoning: Optional[str] = None
    is_validated: bool = False
    actual_outcome: Optional[float] = None
    accuracy_score: Optional[float] = None
    
    class Config:
        from_attributes = True

class TechnologyPredictionResponse(BaseModel):
    technology: Dict[str, Any]
    predictions: List[PredictionResponse]
    current_trend_score: float
    predicted_trend_score: float
    confidence_level: str

class PredictionSummaryResponse(BaseModel):
    total_predictions: int
    validated_predictions: int
    average_accuracy: float
    high_confidence_predictions: int
    predictions_by_category: Dict[str, int]
    recent_predictions: List[PredictionResponse]

class PredictionsListResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int

# Simple endpoint for frontend
@router.get("/", response_model=PredictionsListResponse)
async def get_predictions_list(
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get list of predictions for frontend"""
    try:
        # Get recent predictions
        predictions_query = db.query(Prediction).order_by(
            Prediction.prediction_date.desc()
        ).limit(limit)
        
        predictions = predictions_query.all()
        
        # Convert to response format
        prediction_responses = []
        for prediction in predictions:
            prediction_responses.append(PredictionResponse(
                id=prediction.id,
                technology_id=prediction.technology_id,
                prediction_date=prediction.prediction_date,
                target_date=prediction.target_date,
                adoption_probability=prediction.adoption_probability,
                market_impact_score=prediction.market_impact_score,
                risk_score=prediction.risk_score,
                confidence_interval=prediction.confidence_interval,
                model_used=prediction.model_used,
                features_used=prediction.features_used,
                prediction_reasoning=prediction.prediction_reasoning,
                is_validated=prediction.is_validated,
                actual_outcome=prediction.actual_outcome,
                accuracy_score=prediction.accuracy_score
            ))
        
        return PredictionsListResponse(
            predictions=prediction_responses,
            total=len(prediction_responses)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching predictions: {str(e)}")

@router.get("/technology/{tech_id}", response_model=TechnologyPredictionResponse)
async def get_technology_predictions(
    tech_id: int,
    db: Session = Depends(get_db)
):
    """Get predictions for a specific technology"""
    try:
        # Get technology
        technology = db.query(Technology).filter(Technology.id == tech_id).first()
        if not technology:
            raise HTTPException(status_code=404, detail="Technology not found")
        
        # Get predictions
        predictions = db.query(Prediction).filter(
            Prediction.technology_id == tech_id
        ).order_by(Prediction.prediction_date.desc()).all()
        
        # Get current trend score
        latest_trend = db.query(TrendData).filter(
            TrendData.technology_id == tech_id
        ).order_by(TrendData.date.desc()).first()
        
        current_trend_score = latest_trend.trend_score if latest_trend else 0.0
        
        # Calculate predicted trend score (simplified)
        predicted_trend_score = current_trend_score
        confidence_level = "medium"
        
        if predictions:
            latest_prediction = predictions[0]
            predicted_trend_score = latest_prediction.adoption_probability
            
            if latest_prediction.confidence_interval > 0.8:
                confidence_level = "high"
            elif latest_prediction.confidence_interval < 0.5:
                confidence_level = "low"
        
        return TechnologyPredictionResponse(
            technology={
                "id": technology.id,
                "name": technology.name,
                "category": technology.category,
                "description": technology.description
            },
            predictions=[PredictionResponse.from_orm(p) for p in predictions],
            current_trend_score=current_trend_score,
            predicted_trend_score=predicted_trend_score,
            confidence_level=confidence_level
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching technology predictions: {str(e)}")

@router.get("/category/{category}", response_model=List[TechnologyPredictionResponse])
async def get_category_predictions(
    category: str,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get predictions for technologies in a specific category"""
    try:
        # Get technologies in category
        technologies = db.query(Technology).filter(
            Technology.category == category
        ).limit(limit).all()
        
        if not technologies:
            raise HTTPException(status_code=404, detail="Category not found")
        
        results = []
        for tech in technologies:
            # Get predictions for this technology
            predictions = db.query(Prediction).filter(
                Prediction.technology_id == tech.id
            ).order_by(Prediction.prediction_date.desc()).all()
            
            # Get current trend score
            latest_trend = db.query(TrendData).filter(
                TrendData.technology_id == tech.id
            ).order_by(TrendData.date.desc()).first()
            
            current_trend_score = latest_trend.trend_score if latest_trend else 0.0
            
            # Calculate predicted trend score and confidence
            predicted_trend_score = current_trend_score
            confidence_level = "medium"
            
            if predictions:
                latest_prediction = predictions[0]
                predicted_trend_score = latest_prediction.adoption_probability
                
                if latest_prediction.confidence_interval > 0.8:
                    confidence_level = "high"
                elif latest_prediction.confidence_interval < 0.5:
                    confidence_level = "low"
            
            results.append(TechnologyPredictionResponse(
                technology={
                    "id": tech.id,
                    "name": tech.name,
                    "category": tech.category,
                    "description": tech.description
                },
                predictions=[PredictionResponse.from_orm(p) for p in predictions],
                current_trend_score=current_trend_score,
                predicted_trend_score=predicted_trend_score,
                confidence_level=confidence_level
            ))
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching category predictions: {str(e)}")

@router.post("/generate/{tech_id}")
async def generate_prediction(
    tech_id: int,
    target_days: int = Query(default=90, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Generate a new prediction for a technology using ML models"""
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
            raise HTTPException(status_code=400, detail="No trend data available for prediction")
        
        # Prepare technology data for ML prediction
        technology_data = {
            'name': technology.name,
            'category': technology.category,
            'description': technology.description or '',
            'keywords': technology.keywords or [],
            'trend_score': latest_trend.trend_score,
            'github_stars': latest_trend.github_stars,
            'github_forks': latest_trend.github_forks,
            'github_issues': latest_trend.github_issues,
            'arxiv_papers': latest_trend.arxiv_papers,
            'patent_filings': latest_trend.patent_filings,
            'job_postings': latest_trend.job_postings,
            'social_mentions': latest_trend.social_mentions,
            'momentum_score': latest_trend.momentum_score,
            'adoption_score': latest_trend.adoption_score
        }
        
        # Try to use ML models if available
        try:
            if model_manager.is_initialized:
                # Use ML model for prediction
                ml_prediction = model_manager.predict_technology(technology_data)
                
                # Create prediction record
                target_date = datetime.utcnow() + timedelta(days=target_days)
                
                prediction = Prediction(
                    technology_id=tech_id,
                    target_date=target_date,
                    adoption_probability=ml_prediction['predictions'].get('adoption_probability', 0.5),
                    market_impact_score=ml_prediction['predictions'].get('market_impact_score', 0.5),
                    risk_score=ml_prediction['predictions'].get('risk_score', 0.5),
                    confidence_interval=ml_prediction['confidence'].get('overall', 0.5),
                    model_used="ensemble_ml_model",
                    features_used=["trend_score", "github_stars", "github_forks", "github_issues", 
                                 "arxiv_papers", "patent_filings", "job_postings", "social_mentions"],
                    prediction_reasoning='; '.join(ml_prediction.get('recommendations', []))
                )
                
                db.add(prediction)
                db.commit()
                db.refresh(prediction)
                
                return {
                    "message": "ML prediction generated successfully",
                    "prediction": PredictionResponse.from_orm(prediction),
                    "ml_insights": {
                        "confidence": ml_prediction['confidence'],
                        "recommendations": ml_prediction['recommendations']
                    }
                }
            else:
                # Fallback to simple prediction
                raise ValueError("ML models not initialized")
                
        except Exception as ml_error:
            logger.warning(f"ML prediction failed, using fallback: {ml_error}")
            
            # Fallback to simple prediction logic
            historical_data = db.query(TrendData).filter(
                TrendData.technology_id == tech_id
            ).order_by(TrendData.date.desc()).limit(30).all()
            
            recent_trends = [data.trend_score for data in historical_data[:7]]
            avg_recent_trend = sum(recent_trends) / len(recent_trends) if recent_trends else 0.0
            
            # Calculate trend momentum
            if len(historical_data) >= 14:
                older_trends = [data.trend_score for data in historical_data[7:14]]
                avg_older_trend = sum(older_trends) / len(older_trends) if older_trends else 0.0
                momentum = avg_recent_trend - avg_older_trend
            else:
                momentum = 0.0
            
            # Generate prediction values
            adoption_probability = min(max(avg_recent_trend + (momentum * 0.1), 0.0), 1.0)
            market_impact_score = min(max(avg_recent_trend * 1.2, 0.0), 1.0)
            risk_score = max(0.1, 1.0 - adoption_probability)
            confidence_interval = min(0.9, 0.5 + (len(historical_data) / 100))
            
            # Create prediction
            target_date = datetime.utcnow() + timedelta(days=target_days)
            
            prediction = Prediction(
                technology_id=tech_id,
                target_date=target_date,
                adoption_probability=adoption_probability,
                market_impact_score=market_impact_score,
                risk_score=risk_score,
                confidence_interval=confidence_interval,
                model_used="simple_trend_analysis",
                features_used=["trend_score", "momentum", "historical_data"],
                prediction_reasoning=f"Based on {len(historical_data)} historical data points. "
                                   f"Recent trend: {avg_recent_trend:.3f}, Momentum: {momentum:.3f}"
            )
            
            db.add(prediction)
            db.commit()
            db.refresh(prediction)
            
            return {
                "message": "Fallback prediction generated successfully",
                "prediction": PredictionResponse.from_orm(prediction)
            }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error generating prediction: {str(e)}")

@router.get("/validate/{prediction_id}")
async def validate_prediction(
    prediction_id: int,
    db: Session = Depends(get_db)
):
    """Validate a prediction against actual outcomes"""
    try:
        # Get prediction
        prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        # Check if target date has passed
        if prediction.target_date > datetime.utcnow():
            raise HTTPException(status_code=400, detail="Target date has not yet been reached")
        
        # Get actual trend data around target date
        actual_trend = db.query(TrendData).filter(
            TrendData.technology_id == prediction.technology_id,
            TrendData.date >= prediction.target_date - timedelta(days=7),
            TrendData.date <= prediction.target_date + timedelta(days=7)
        ).order_by(TrendData.date.desc()).first()
        
        if not actual_trend:
            raise HTTPException(status_code=400, detail="No actual trend data available for validation")
        
        # Calculate accuracy
        predicted_value = prediction.adoption_probability
        actual_value = actual_trend.trend_score
        accuracy = 1.0 - abs(predicted_value - actual_value)
        
        # Update prediction
        prediction.is_validated = True
        prediction.actual_outcome = actual_value
        prediction.accuracy_score = accuracy
        
        db.commit()
        
        return {
            "message": "Prediction validated successfully",
            "predicted_value": predicted_value,
            "actual_value": actual_value,
            "accuracy_score": accuracy
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error validating prediction: {str(e)}")

@router.get("/accuracy", response_model=Dict[str, Any])
async def get_prediction_accuracy(db: Session = Depends(get_db)):
    """Get overall prediction accuracy statistics"""
    try:
        validated_predictions = db.query(Prediction).filter(
            Prediction.is_validated == True,
            Prediction.accuracy_score.isnot(None)
        ).all()
        
        if not validated_predictions:
            return {
                "total_validated": 0,
                "average_accuracy": 0.0,
                "accuracy_by_model": {},
                "accuracy_by_category": {},
                "high_accuracy_count": 0
            }
        
        total_validated = len(validated_predictions)
        average_accuracy = sum(p.accuracy_score for p in validated_predictions) / total_validated
        
        # Accuracy by model
        accuracy_by_model = {}
        for prediction in validated_predictions:
            model = prediction.model_used
            if model not in accuracy_by_model:
                accuracy_by_model[model] = []
            accuracy_by_model[model].append(prediction.accuracy_score)
        
        for model in accuracy_by_model:
            scores = accuracy_by_model[model]
            accuracy_by_model[model] = sum(scores) / len(scores)
        
        # Accuracy by category
        accuracy_by_category = {}
        for prediction in validated_predictions:
            # Get technology category
            tech = db.query(Technology).filter(Technology.id == prediction.technology_id).first()
            if tech:
                category = tech.category
                if category not in accuracy_by_category:
                    accuracy_by_category[category] = []
                accuracy_by_category[category].append(prediction.accuracy_score)
        
        for category in accuracy_by_category:
            scores = accuracy_by_category[category]
            accuracy_by_category[category] = sum(scores) / len(scores)
        
        # High accuracy predictions (>0.8)
        high_accuracy_count = sum(1 for p in validated_predictions if p.accuracy_score > 0.8)
        
        return {
            "total_validated": total_validated,
            "average_accuracy": average_accuracy,
            "accuracy_by_model": accuracy_by_model,
            "accuracy_by_category": accuracy_by_category,
            "high_accuracy_count": high_accuracy_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching prediction accuracy: {str(e)}")

@router.post("/ml/initialize")
async def initialize_ml_models(db: Session = Depends(get_db)):
    """Initialize and train ML models"""
    try:
        result = model_manager.initialize_models(db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing ML models: {str(e)}")

@router.get("/ml/status")
async def get_ml_model_status():
    """Get ML model status"""
    try:
        return model_manager.get_model_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting ML model status: {str(e)}")

@router.post("/ml/retrain")
async def retrain_ml_models(db: Session = Depends(get_db)):
    """Retrain ML models with fresh data"""
    try:
        result = model_manager.retrain_models(db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retraining ML models: {str(e)}")

@router.post("/ml/predict")
async def predict_with_ml(
    technology_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Make ML prediction for a technology"""
    try:
        if not model_manager.is_initialized:
            raise HTTPException(status_code=400, detail="ML models not initialized")
        
        prediction = model_manager.predict_technology(technology_data)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making ML prediction: {str(e)}")

@router.get("/ml/feature-importance")
async def get_feature_importance(model_name: str = None):
    """Get feature importance for ML models"""
    try:
        return model_manager.get_feature_importance(model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feature importance: {str(e)}") 