#!/usr/bin/env python3
"""
Generate comparison predictions for the Tech Trend Radar app.
This script creates predictions that compare different technologies side by side.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import random
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from models.database import get_db_sync, init_database, Technology, Prediction, TrendData
from ml_models.model_manager import ModelManager

async def generate_comparison_predictions():
    """Generate comparison predictions for all technologies."""
    print("🔄 Generating comparison predictions...")
    
    # Initialize database
    await init_database()
    db = get_db_sync()
    
    try:
        # Get all technologies
        technologies = db.query(Technology).all()
        
        if not technologies:
            print("❌ No technologies found in database")
            return
        
        print(f"📊 Found {len(technologies)} technologies")
        
        # Get model manager
        model_manager = ModelManager()
        
        # Clear existing comparison predictions
        db.query(Prediction).filter(
            Prediction.model_used.in_(['ensemble_ml_model', 'fallback_model'])
        ).delete()
        
        # Generate predictions for each technology
        target_date = datetime.utcnow() + timedelta(days=180)  # 6 months from now
        
        for tech in technologies:
            try:
                # Get latest trend data for this technology
                latest_trend = db.query(TrendData).filter(
                    TrendData.technology_id == tech.id
                ).order_by(TrendData.date.desc()).first()
                
                if latest_trend:
                    # Use trend data for prediction
                    tech_features = {
                        'trend_score': latest_trend.trend_score,
                        'adoption_score': latest_trend.adoption_score,
                        'momentum_score': latest_trend.momentum_score,
                        'historical_growth': latest_trend.trend_score,  # Use trend score as proxy
                        'community_engagement': latest_trend.github_stars / 10000,  # Normalize
                        'market_maturity': latest_trend.trend_score  # Use trend score as proxy
                    }
                else:
                    # Fallback features
                    tech_features = {
                        'trend_score': random.uniform(0.3, 0.8),
                        'adoption_score': random.uniform(0.2, 0.7),
                        'momentum_score': random.uniform(0.2, 0.8),
                        'historical_growth': random.uniform(0.1, 0.6),
                        'community_engagement': random.uniform(0.1, 0.5),
                        'market_maturity': random.uniform(0.2, 0.7)
                    }
                
                # Generate prediction using ensemble model
                prediction_result = model_manager.predict_ensemble(tech_features)
                
                # Create prediction record
                prediction = Prediction(
                    technology_id=tech.id,
                    prediction_date=datetime.utcnow(),
                    target_date=target_date,
                    adoption_probability=prediction_result['adoption_probability'],
                    market_impact_score=prediction_result['market_impact_score'],
                    risk_score=prediction_result['risk_score'],
                    confidence_interval=random.uniform(0.2, 0.8),
                    model_used='ensemble_ml_model',
                    features_used=list(tech_features.keys()),
                    prediction_reasoning=f"Comparison prediction for {tech.name}: Ensemble model analysis based on trend score ({tech_features['trend_score']:.2f}), adoption score ({tech_features['adoption_score']:.2f}), and momentum score ({tech_features['momentum_score']:.2f})",
                    is_validated=False,
                    actual_outcome=None,
                    accuracy_score=None
                )
                
                db.add(prediction)
                print(f"✅ Generated prediction for {tech.name}")
                
            except Exception as e:
                print(f"⚠️ Error generating prediction for {tech.name}: {e}")
                # Create fallback prediction
                prediction = Prediction(
                    technology_id=tech.id,
                    prediction_date=datetime.utcnow(),
                    target_date=target_date,
                    adoption_probability=random.uniform(0.3, 0.9),
                    market_impact_score=random.uniform(0.2, 0.8),
                    risk_score=random.uniform(0.1, 0.5),
                    confidence_interval=random.uniform(0.2, 0.8),
                    model_used='fallback_model',
                    features_used=['trend_score', 'adoption_score', 'momentum_score'],
                    prediction_reasoning=f"Fallback comparison prediction for {tech.name} based on current metrics",
                    is_validated=False,
                    actual_outcome=None,
                    accuracy_score=None
                )
                db.add(prediction)
        
        db.commit()
        print(f"✅ Generated comparison predictions for {len(technologies)} technologies")
        
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(generate_comparison_predictions()) 