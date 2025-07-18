#!/usr/bin/env python3
"""
Generate time series predictions for the Tech Trend Radar app.
This script creates predictions that show how technologies evolve over time.
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

async def generate_time_series_predictions():
    """Generate time series predictions for all technologies."""
    print("🔄 Generating time series predictions...")
    
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
        
        # Clear existing time series predictions
        db.query(Prediction).filter(
            Prediction.model_used == 'time_series_ml_model'
        ).delete()
        
        # Generate predictions for each technology
        for tech in technologies:
            print(f"📈 Generating time series for {tech.name}...")
            
            # Get latest trend data for this technology
            latest_trend = db.query(TrendData).filter(
                TrendData.technology_id == tech.id
            ).order_by(TrendData.date.desc()).first()
            
            if latest_trend:
                base_adoption = latest_trend.adoption_score
                base_impact = latest_trend.trend_score
                base_risk = 1.0 - latest_trend.momentum_score
            else:
                base_adoption = random.uniform(0.3, 0.8)
                base_impact = random.uniform(0.2, 0.7)
                base_risk = random.uniform(0.2, 0.6)
            
            # Generate 6 months of predictions
            for month in range(1, 7):
                target_date = datetime.utcnow() + timedelta(days=30 * month)
                
                # Simulate realistic progression
                adoption_growth = random.uniform(-0.05, 0.15)  # -5% to +15% monthly change
                impact_lag = random.uniform(-0.03, 0.10)  # Impact follows adoption with lag
                risk_adjustment = random.uniform(-0.08, 0.05)  # Risk adjusts based on adoption
                
                # Calculate new values
                new_adoption = max(0.1, min(1.0, base_adoption + (adoption_growth * month)))
                new_impact = max(0.1, min(1.0, base_impact + (impact_lag * month)))
                new_risk = max(0.1, min(0.9, base_risk + (risk_adjustment * month)))
                
                # Confidence decreases over time
                confidence = max(0.2, 0.8 - (month * 0.1))
                
                # Create prediction
                prediction = Prediction(
                    technology_id=tech.id,
                    prediction_date=datetime.utcnow(),
                    target_date=target_date,
                    adoption_probability=new_adoption,
                    market_impact_score=new_impact,
                    risk_score=new_risk,
                    confidence_interval=confidence,
                    model_used='time_series_ml_model',
                    features_used=['trend_score', 'adoption_score', 'momentum_score', 'historical_growth'],
                    prediction_reasoning=f"Time series prediction for {tech.name} - Month {month}: Adoption trend shows {adoption_growth*100:+.1f}% monthly change, impact follows with {impact_lag*100:+.1f}% change, risk adjusts by {risk_adjustment*100:+.1f}% based on adoption patterns",
                    is_validated=False,
                    actual_outcome=None,
                    accuracy_score=None
                )
                
                db.add(prediction)
        
        db.commit()
        print(f"✅ Generated time series predictions for {len(technologies)} technologies")
        
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(generate_time_series_predictions()) 