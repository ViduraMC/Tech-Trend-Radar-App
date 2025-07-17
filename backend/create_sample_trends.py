#!/usr/bin/env python3
"""
Script to create sample trend and prediction data for Tech Trend Radar
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.database import get_db_sync, Technology, TrendData, Prediction
from datetime import datetime, timedelta
import random

def create_sample_trends_and_predictions():
    """Create sample trend and prediction data"""
    db = get_db_sync()
    
    # Get existing technologies
    technologies = db.query(Technology).all()
    
    if not technologies:
        print("❌ No technologies found. Please run create_sample_data.py first.")
        return
    
    print(f"📊 Creating sample trends and predictions for {len(technologies)} technologies...")
    
    # Create sample trend data
    for tech in technologies:
        # Create multiple trend entries for each technology
        for i in range(5):
            trend_date = datetime.utcnow() - timedelta(days=i*7)
            
            trend = TrendData(
                technology_id=tech.id,
                source="github",
                date=trend_date,
                github_stars=random.randint(1000, 50000),
                github_forks=random.randint(100, 5000),
                github_issues=random.randint(50, 500),
                arxiv_papers=random.randint(0, 15),
                patent_filings=random.randint(0, 20),
                job_postings=random.randint(10, 200),
                social_mentions=random.randint(5, 100),
                trend_score=random.uniform(0.1, 0.9),
                momentum_score=random.uniform(0.1, 0.8),
                adoption_score=random.uniform(0.1, 0.7)
            )
            db.add(trend)
            print(f"✅ Created trend for {tech.name} on {trend_date.strftime('%Y-%m-%d')}")
    
    # Create sample predictions
    for tech in technologies:
        # Create predictions for different time horizons
        for days_ahead in [30, 60, 90]:
            prediction_date = datetime.utcnow()
            target_date = datetime.utcnow() + timedelta(days=days_ahead)
            
            prediction = Prediction(
                technology_id=tech.id,
                prediction_date=prediction_date,
                target_date=target_date,
                adoption_probability=random.uniform(0.2, 0.9),
                market_impact_score=random.uniform(0.1, 0.8),
                risk_score=random.uniform(0.1, 0.7),
                confidence_interval=random.uniform(0.5, 0.95),
                model_used="ensemble_model_v1",
                features_used=["github_activity", "job_market", "research_papers", "patent_filings"],
                prediction_reasoning=f"Based on current {tech.category} trends and market analysis",
                is_validated=False
            )
            db.add(prediction)
            print(f"🔮 Created prediction for {tech.name} ({days_ahead} days ahead)")
    
    db.commit()
    
    # Show summary
    total_trends = db.query(TrendData).count()
    total_predictions = db.query(Prediction).count()
    
    print(f"\n📈 Summary:")
    print(f"   Total trends: {total_trends}")
    print(f"   Total predictions: {total_predictions}")
    print(f"   Technologies: {len(technologies)}")
    
    return total_trends, total_predictions

if __name__ == "__main__":
    print("🔧 Creating sample trends and predictions...")
    trends, predictions = create_sample_trends_and_predictions()
    print(f"\n🎉 Database populated with {trends} trends and {predictions} predictions!") 