#!/usr/bin/env python3
"""
Create more comprehensive sample data for ML model training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import random
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from models.database import get_db_sync, init_database, Technology, TrendData, Prediction
from config import settings

async def create_comprehensive_sample_data():
    """Create comprehensive sample data for ML training"""
    print("🔧 Creating comprehensive sample data for ML training...")
    
    # Initialize database
    await init_database()
    db = get_db_sync()
    
    try:
        # Create more technologies
        technologies_data = [
            {
                'name': 'Angular',
                'category': 'Frontend',
                'description': 'Platform for building mobile and desktop web applications',
                'keywords': ['typescript', 'framework', 'google', 'spa']
            },
            {
                'name': 'Node.js',
                'category': 'Backend',
                'description': 'JavaScript runtime built on Chrome V8 JavaScript engine',
                'keywords': ['javascript', 'runtime', 'server', 'npm']
            },
            {
                'name': 'Python',
                'category': 'Backend',
                'description': 'High-level programming language for general-purpose programming',
                'keywords': ['programming', 'language', 'data-science', 'web']
            },
            {
                'name': 'AWS',
                'category': 'Cloud',
                'description': 'Amazon Web Services cloud computing platform',
                'keywords': ['cloud', 'amazon', 'infrastructure', 'services']
            },
            {
                'name': 'PyTorch',
                'category': 'AI/ML',
                'description': 'Open source machine learning framework',
                'keywords': ['deep-learning', 'neural-networks', 'facebook', 'research']
            },
            {
                'name': 'Flutter',
                'category': 'Mobile',
                'description': 'UI toolkit for building natively compiled applications',
                'keywords': ['mobile', 'dart', 'google', 'cross-platform']
            },
            {
                'name': 'Rust',
                'category': 'Backend',
                'description': 'Systems programming language focused on safety and performance',
                'keywords': ['systems', 'performance', 'memory-safety', 'mozilla']
            },
            {
                'name': 'GraphQL',
                'category': 'Backend',
                'description': 'Query language and runtime for APIs',
                'keywords': ['api', 'query', 'facebook', 'data-fetching']
            },
            {
                'name': 'MongoDB',
                'category': 'Data',
                'description': 'Document-oriented NoSQL database',
                'keywords': ['database', 'nosql', 'document', 'scalable']
            },
            {
                'name': 'Redis',
                'category': 'Data',
                'description': 'In-memory data structure store',
                'keywords': ['cache', 'memory', 'key-value', 'fast']
            }
        ]
        
        # Add technologies
        for tech_data in technologies_data:
            existing = db.query(Technology).filter(Technology.name == tech_data['name']).first()
            if not existing:
                tech = Technology(
                    name=tech_data['name'],
                    category=tech_data['category'],
                    description=tech_data['description'],
                    keywords=tech_data['keywords'],
                    first_detected=datetime.utcnow() - timedelta(days=random.randint(100, 1000)),
                    last_updated=datetime.utcnow()
                )
                db.add(tech)
                print(f"✅ Added: {tech_data['name']}")
            else:
                print(f"⚠️  Exists: {tech_data['name']}")
        
        db.commit()
        
        # Get all technologies
        technologies = db.query(Technology).all()
        print(f"\n📊 Total technologies in database: {len(technologies)}")
        
        # Create trend data for each technology (multiple entries for time series)
        for tech in technologies:
            print(f"📈 Creating trend data for {tech.name}...")
            
            # Create 30 days of trend data
            for i in range(30):
                date = datetime.utcnow() - timedelta(days=30-i)
                
                # Generate realistic trend data
                base_trend = random.uniform(0.3, 0.8)
                trend_score = base_trend + random.uniform(-0.1, 0.1)
                trend_score = max(0.0, min(1.0, trend_score))
                
                github_stars = random.randint(1000, 50000)
                github_forks = int(github_stars * random.uniform(0.1, 0.3))
                github_issues = random.randint(10, 500)
                arxiv_papers = random.randint(0, 20)
                patent_filings = random.randint(0, 10)
                job_postings = random.randint(50, 2000)
                social_mentions = random.randint(100, 5000)
                
                momentum_score = trend_score + random.uniform(-0.05, 0.05)
                momentum_score = max(0.0, min(1.0, momentum_score))
                
                adoption_score = trend_score * random.uniform(0.8, 1.2)
                adoption_score = max(0.0, min(1.0, adoption_score))
                
                trend_data = TrendData(
                    technology_id=tech.id,
                    source="sample_data",
                    date=date,
                    trend_score=trend_score,
                    github_stars=github_stars,
                    github_forks=github_forks,
                    github_issues=github_issues,
                    arxiv_papers=arxiv_papers,
                    patent_filings=patent_filings,
                    job_postings=job_postings,
                    social_mentions=social_mentions,
                    momentum_score=momentum_score,
                    adoption_score=adoption_score
                )
                db.add(trend_data)
            
            # Create prediction data
            prediction = Prediction(
                technology_id=tech.id,
                target_date=datetime.utcnow() + timedelta(days=90),
                adoption_probability=random.uniform(0.3, 0.9),
                market_impact_score=random.uniform(0.2, 0.8),
                risk_score=random.uniform(0.1, 0.7),
                confidence_interval=random.uniform(0.6, 0.9),
                model_used="sample_data",
                features_used=["trend_score", "github_stars", "social_mentions"],
                prediction_reasoning="Sample prediction for ML training"
            )
            db.add(prediction)
        
        db.commit()
        
        # Print summary
        total_technologies = db.query(Technology).count()
        total_trends = db.query(TrendData).count()
        total_predictions = db.query(Prediction).count()
        
        print(f"\n🎉 Comprehensive sample data created!")
        print(f"📊 Technologies: {total_technologies}")
        print(f"📈 Trend data points: {total_trends}")
        print(f"🔮 Predictions: {total_predictions}")
        print(f"📅 Data spans: 30 days of historical data")
        
        # Show some examples
        print(f"\n📋 Sample technologies:")
        for tech in technologies[:5]:
            print(f"  - {tech.name} ({tech.category})")
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(create_comprehensive_sample_data()) 