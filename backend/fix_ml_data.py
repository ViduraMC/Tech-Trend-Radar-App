#!/usr/bin/env python3
"""
Fix ML data by adding more diverse technologies
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

async def fix_ml_data():
    """Add more diverse technologies to fix ML training"""
    print("🔧 Fixing ML data with more diverse technologies...")
    
    # Initialize database
    await init_database()
    db = get_db_sync()
    
    try:
        # Create more diverse technologies
        technologies_data = [
            # AI/ML - Add more
            {
                'name': 'Scikit-learn',
                'category': 'AI/ML',
                'description': 'Machine learning library for Python',
                'keywords': ['machine-learning', 'python', 'scikit', 'data-science']
            },
            {
                'name': 'Keras',
                'category': 'AI/ML',
                'description': 'Deep learning API for Python',
                'keywords': ['deep-learning', 'neural-networks', 'tensorflow', 'python']
            },
            {
                'name': 'OpenAI GPT',
                'category': 'AI/ML',
                'description': 'Large language model for natural language processing',
                'keywords': ['nlp', 'language-model', 'openai', 'transformer']
            },
            
            # Cloud - Add more
            {
                'name': 'Google Cloud',
                'category': 'Cloud',
                'description': 'Google Cloud Platform services',
                'keywords': ['cloud', 'google', 'infrastructure', 'services']
            },
            {
                'name': 'Azure',
                'category': 'Cloud',
                'description': 'Microsoft Azure cloud platform',
                'keywords': ['cloud', 'microsoft', 'azure', 'infrastructure']
            },
            {
                'name': 'DigitalOcean',
                'category': 'Cloud',
                'description': 'Cloud infrastructure provider',
                'keywords': ['cloud', 'vps', 'droplets', 'infrastructure']
            },
            
            # Frontend - Add more
            {
                'name': 'Svelte',
                'category': 'Frontend',
                'description': 'Frontend framework for building web applications',
                'keywords': ['frontend', 'framework', 'javascript', 'reactive']
            },
            {
                'name': 'Next.js',
                'category': 'Frontend',
                'description': 'React framework for production',
                'keywords': ['react', 'framework', 'ssr', 'next']
            },
            {
                'name': 'Nuxt.js',
                'category': 'Frontend',
                'description': 'Vue.js framework for production',
                'keywords': ['vue', 'framework', 'ssr', 'nuxt']
            },
            
            # Backend - Add more
            {
                'name': 'Django',
                'category': 'Backend',
                'description': 'High-level Python web framework',
                'keywords': ['python', 'web-framework', 'django', 'mvc']
            },
            {
                'name': 'Flask',
                'category': 'Backend',
                'description': 'Lightweight Python web framework',
                'keywords': ['python', 'web-framework', 'flask', 'micro']
            },
            {
                'name': 'Express.js',
                'category': 'Backend',
                'description': 'Web application framework for Node.js',
                'keywords': ['nodejs', 'web-framework', 'express', 'javascript']
            },
            
            # Mobile - Add more
            {
                'name': 'React Native',
                'category': 'Mobile',
                'description': 'Mobile app development framework',
                'keywords': ['mobile', 'react', 'cross-platform', 'javascript']
            },
            {
                'name': 'Xamarin',
                'category': 'Mobile',
                'description': 'Cross-platform mobile development',
                'keywords': ['mobile', 'csharp', 'microsoft', 'cross-platform']
            },
            {
                'name': 'Ionic',
                'category': 'Mobile',
                'description': 'Cross-platform mobile app development',
                'keywords': ['mobile', 'web-technologies', 'angular', 'cross-platform']
            },
            
            # Data - Add more
            {
                'name': 'PostgreSQL',
                'category': 'Data',
                'description': 'Advanced open source database',
                'keywords': ['database', 'sql', 'postgresql', 'relational']
            },
            {
                'name': 'MySQL',
                'category': 'Data',
                'description': 'Open source relational database',
                'keywords': ['database', 'sql', 'mysql', 'relational']
            },
            {
                'name': 'Apache Kafka',
                'category': 'Data',
                'description': 'Distributed streaming platform',
                'keywords': ['streaming', 'kafka', 'apache', 'real-time']
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
                print(f"✅ Added: {tech_data['name']} ({tech_data['category']})")
            else:
                print(f"⚠️  Exists: {tech_data['name']}")
        
        db.commit()
        
        # Get all technologies
        technologies = db.query(Technology).all()
        print(f"\n📊 Total technologies in database: {len(technologies)}")
        
        # Show category distribution
        categories = {}
        for tech in technologies:
            cat = tech.category
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\n📋 Category distribution:")
        for cat, count in categories.items():
            print(f"  - {cat}: {count} technologies")
        
        # Create trend data for new technologies
        new_technologies = [tech for tech in technologies if tech.name in [t['name'] for t in technologies_data]]
        
        for tech in new_technologies:
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
        
        print(f"\n🎉 ML data fixed!")
        print(f"📊 Technologies: {total_technologies}")
        print(f"📈 Trend data points: {total_trends}")
        print(f"🔮 Predictions: {total_predictions}")
        
    except Exception as e:
        print(f"❌ Error fixing ML data: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(fix_ml_data()) 