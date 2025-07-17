#!/usr/bin/env python3
"""
Test script for Tech Trend Radar backend
This script will:
1. Initialize the database
2. Create sample technologies
3. Add sample trend data
4. Test API endpoints
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import random
from models.database import init_database, get_db_sync, create_technology, add_trend_data, create_prediction
from config import TECHNOLOGY_CATEGORIES

async def initialize_test_data():
    """Initialize database with sample data"""
    print("🔧 Initializing database...")
    await init_database()
    
    db = get_db_sync()
    
    # Sample technologies for each category
    sample_technologies = {
        "AI/ML": [
            {"name": "TensorFlow", "description": "Open-source machine learning framework", "keywords": ["tensorflow", "ml", "ai", "google"]},
            {"name": "PyTorch", "description": "Deep learning framework", "keywords": ["pytorch", "deep learning", "facebook"]},
            {"name": "Transformers", "description": "State-of-the-art NLP library", "keywords": ["transformers", "nlp", "bert", "gpt"]},
            {"name": "OpenAI GPT", "description": "Large language model", "keywords": ["gpt", "openai", "llm", "chat"]},
            {"name": "Stable Diffusion", "description": "AI image generation", "keywords": ["stable diffusion", "ai art", "image generation"]},
        ],
        "Blockchain": [
            {"name": "Ethereum", "description": "Decentralized platform", "keywords": ["ethereum", "smart contracts", "defi"]},
            {"name": "Solana", "description": "High-performance blockchain", "keywords": ["solana", "sol", "fast blockchain"]},
            {"name": "Polygon", "description": "Ethereum scaling solution", "keywords": ["polygon", "matic", "layer 2"]},
            {"name": "Chainlink", "description": "Decentralized oracle network", "keywords": ["chainlink", "oracle", "link"]},
        ],
        "Cloud": [
            {"name": "Kubernetes", "description": "Container orchestration", "keywords": ["kubernetes", "k8s", "containers"]},
            {"name": "Docker", "description": "Container platform", "keywords": ["docker", "containers", "devops"]},
            {"name": "Terraform", "description": "Infrastructure as code", "keywords": ["terraform", "iac", "cloud"]},
            {"name": "AWS Lambda", "description": "Serverless computing", "keywords": ["aws", "lambda", "serverless"]},
        ],
        "Frontend": [
            {"name": "React", "description": "JavaScript library for UIs", "keywords": ["react", "javascript", "ui", "facebook"]},
            {"name": "Vue.js", "description": "Progressive JavaScript framework", "keywords": ["vue", "vuejs", "javascript"]},
            {"name": "Next.js", "description": "React framework", "keywords": ["nextjs", "react", "ssr"]},
            {"name": "Svelte", "description": "Compile-time framework", "keywords": ["svelte", "javascript", "compiler"]},
        ],
        "Backend": [
            {"name": "FastAPI", "description": "Modern Python web framework", "keywords": ["fastapi", "python", "api"]},
            {"name": "Node.js", "description": "JavaScript runtime", "keywords": ["nodejs", "javascript", "backend"]},
            {"name": "Django", "description": "Python web framework", "keywords": ["django", "python", "web"]},
            {"name": "Go", "description": "Programming language", "keywords": ["golang", "go", "google"]},
        ],
        "Mobile": [
            {"name": "Flutter", "description": "Cross-platform mobile framework", "keywords": ["flutter", "dart", "mobile"]},
            {"name": "React Native", "description": "Mobile app framework", "keywords": ["react native", "mobile", "facebook"]},
            {"name": "Swift", "description": "iOS programming language", "keywords": ["swift", "ios", "apple"]},
            {"name": "Kotlin", "description": "Android programming language", "keywords": ["kotlin", "android", "jetbrains"]},
        ]
    }
    
    print("📊 Creating sample technologies...")
    created_technologies = []
    
    for category, technologies in sample_technologies.items():
        for tech_data in technologies:
            try:
                # Check if technology already exists
                existing_tech = db.query(db.query(sys.modules[__name__].Technology).filter(
                    sys.modules[__name__].Technology.name == tech_data["name"]
                ).first() if hasattr(sys.modules[__name__], 'Technology') else None)
                
                if not existing_tech:
                    tech = create_technology(
                        db=db,
                        name=tech_data["name"],
                        category=category,
                        description=tech_data["description"],
                        keywords=tech_data["keywords"]
                    )
                    created_technologies.append(tech)
                    print(f"  ✅ Created {tech_data['name']} ({category})")
                else:
                    print(f"  ⚠️  {tech_data['name']} already exists")
                    
            except Exception as e:
                print(f"  ❌ Error creating {tech_data['name']}: {e}")
    
    print(f"\n📈 Adding sample trend data for {len(created_technologies)} technologies...")
    
    # Add trend data for the last 30 days
    for tech in created_technologies:
        for days_ago in range(30, 0, -1):
            date = datetime.utcnow() - timedelta(days=days_ago)
            
            # Generate realistic trend data
            base_score = random.uniform(0.3, 0.9)
            trend_score = base_score + random.uniform(-0.1, 0.1)
            trend_score = max(0.0, min(1.0, trend_score))
            
            github_stars = random.randint(100, 50000)
            github_forks = int(github_stars * random.uniform(0.05, 0.3))
            github_issues = int(github_stars * random.uniform(0.01, 0.1))
            
            try:
                add_trend_data(
                    db=db,
                    technology_id=tech.id,
                    source="github",
                    github_stars=github_stars,
                    github_forks=github_forks,
                    github_issues=github_issues,
                    trend_score=trend_score,
                    momentum_score=random.uniform(0.0, 1.0),
                    adoption_score=random.uniform(0.0, 1.0)
                )
                
            except Exception as e:
                print(f"  ❌ Error adding trend data for {tech.name}: {e}")
    
    print("\n🔮 Creating sample predictions...")
    
    # Create predictions for some technologies
    for tech in created_technologies[:5]:  # Just first 5 technologies
        try:
            target_date = datetime.utcnow() + timedelta(days=90)
            
            prediction = create_prediction(
                db=db,
                technology_id=tech.id,
                target_date=target_date,
                adoption_probability=random.uniform(0.4, 0.9),
                market_impact_score=random.uniform(0.3, 0.8),
                risk_score=random.uniform(0.1, 0.5),
                confidence_interval=random.uniform(0.6, 0.9),
                model_used="sample_model",
                features_used=["trend_score", "github_stars", "momentum"],
                prediction_reasoning=f"Sample prediction for {tech.name} based on recent trends"
            )
            
            print(f"  ✅ Created prediction for {tech.name}")
            
        except Exception as e:
            print(f"  ❌ Error creating prediction for {tech.name}: {e}")
    
    db.close()
    print("\n🎉 Test data initialization complete!")

def test_api_endpoints():
    """Test API endpoints"""
    print("\n🧪 Testing API endpoints...")
    
    try:
        import requests
        
        base_url = "http://localhost:8000"
        
        # Test root endpoint
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("  ✅ Root endpoint working")
        else:
            print(f"  ❌ Root endpoint failed: {response.status_code}")
        
        # Test health endpoint
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("  ✅ Health endpoint working")
        else:
            print(f"  ❌ Health endpoint failed: {response.status_code}")
        
        # Test trends endpoint
        response = requests.get(f"{base_url}/api/trends/")
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Trends endpoint working ({data['total_technologies']} technologies)")
        else:
            print(f"  ❌ Trends endpoint failed: {response.status_code}")
        
        # Test predictions endpoint
        response = requests.get(f"{base_url}/api/predictions/")
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Predictions endpoint working ({data['total_predictions']} predictions)")
        else:
            print(f"  ❌ Predictions endpoint failed: {response.status_code}")
        
        # Test technologies endpoint
        response = requests.get(f"{base_url}/api/technologies/")
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Technologies endpoint working ({len(data)} technologies)")
        else:
            print(f"  ❌ Technologies endpoint failed: {response.status_code}")
        
    except ImportError:
        print("  ⚠️  'requests' library not available. Please install it to test API endpoints.")
    except Exception as e:
        print(f"  ❌ Error testing API endpoints: {e}")

if __name__ == "__main__":
    print("🚀 Tech Trend Radar Backend Test")
    print("=" * 50)
    
    # Initialize test data
    asyncio.run(initialize_test_data())
    
    # Test API endpoints (requires server to be running)
    print("\n" + "=" * 50)
    print("To test API endpoints, run the server with:")
    print("  cd backend && python main.py")
    print("Then run this script again to test endpoints.")
    
    print("\n🎯 Backend setup complete!")
    print("Next steps:")
    print("1. Run: cd backend && python main.py")
    print("2. Visit: http://localhost:8000")
    print("3. API docs: http://localhost:8000/docs")
    print("4. Run data collection: python -c \"import asyncio; from data_collectors.github_collector import run_github_collection; asyncio.run(run_github_collection())\"") 