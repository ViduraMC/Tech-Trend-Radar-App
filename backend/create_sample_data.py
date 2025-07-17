#!/usr/bin/env python3
"""
Simple script to create sample technologies for Tech Trend Radar
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.database import get_db_sync, Technology
from datetime import datetime

def create_sample_technologies():
    """Create sample technologies"""
    db = get_db_sync()
    
    sample_techs = [
        {"name": "React", "category": "Frontend", "description": "JavaScript library for building user interfaces"},
        {"name": "Vue.js", "category": "Frontend", "description": "Progressive JavaScript framework"},
        {"name": "TensorFlow", "category": "AI/ML", "description": "Open-source machine learning framework"},
        {"name": "Kubernetes", "category": "Cloud", "description": "Container orchestration platform"},
        {"name": "Docker", "category": "Cloud", "description": "Container platform"},
    ]
    
    created_count = 0
    for tech_data in sample_techs:
        # Check if exists
        existing = db.query(Technology).filter(Technology.name == tech_data["name"]).first()
        if not existing:
            tech = Technology(
                name=tech_data["name"],
                category=tech_data["category"],
                description=tech_data["description"],
                keywords=[tech_data["name"].lower()],
                first_detected=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            db.add(tech)
            created_count += 1
            print(f"✅ Created: {tech_data['name']}")
        else:
            print(f"⚠️  Exists: {tech_data['name']} (ID: {existing.id})")
    
    db.commit()
    
    # Show all technologies
    all_techs = db.query(Technology).all()
    print(f"\n📊 Total technologies in database: {len(all_techs)}")
    for tech in all_techs[:10]:  # Show first 10
        print(f"ID: {tech.id} | Name: {tech.name} | Category: {tech.category}")
    
    db.close()
    return len(all_techs)

if __name__ == "__main__":
    print("🔧 Creating sample technologies...")
    count = create_sample_technologies()
    print(f"\n🎉 Database has {count} technologies ready!") 