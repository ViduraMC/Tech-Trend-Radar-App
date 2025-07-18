#!/usr/bin/env python3
"""
Enable real data collection for Tech Trend Radar
This script will:
1. Set up GitHub API token
2. Configure real data sources
3. Start collecting real technology trends
"""

import os
import sys
import asyncio
from datetime import datetime
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collectors.github_collector import GitHubCollector, run_github_collection
from models.database import get_db_sync, Technology, TrendData
from config import REAL_DATA_SOURCES, COLLECTION_SETTINGS

def setup_environment():
    """Set up environment variables for real data collection"""
    print("🔧 Setting up real data collection...")
    
    # Check if .env file exists
    env_file = ".env"
    if not os.path.exists(env_file):
        print("📝 Creating .env file...")
        with open(env_file, "w") as f:
            f.write("# Tech Trend Radar Environment Variables\n")
            f.write("# Database Configuration\n")
            f.write("DATABASE_URL=sqlite:///./tech_trend_radar.db\n")
            f.write("DATABASE_ECHO=false\n\n")
            f.write("# API Configuration\n")
            f.write("API_HOST=0.0.0.0\n")
            f.write("API_PORT=8000\n")
            f.write("DEBUG=true\n\n")
            f.write("# Data Collection Configuration\n")
            f.write("REQUEST_TIMEOUT=30\n")
            f.write("COLLECTION_INTERVAL=3600\n\n")
            f.write("# GitHub API (Get token from: https://github.com/settings/tokens)\n")
            f.write("GITHUB_TOKEN=your_github_token_here\n\n")
            f.write("# Other APIs (Optional)\n")
            f.write("ARXIV_EMAIL=your_email@example.com\n")
            f.write("PATENT_API_KEY=your_patent_api_key_here\n")
            f.write("JOB_API_KEY=your_job_api_key_here\n")
    
    print("✅ Environment file ready!")
    print("\n📋 Next steps:")
    print("1. Get a GitHub token from: https://github.com/settings/tokens")
    print("2. Edit .env file and add your GitHub token")
    print("3. Run this script again to start data collection")

async def collect_real_github_data():
    """Collect real data from GitHub"""
    print("🚀 Starting real GitHub data collection...")
    
    try:
        # Check if GitHub token is configured
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token or github_token == "your_github_token_here":
            print("❌ GitHub token not configured!")
            print("Please edit .env file and add your GitHub token")
            return False
        
        # Initialize GitHub collector
        async with GitHubCollector() as collector:
            print("📊 Collecting technology trends from GitHub...")
            
            # Collect data for each technology category
            for category, keywords in REAL_DATA_SOURCES["github"]["topics"]:
                print(f"🔍 Searching for {category} technologies...")
                
                # Search for repositories in this category
                repos = await collector.search_repositories(
                    query=f"topic:{category} stars:>100",
                    sort="stars",
                    order="desc",
                    per_page=20
                )
                
                print(f"✅ Found {len(repos)} repositories for {category}")
                
                # Process each repository
                for repo in repos:
                    try:
                        # Check if technology already exists
                        db = get_db_sync()
                        existing_tech = db.query(Technology).filter(
                            Technology.name == repo.name
                        ).first()
                        
                        if not existing_tech:
                            # Create new technology
                            tech = Technology(
                                name=repo.name,
                                category=collector._categorize_technology(repo) or "Other",
                                description=repo.description,
                                keywords=repo.topics,
                                first_detected=datetime.utcnow(),
                                last_updated=datetime.utcnow()
                            )
                            db.add(tech)
                            db.commit()
                            db.refresh(tech)
                            print(f"  ✅ Added: {repo.name}")
                        else:
                            tech = existing_tech
                            print(f"  ⚠️  Exists: {repo.name}")
                        
                        # Add trend data
                        from models.database import add_trend_data
                        add_trend_data(
                            db=db,
                            technology_id=tech.id,
                            source="github",
                            github_stars=repo.stars,
                            github_forks=repo.forks,
                            github_issues=repo.issues,
                            trend_score=collector._calculate_trend_score(repo),
                            momentum_score=collector._calculate_momentum_score(repo),
                            adoption_score=collector._calculate_adoption_score(repo)
                        )
                        
                        db.close()
                        
                    except Exception as e:
                        print(f"  ❌ Error processing {repo.name}: {e}")
                        continue
                
                # Rate limiting - wait between categories
                await asyncio.sleep(2)
        
        print("🎉 Real data collection completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during data collection: {e}")
        return False

def show_data_collection_status():
    """Show current data collection status"""
    print("\n📊 Current Data Collection Status:")
    
    # Check GitHub configuration
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token and github_token != "your_github_token_here":
        print("✅ GitHub API: Configured")
    else:
        print("❌ GitHub API: Not configured")
    
    # Check database status
    db = get_db_sync()
    total_techs = db.query(Technology).count()
    total_trends = db.query(TrendData).count()
    
    print(f"📈 Database: {total_techs} technologies, {total_trends} trend records")
    
    # Show recent technologies
    recent_techs = db.query(Technology).order_by(Technology.last_updated.desc()).limit(5).all()
    print("\n🆕 Recent technologies:")
    for tech in recent_techs:
        print(f"  - {tech.name} ({tech.category})")
    
    db.close()

def main():
    """Main function"""
    print("🌟 Tech Trend Radar - Real Data Collection Setup")
    print("=" * 50)
    
    # Check if .env exists
    if not os.path.exists(".env"):
        setup_environment()
        return
    
    # Show current status
    show_data_collection_status()
    
    # Ask user what to do
    print("\n🔧 What would you like to do?")
    print("1. Set up GitHub token")
    print("2. Collect real data from GitHub")
    print("3. Show current status")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        setup_environment()
    elif choice == "2":
        asyncio.run(collect_real_github_data())
    elif choice == "3":
        show_data_collection_status()
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main() 