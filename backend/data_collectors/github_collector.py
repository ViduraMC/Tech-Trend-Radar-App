import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
from dataclasses import dataclass
import json
import re

from config import settings, TECHNOLOGY_CATEGORIES
from models.database import get_db_sync, create_technology, get_technology_by_name, add_trend_data, log_collection

logger = logging.getLogger(__name__)

@dataclass
class GitHubRepoData:
    """Data structure for GitHub repository information"""
    name: str
    full_name: str
    description: str
    stars: int
    forks: int
    watchers: int
    issues: int
    language: str
    created_at: datetime
    updated_at: datetime
    topics: List[str]
    url: str

class GitHubCollector:
    """Collects technology trend data from GitHub"""
    
    def __init__(self):
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "TechTrendRadar/1.0"
        }
        if settings.GITHUB_TOKEN:
            self.headers["Authorization"] = f"token {settings.GITHUB_TOKEN}"
        
        self.session = None
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = datetime.utcnow()
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.REQUEST_TIMEOUT),
            headers=self.headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, url: str, params: Dict = None) -> Dict:
        """Make authenticated request to GitHub API"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        # Check rate limit
        if self.rate_limit_remaining <= 10:
            wait_time = (self.rate_limit_reset - datetime.utcnow()).total_seconds()
            if wait_time > 0:
                logger.warning(f"Rate limit approaching, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        try:
            async with self.session.get(url, params=params) as response:
                # Update rate limit info
                self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
                reset_timestamp = int(response.headers.get('X-RateLimit-Reset', 0))
                self.rate_limit_reset = datetime.fromtimestamp(reset_timestamp)
                
                if response.status == 200:
                    return await response.json()
                elif response.status == 403:
                    logger.error("GitHub API rate limit exceeded")
                    raise Exception("Rate limit exceeded")
                else:
                    logger.error(f"GitHub API error: {response.status}")
                    response.raise_for_status()
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error accessing GitHub API: {e}")
            raise
    
    async def search_repositories(self, query: str, sort: str = "stars", order: str = "desc", per_page: int = 100) -> List[GitHubRepoData]:
        """Search GitHub repositories"""
        url = f"{self.base_url}/search/repositories"
        params = {
            "q": query,
            "sort": sort,
            "order": order,
            "per_page": per_page
        }
        
        try:
            data = await self._make_request(url, params)
            repositories = []
            
            for item in data.get("items", []):
                repo_data = GitHubRepoData(
                    name=item["name"],
                    full_name=item["full_name"],
                    description=item.get("description", ""),
                    stars=item["stargazers_count"],
                    forks=item["forks_count"],
                    watchers=item["watchers_count"],
                    issues=item["open_issues_count"],
                    language=item.get("language", ""),
                    created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
                    updated_at=datetime.fromisoformat(item["updated_at"].replace("Z", "+00:00")),
                    topics=item.get("topics", []),
                    url=item["html_url"]
                )
                repositories.append(repo_data)
            
            return repositories
            
        except Exception as e:
            logger.error(f"Error searching repositories: {e}")
            return []
    
    async def get_repository_stats(self, repo_full_name: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific repository"""
        url = f"{self.base_url}/repos/{repo_full_name}"
        
        try:
            data = await self._make_request(url)
            return {
                "stars": data["stargazers_count"],
                "forks": data["forks_count"],
                "watchers": data["watchers_count"],
                "issues": data["open_issues_count"],
                "subscribers": data["subscribers_count"],
                "size": data["size"],
                "language": data.get("language", ""),
                "topics": data.get("topics", []),
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
                "pushed_at": data["pushed_at"]
            }
        except Exception as e:
            logger.error(f"Error getting repository stats for {repo_full_name}: {e}")
            return {}
    
    async def get_trending_repositories(self, language: str = None, since: str = "daily") -> List[GitHubRepoData]:
        """Get trending repositories (using search with date filters)"""
        # Calculate date range based on 'since' parameter
        end_date = datetime.utcnow()
        if since == "daily":
            start_date = end_date - timedelta(days=1)
        elif since == "weekly":
            start_date = end_date - timedelta(weeks=1)
        elif since == "monthly":
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=1)
        
        # Build query
        query_parts = [
            f"created:{start_date.strftime('%Y-%m-%d')}..{end_date.strftime('%Y-%m-%d')}"
        ]
        
        if language:
            query_parts.append(f"language:{language}")
        
        query = " ".join(query_parts)
        
        return await self.search_repositories(query, sort="stars", order="desc")
    
    def _categorize_technology(self, repo_data: GitHubRepoData) -> Optional[str]:
        """Categorize a repository based on its content"""
        text_to_analyze = f"{repo_data.name} {repo_data.description} {' '.join(repo_data.topics)} {repo_data.language}".lower()
        
        category_scores = {}
        
        for category, keywords in TECHNOLOGY_CATEGORIES.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text_to_analyze:
                    score += 1
            
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return None
    
    async def collect_technology_trends(self) -> Dict[str, Any]:
        """Collect technology trends from GitHub"""
        collection_start = datetime.utcnow()
        results = {
            "total_repositories": 0,
            "categories_found": {},
            "errors": [],
            "execution_time": 0
        }
        
        try:
            db = get_db_sync()
            
            # Collect trending repositories
            trending_repos = await self.get_trending_repositories(since="weekly")
            results["total_repositories"] = len(trending_repos)
            
            for repo in trending_repos:
                try:
                    # Categorize the repository
                    category = self._categorize_technology(repo)
                    
                    if category:
                        # Update category count
                        if category not in results["categories_found"]:
                            results["categories_found"][category] = 0
                        results["categories_found"][category] += 1
                        
                        # Get or create technology entry
                        tech = get_technology_by_name(db, repo.name)
                        if not tech:
                            tech = create_technology(
                                db=db,
                                name=repo.name,
                                category=category,
                                description=repo.description,
                                keywords=repo.topics
                            )
                        
                        # Add trend data
                        add_trend_data(
                            db=db,
                            technology_id=tech.id,
                            source="github",
                            github_stars=repo.stars,
                            github_forks=repo.forks,
                            github_issues=repo.issues,
                            trend_score=self._calculate_trend_score(repo),
                            momentum_score=self._calculate_momentum_score(repo),
                            adoption_score=self._calculate_adoption_score(repo)
                        )
                        
                        logger.info(f"Processed {repo.name} ({category})")
                        
                except Exception as e:
                    error_msg = f"Error processing repository {repo.name}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            # Search for specific technology categories
            for category, keywords in TECHNOLOGY_CATEGORIES.items():
                try:
                    # Search for top repositories in this category
                    query = f"{keywords[0]} OR {keywords[1] if len(keywords) > 1 else keywords[0]}"
                    repos = await self.search_repositories(query, per_page=20)
                    
                    for repo in repos:
                        try:
                            # Get or create technology entry
                            tech = get_technology_by_name(db, repo.name)
                            if not tech:
                                tech = create_technology(
                                    db=db,
                                    name=repo.name,
                                    category=category,
                                    description=repo.description,
                                    keywords=repo.topics
                                )
                            
                            # Add trend data
                            add_trend_data(
                                db=db,
                                technology_id=tech.id,
                                source="github",
                                github_stars=repo.stars,
                                github_forks=repo.forks,
                                github_issues=repo.issues,
                                trend_score=self._calculate_trend_score(repo),
                                momentum_score=self._calculate_momentum_score(repo),
                                adoption_score=self._calculate_adoption_score(repo)
                            )
                            
                        except Exception as e:
                            error_msg = f"Error processing {repo.name}: {e}"
                            logger.error(error_msg)
                            results["errors"].append(error_msg)
                    
                    # Small delay to respect rate limits
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    error_msg = f"Error searching for {category}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            db.close()
            
        except Exception as e:
            error_msg = f"Critical error in GitHub collection: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
        
        finally:
            results["execution_time"] = (datetime.utcnow() - collection_start).total_seconds()
            
            # Log collection results
            db = get_db_sync()
            log_collection(
                db=db,
                source_name="github",
                status="success" if not results["errors"] else "error",
                records_collected=results["total_repositories"],
                errors_encountered=len(results["errors"]),
                execution_time=results["execution_time"],
                details=results
            )
            db.close()
        
        return results
    
    def _calculate_trend_score(self, repo: GitHubRepoData) -> float:
        """Calculate trend score based on repository metrics"""
        # Normalize values (these are rough approximations)
        star_score = min(repo.stars / 10000, 1.0)  # Max score at 10k stars
        fork_score = min(repo.forks / 1000, 1.0)   # Max score at 1k forks
        
        # Recent activity bonus
        days_since_update = (datetime.utcnow() - repo.updated_at.replace(tzinfo=None)).days
        recency_score = max(0, 1 - (days_since_update / 30))  # Decay over 30 days
        
        # Weighted combination
        trend_score = (star_score * 0.4) + (fork_score * 0.3) + (recency_score * 0.3)
        
        return round(trend_score, 3)
    
    def _calculate_momentum_score(self, repo: GitHubRepoData) -> float:
        """Calculate momentum score based on repository activity"""
        # This is a simplified version - in practice, you'd need historical data
        days_since_creation = (datetime.utcnow() - repo.created_at.replace(tzinfo=None)).days
        
        if days_since_creation > 0:
            # Stars per day since creation
            star_velocity = repo.stars / days_since_creation
            momentum_score = min(star_velocity * 10, 1.0)  # Normalize
        else:
            momentum_score = 0.0
        
        return round(momentum_score, 3)
    
    def _calculate_adoption_score(self, repo: GitHubRepoData) -> float:
        """Calculate adoption score based on repository usage indicators"""
        # Simple adoption score based on forks and watchers
        if repo.stars > 0:
            fork_ratio = repo.forks / repo.stars
            watcher_ratio = repo.watchers / repo.stars
            
            # Higher fork ratio indicates more usage/adoption
            adoption_score = min((fork_ratio * 5) + (watcher_ratio * 2), 1.0)
        else:
            adoption_score = 0.0
        
        return round(adoption_score, 3)

# Convenience function for running collection
async def run_github_collection():
    """Run GitHub data collection"""
    async with GitHubCollector() as collector:
        results = await collector.collect_technology_trends()
        logger.info(f"GitHub collection completed: {results}")
        return results

if __name__ == "__main__":
    asyncio.run(run_github_collection()) 