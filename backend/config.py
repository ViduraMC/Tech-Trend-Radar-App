from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Tech Trend Radar"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ]
    
    # Database
    DATABASE_URL: str = "sqlite:///./tech_trend_radar.db"  # Using SQLite for simplicity
    DATABASE_ECHO: bool = False
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # External APIs
    GITHUB_TOKEN: str = ""
    OPENAI_API_KEY: str = ""
    
    # Google Patents API (if available)
    GOOGLE_PATENTS_API_KEY: str = ""
    
    # ArXiv API (no key needed, but rate limits apply)
    ARXIV_API_DELAY: float = 3.0  # seconds between requests
    
    # Job APIs
    INDEED_API_KEY: str = ""
    LINKEDIN_API_KEY: str = ""
    
    # Data Collection
    COLLECTION_INTERVAL: int = 3600  # seconds (1 hour)
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 30
    
    # Machine Learning
    MODEL_UPDATE_INTERVAL: int = 86400  # seconds (24 hours)
    TREND_THRESHOLD: float = 0.7
    PREDICTION_HORIZON_DAYS: int = 90
    
    # Caching
    CACHE_TTL: int = 3600  # seconds
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Technology categories for classification
TECHNOLOGY_CATEGORIES = {
    "AI/ML": [
        "artificial intelligence", "machine learning", "deep learning", 
        "neural networks", "nlp", "computer vision", "reinforcement learning"
    ],
    "Blockchain": [
        "blockchain", "cryptocurrency", "bitcoin", "ethereum", "smart contracts",
        "defi", "nft", "web3"
    ],
    "Cloud": [
        "cloud computing", "aws", "azure", "google cloud", "kubernetes",
        "docker", "serverless", "microservices"
    ],
    "Data": [
        "big data", "data science", "analytics", "data engineering",
        "apache spark", "kafka", "elasticsearch"
    ],
    "DevOps": [
        "devops", "ci/cd", "jenkins", "github actions", "terraform",
        "ansible", "monitoring"
    ],
    "Frontend": [
        "react", "vue", "angular", "javascript", "typescript",
        "css", "html", "web development"
    ],
    "Backend": [
        "python", "java", "golang", "rust", "nodejs", "api",
        "database", "sql", "nosql"
    ],
    "Mobile": [
        "mobile development", "ios", "android", "react native",
        "flutter", "swift", "kotlin"
    ],
    "IoT": [
        "internet of things", "iot", "embedded systems", "sensors",
        "edge computing", "arduino", "raspberry pi"
    ],
    "Quantum": [
        "quantum computing", "quantum algorithms", "qubits",
        "quantum cryptography", "quantum machine learning"
    ],
    "AR/VR": [
        "augmented reality", "virtual reality", "mixed reality",
        "unity", "unreal engine", "metaverse"
    ],
    "Cybersecurity": [
        "cybersecurity", "security", "encryption", "authentication",
        "vulnerability", "penetration testing", "firewall"
    ]
}

# Real Data Sources Configuration
REAL_DATA_SOURCES = {
    "github": {
        "enabled": True,
        "rate_limit": 5000,  # requests per hour
        "collection_interval": 3600,  # 1 hour
        "topics": [
            "machine-learning", "artificial-intelligence", "deep-learning",
            "react", "vue", "angular", "python", "javascript", "typescript",
            "docker", "kubernetes", "aws", "azure", "gcp", "blockchain",
            "flutter", "react-native", "mobile", "data-science", "devops"
        ]
    },
    "arxiv": {
        "enabled": False,  # Enable when you have API access
        "categories": ["cs.AI", "cs.LG", "cs.CV", "cs.NE", "cs.CR", "cs.DC"],
        "collection_interval": 86400  # 24 hours
    },
    "patents": {
        "enabled": False,  # Enable when you have API access
        "collection_interval": 86400  # 24 hours
    },
    "job_postings": {
        "enabled": False,  # Enable when you have API access
        "collection_interval": 3600  # 1 hour
    }
}

# Data Collection Settings
COLLECTION_SETTINGS = {
    "max_technologies_per_category": 50,
    "min_stars_threshold": 100,  # Minimum GitHub stars to consider
    "trend_score_weight": {
        "github_stars": 0.3,
        "github_forks": 0.2,
        "github_issues": 0.1,
        "arxiv_papers": 0.2,
        "patent_filings": 0.1,
        "job_postings": 0.1
    }
} 