from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Optional
from datetime import datetime
import logging
from contextlib import asynccontextmanager

from api.trends import router as trends_router
from api.predictions import router as predictions_router
from api.technologies import router as technologies_router
from api.analytics import router as analytics_router
from data_collectors.github_collector import GitHubCollector
from models.database import init_database, get_db
from config import settings
from ml_models.model_manager import ModelManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model manager globally
model_manager = ModelManager()

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Tech Trend Radar API...")
    await init_database()
    logger.info("✅ Database initialized")
    
    # Try to load ML models
    logger.info("🤖 Loading ML models...")
    try:
        if model_manager.load_models():
            logger.info("✅ ML models loaded successfully")
        else:
            logger.info("⚠️ No pre-trained ML models found")
    except Exception as e:
        logger.warning(f"⚠️ Error loading ML models: {e}")
    
    # Start background data collection
    # This would typically be handled by Celery in production
    logger.info("🔄 Starting data collection services...")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Tech Trend Radar API...")

# Create FastAPI app
app = FastAPI(
    title="Tech Trend Radar API",
    description="A comprehensive platform for monitoring and predicting emerging technology trends",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(trends_router, prefix="/api/trends", tags=["trends"])
app.include_router(predictions_router, prefix="/api/predictions", tags=["predictions"])
app.include_router(technologies_router, prefix="/api/technologies", tags=["technologies"])
app.include_router(analytics_router, prefix="/api/analytics", tags=["analytics"])

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Tech Trend Radar API",
        "version": "1.0.0",
        "description": "A comprehensive platform for monitoring and predicting emerging technology trends",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "endpoints": {
            "trends": "/api/trends",
            "predictions": "/api/predictions",
            "technologies": "/api/technologies",
            "analytics": "/api/analytics"
        }
    }

@app.get("/health", response_model=dict)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/stats", response_model=dict)
async def get_system_stats(db=Depends(get_db)):
    """Get system statistics"""
    try:
        # This would query the database for actual stats
        return {
            "total_technologies": 1250,
            "active_trends": 89,
            "predictions_made": 567,
            "data_sources": 5,
            "last_update": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    ) 