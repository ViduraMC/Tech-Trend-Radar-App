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
from background_tasks import start_background_tasks, stop_background_tasks, get_task_status

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
    logger.info("🔄 Starting data collection services...")
    start_background_tasks()
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Tech Trend Radar API...")
    stop_background_tasks()

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
        from models.database import Technology, TrendData, Prediction
        
        total_technologies = db.query(Technology).count()
        total_trends = db.query(TrendData).count()
        total_predictions = db.query(Prediction).count()
        
        # Get background task status
        task_status = get_task_status()
        
        return {
            "total_technologies": total_technologies,
            "active_trends": total_trends,
            "predictions_made": total_predictions,
            "data_sources": 5,
            "last_update": datetime.utcnow().isoformat(),
            "background_tasks": task_status
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/collect-data", response_model=dict)
async def trigger_data_collection():
    """Manually trigger data collection"""
    try:
        import asyncio
        from data_collectors.github_collector import run_github_collection
        
        logger.info("🔄 Manual data collection triggered...")
        
        # Run GitHub collection
        results = await run_github_collection()
        
        return {
            "message": "Data collection completed successfully",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in manual data collection: {e}")
        raise HTTPException(status_code=500, detail=f"Data collection failed: {str(e)}")

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