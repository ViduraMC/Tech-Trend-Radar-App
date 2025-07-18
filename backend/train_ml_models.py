#!/usr/bin/env python3
"""
Script to train ML models for Tech Trend Radar
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from ml_models.model_manager import ModelManager
from models.database import get_db_sync, init_database

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main function to train ML models"""
    logger.info("🚀 Starting ML model training for Tech Trend Radar...")
    
    try:
        # Initialize database
        logger.info("📊 Initializing database...")
        await init_database()
        
        # Get database session
        db = get_db_sync()
        
        # Initialize model manager
        logger.info("🤖 Initializing model manager...")
        model_manager = ModelManager()
        
        # Train models
        logger.info("🎯 Training ML models...")
        result = model_manager.initialize_models(db)
        
        if result['status'] == 'success':
            logger.info("✅ ML models trained successfully!")
            logger.info(f"📈 Training results: {result['training_results']}")
            logger.info(f"📊 Data samples used: {result['data_samples']}")
            
            # Get model status
            status = model_manager.get_model_status()
            logger.info(f"🔍 Model status: {status}")
            
        elif result['status'] == 'no_data':
            logger.warning("⚠️ No training data available")
            logger.info("💡 Please ensure you have technologies and trend data in the database")
            
        else:
            logger.error(f"❌ Training failed: {result['message']}")
            return 1
        
        db.close()
        logger.info("🎉 ML model training completed!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error during training: {e}")
        return 1

if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 