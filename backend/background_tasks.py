import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional
import threading
import time

from data_collectors.github_collector import run_github_collection
from config import settings
from models.database import get_db_sync, CollectionLog

logger = logging.getLogger(__name__)

class BackgroundTaskManager:
    """Manages background data collection tasks"""
    
    def __init__(self):
        self.running = False
        self.github_thread = None
        self.last_github_run = None
        
    def start_background_tasks(self):
        """Start all background tasks"""
        if self.running:
            logger.warning("Background tasks already running")
            return
            
        self.running = True
        logger.info("🚀 Starting background data collection tasks...")
        
        # Start GitHub collection in background thread
        self.github_thread = threading.Thread(target=self._github_collection_loop, daemon=True)
        self.github_thread.start()
        
        logger.info("✅ Background tasks started successfully")
    
    def stop_background_tasks(self):
        """Stop all background tasks"""
        self.running = False
        logger.info("🛑 Stopping background tasks...")
        
        if self.github_thread and self.github_thread.is_alive():
            self.github_thread.join(timeout=5)
        
        logger.info("✅ Background tasks stopped")
    
    def _github_collection_loop(self):
        """Background loop for GitHub data collection"""
        while self.running:
            try:
                # Check if it's time to run collection
                if self._should_run_github_collection():
                    logger.info("📊 Starting GitHub data collection...")
                    
                    # Run collection in new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        results = loop.run_until_complete(run_github_collection())
                        self.last_github_run = datetime.utcnow()
                        
                        logger.info(f"✅ GitHub collection completed: {results.get('total_repositories', 0)} repositories")
                        
                    except Exception as e:
                        logger.error(f"❌ GitHub collection failed: {e}")
                    finally:
                        loop.close()
                
                # Sleep for 5 minutes before checking again
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Error in GitHub collection loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _should_run_github_collection(self) -> bool:
        """Check if GitHub collection should run based on interval"""
        if not self.last_github_run:
            return True
        
        time_since_last = datetime.utcnow() - self.last_github_run
        interval = timedelta(seconds=settings.COLLECTION_INTERVAL)
        
        return time_since_last >= interval
    
    def get_status(self) -> dict:
        """Get status of background tasks"""
        return {
            "running": self.running,
            "last_github_run": self.last_github_run.isoformat() if self.last_github_run else None,
            "next_github_run": self._get_next_run_time(),
            "collection_interval_hours": settings.COLLECTION_INTERVAL / 3600
        }
    
    def _get_next_run_time(self) -> Optional[str]:
        """Get next scheduled run time"""
        if not self.last_github_run:
            return None
        
        next_run = self.last_github_run + timedelta(seconds=settings.COLLECTION_INTERVAL)
        return next_run.isoformat()

# Global task manager instance
task_manager = BackgroundTaskManager()

def start_background_tasks():
    """Start background tasks (called from main.py)"""
    task_manager.start_background_tasks()

def stop_background_tasks():
    """Stop background tasks (called from main.py)"""
    task_manager.stop_background_tasks()

def get_task_status():
    """Get current task status"""
    return task_manager.get_status() 