#!/usr/bin/env python3
"""
Test script to demonstrate automatic data collection
"""

import asyncio
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from background_tasks import start_background_tasks, stop_background_tasks, get_task_status
from config import settings

async def test_automatic_updates():
    """Test automatic data collection"""
    print("🧪 Testing Automatic Data Collection")
    print("=" * 50)
    
    # Show current status
    status = get_task_status()
    print(f"📊 Current status: {status}")
    
    # Start background tasks
    print("\n🚀 Starting background tasks...")
    start_background_tasks()
    
    # Monitor for 2 minutes
    print("\n⏰ Monitoring for 2 minutes...")
    for i in range(12):  # 12 * 10 seconds = 2 minutes
        time.sleep(10)
        status = get_task_status()
        print(f"⏱️  {i+1}/12 - Status: {status}")
    
    # Stop background tasks
    print("\n🛑 Stopping background tasks...")
    stop_background_tasks()
    
    # Final status
    final_status = get_task_status()
    print(f"\n📊 Final status: {final_status}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_automatic_updates()) 