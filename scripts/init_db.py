"""
Database Initialization Script
Creates PostgreSQL tables for multi-tier memory system
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.multi_tier_memory import MultiTierMemory
from config import settings


async def main():
    """Initialize database tables."""
    print("🚀 Initializing FAQ Bot Database...")
    print(f"   Host: {settings.POSTGRES_HOST}")
    print(f"   Database: {settings.POSTGRES_DB}")
    print(f"   User: {settings.POSTGRES_USER}")
    print()
    
    try:
        memory = MultiTierMemory()
        await memory.initialize()
        
        print("✅ Database initialized successfully!")
        print()
        print("Created tables:")
        print("  - user_profiles")
        print("  - conversation_sessions")
        print("  - conversation_turns")
        print()
        print("Indexes created:")
        print("  - idx_user_sessions")
        print("  - idx_session_turns")
        print()
        print("You can now start the API server.")
        
    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}")
        print()
        print("Troubleshooting:")
        print("  1. Check PostgreSQL is running: docker ps | grep postgres")
        print("  2. Verify credentials in .env file")
        print("  3. Test connection: psql -h localhost -U postgres -d faq_bot")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())