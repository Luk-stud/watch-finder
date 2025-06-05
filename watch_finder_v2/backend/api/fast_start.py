#!/usr/bin/env python3
"""
Fast Start Production API
========================

Optimized startup for Railway deployment:
- Flask app starts immediately for health checks
- System initialization happens in background
- Graceful handling of early requests
"""

import os
import sys
import logging
from datetime import datetime

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import the app from production_linucb_app
from production_linucb_app import app, initialize_system_background, DEFAULT_PORT, DEBUG

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("⚡ Fast Start: Production Watch Finder API")
    logger.info("🏥 Health endpoint available immediately")
    logger.info("🔄 System initialization starting in background...")
    
    # Start background initialization
    init_thread = initialize_system_background()
    
    # Get port from environment (Railway sets this)
    port = int(os.environ.get('PORT', DEFAULT_PORT))
    
    logger.info(f"🌐 Starting server immediately on port {port}")
    logger.info("✅ Ready for health checks!")
    
    try:
        # Run production server immediately - no blocking initialization
        app.run(
            host='0.0.0.0',
            port=port,
            debug=DEBUG,
            threaded=True
        )
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1) 