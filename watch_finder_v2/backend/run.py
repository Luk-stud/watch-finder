"""
Main entry point for the Watch Finder backend.
"""

import os
import sys
from api.linucb_app import app, initialize_system
from config import DEFAULT_PORT, DEBUG

if __name__ == '__main__':
    port = int(os.getenv('PORT', DEFAULT_PORT))
    
    print("🚀 Starting Watch Finder v2 Backend")
    print("=" * 50)
    
    # Initialize the system before starting the server
    print("Initializing LinUCB engine...")
    if initialize_system():
        print("✅ LinUCB engine initialized successfully!")
        print(f"Server running on port {port}")
        print("=" * 50)
        app.run(
            host='0.0.0.0',
            port=port,
            debug=DEBUG
        )
    else:
        print("❌ Failed to initialize system. Exiting.")
        sys.exit(1) 