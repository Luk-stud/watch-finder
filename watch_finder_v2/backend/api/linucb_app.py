#!/usr/bin/env python3
"""
Watch Finder v2 Backend API
==========================

Flask API for the Multi-Expert LinUCB recommendation system.
"""

import os
import sys
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import configuration
from config import (
    DEFAULT_PORT, DEBUG, 
    LINUCB_DIMENSION, LINUCB_ALPHA, RECOMMENDATION_BATCH_SIZE,
    NUM_EXPERTS, MAX_EXPERTS, EXPERT_RECOMMENDATION_SIZE, 
    EXPERT_SIMILARITY_THRESHOLD, MIN_EXPERT_SIZE,
    ENABLE_MULTI_EXPERT, SESSION_TIMEOUT_MINUTES,
    MIN_LIKES_FOR_FIRST_EXPERT, MIN_LIKES_FOR_NEW_EXPERT,
    LIKE_CLUSTERING_SIMILARITY_THRESHOLD, PURE_EXPLORATION_MODE
)

# Import our LinUCB components
try:
    from models.linucb_engine import MultiExpertLinUCBEngine
from models.simple_session import SimpleSession
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the backend directory or the models are available")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global state
recommendation_engine: Optional[MultiExpertLinUCBEngine] = None
sessions: Dict[str, SimpleSession] = {}

def initialize_system() -> bool:
    """Initialize the Like-Driven Dynamic Multi-Expert LinUCB recommendation system."""
    global recommendation_engine
    
    try:
        print(f"🎯 Initializing Like-Driven Dynamic Multi-Expert LinUCB...")
        print(f"   📊 Starting experts: {NUM_EXPERTS} (will create from likes)")
        print(f"   📈 Max experts: {MAX_EXPERTS}")
        print(f"   🎯 Need {MIN_LIKES_FOR_FIRST_EXPERT} likes for first expert")
        print(f"   🎯 Need {MIN_LIKES_FOR_NEW_EXPERT} likes for additional experts")
        print(f"   🔍 Pure exploration mode: {PURE_EXPLORATION_MODE}")
        
        recommendation_engine = MultiExpertLinUCBEngine(
            dim=LINUCB_DIMENSION,
            alpha=LINUCB_ALPHA,
            batch_size=RECOMMENDATION_BATCH_SIZE,
            num_experts=NUM_EXPERTS,  # 0 for like-driven
            max_experts=MAX_EXPERTS,
            similarity_threshold=EXPERT_SIMILARITY_THRESHOLD,
            min_expert_size=MIN_EXPERT_SIZE,
            initialization_strategy='like_driven',  # NEW
            min_likes_for_first_expert=MIN_LIKES_FOR_FIRST_EXPERT,  # NEW
            min_likes_for_new_expert=MIN_LIKES_FOR_NEW_EXPERT,  # NEW
            like_clustering_threshold=LIKE_CLUSTERING_SIMILARITY_THRESHOLD  # NEW
        )
        
        # Test the system with a dummy recommendation
        test_context = [0.5] * LINUCB_DIMENSION
        test_recs = recommendation_engine.get_recommendations(
            session_id="test",
            context=test_context,
            exclude_ids=set()
        )
        
        if len(test_recs) > 0:
            print(f"✅ Like-Driven Dynamic Multi-Expert LinUCB system initialized successfully!")
            print(f"   📊 {len(recommendation_engine.experts)} experts managing {len(recommendation_engine.watch_data)} watches")
            print(f"   🔍 {len(recommendation_engine.unassigned_watches)} watches available for exploration")
            if recommendation_engine.pure_exploration_mode:
                print(f"   🚀 PURE EXPLORATION MODE ACTIVE - collecting likes...")
        return True
        else:
            print("❌ No recommendations generated during test")
            return False
            
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        import traceback
        traceback.print_exc()
        return False

def sanitize_watch_for_json(watch_data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove or convert non-JSON-serializable data from watch data."""
    sanitized = {}
    
    for key, value in watch_data.items():
        if isinstance(value, np.ndarray):
            # Skip numpy arrays (embeddings, etc.)
            continue
        elif isinstance(value, np.integer):
            sanitized[key] = int(value)
        elif isinstance(value, np.floating):
            sanitized[key] = float(value)
        elif isinstance(value, dict):
            # Recursively sanitize nested dictionaries
            sanitized[key] = sanitize_watch_for_json(value)
        elif isinstance(value, list):
            # Sanitize lists
            sanitized_list = []
            for item in value:
                if isinstance(item, dict):
                    sanitized_list.append(sanitize_watch_for_json(item))
                elif isinstance(item, np.ndarray):
                    continue  # Skip numpy arrays
                elif isinstance(item, (np.integer, np.floating)):
                    sanitized_list.append(float(item) if isinstance(item, np.floating) else int(item))
                else:
                    sanitized_list.append(item)
            sanitized[key] = sanitized_list
        else:
            sanitized[key] = value
    
    return sanitized

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

# Health endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    if not recommendation_engine:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Recommendation engine not initialized'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'active_sessions': len(sessions)
    })

# Core API endpoints
@app.route('/api/session', methods=['POST'])
def create_session():
    """Create a new recommendation session."""
    if not recommendation_engine:
        return jsonify({
            'status': 'error',
            'message': 'System not initialized'
        }), 503
    
    try:
        # Create new session
        session = SimpleSession()
        sessions[session.id] = session
        
        # Get initial recommendations (no exclusions for first session)
        recommendations = recommendation_engine.get_recommendations(
            session_id=session.id,
            context=session.get_context(),
            exclude_ids=session.shown_watches
        )
        
        # Sanitize recommendations for JSON serialization
        sanitized_recommendations = [sanitize_watch_for_json(rec) for rec in recommendations]
        
        # Mark these watches as shown
        for rec in recommendations:
            session.add_shown_watch(rec['watch_id'])
        
        return jsonify({
            'status': 'success',
            'session_id': session.id,
            'recommendations': sanitized_recommendations
        })
        
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """Get next batch of recommendations."""
    session_id = request.headers.get('X-Session-ID')
    if not session_id or session_id not in sessions:
        return jsonify({
            'status': 'error',
            'message': 'Invalid session'
        }), 401
    
    try:
        session = sessions[session_id]
        recommendations = recommendation_engine.get_recommendations(
            session_id=session_id,
            context=session.get_context(),
            exclude_ids=session.shown_watches
        )
        
        # Sanitize recommendations for JSON serialization
        sanitized_recommendations = [sanitize_watch_for_json(rec) for rec in recommendations]
        
        # Mark these new watches as shown
        for rec in recommendations:
            session.add_shown_watch(rec['watch_id'])
        
        return jsonify({
            'status': 'success',
            'recommendations': sanitized_recommendations
        })
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/feedback', methods=['POST'])
def add_feedback():
    """Add user feedback for a watch."""
    session_id = request.headers.get('X-Session-ID')
    if not session_id or session_id not in sessions:
        return jsonify({
            'status': 'error',
            'message': 'Invalid session'
        }), 401
    
    try:
        data = request.json
        if not data or 'watch_id' not in data or 'action' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields'
            }), 400
        
        watch_id = data['watch_id']
        action = data['action']
        
        if action not in ['like', 'dislike']:
            return jsonify({
                'status': 'error',
                'message': 'Invalid action'
            }), 400
        
        session = sessions[session_id]
        
        # Update LinUCB model
        recommendation_engine.update(
            session_id=session_id,
            watch_id=watch_id,
            reward=1.0 if action == 'like' else 0.0,
            context=session.get_context()
        )
        
        # Update session
        if action == 'like':
            session.add_liked_watch(watch_id)
        # Note: shown_watches are now tracked in recommendation endpoints
        
        return jsonify({
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error adding feedback: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/liked-watches', methods=['GET'])
def get_liked_watches():
    """Get user's liked watches."""
    session_id = request.headers.get('X-Session-ID')
    if not session_id or session_id not in sessions:
        return jsonify({
            'status': 'error',
            'message': 'Invalid session'
        }), 401
    
    try:
        session = sessions[session_id]
        liked_watches = recommendation_engine.get_watches(session.liked_watches)
        
        # Sanitize liked watches for JSON serialization
        sanitized_liked_watches = [sanitize_watch_for_json(watch) for watch in liked_watches]
        
        return jsonify({
            'status': 'success',
            'liked_watches': sanitized_liked_watches
        })
        
    except Exception as e:
        logger.error(f"Error getting liked watches: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/watch/<int:watch_id>/details', methods=['GET'])
def get_watch_details(watch_id):
    """Get detailed information about a specific watch."""
    try:
        watch = recommendation_engine.get_watch_details(watch_id)
        if not watch:
            return jsonify({
                'status': 'error',
                'message': 'Watch not found'
            }), 404
        
        # Sanitize watch details for JSON serialization
        sanitized_watch = sanitize_watch_for_json(watch)
        
        return jsonify({
            'status': 'success',
            'watch': sanitized_watch
        })
        
    except Exception as e:
        logger.error(f"Error getting watch details: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Watch Finder v2 Backend")
    print("=" * 50)
    
    # Initialize the system before starting the server
    print("Initializing Multi-Expert LinUCB system...")
    if initialize_system():
        print("✅ Multi-Expert LinUCB system initialized successfully!")
        print(f"Server running on port {DEFAULT_PORT}")
        print("=" * 50)
        app.run(host='0.0.0.0', port=DEFAULT_PORT, debug=DEBUG)
    else:
        print("❌ Failed to initialize system. Exiting.")
        sys.exit(1)