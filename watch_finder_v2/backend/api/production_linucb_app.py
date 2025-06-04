#!/usr/bin/env python3
"""
Production Watch Finder v2 Backend API
======================================

Production-ready Flask API with:
- Multi-user session isolation
- Rate limiting and abuse protection
- Session persistence across restarts
- Comprehensive monitoring and analytics
- Thread-safe concurrent operations
- Graceful error handling
"""

import os
import sys
import logging
import signal
import atexit
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
import numpy as np
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import time

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import configuration
from config import (
    DEFAULT_PORT, DEBUG, DATA_DIR
)

# Import production session manager
try:
    from models.production_session_manager import ProductionSessionManager
    from models.linucb_engine import DynamicMultiExpertLinUCBEngine
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the backend directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/production_api.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Handle proxy headers if behind reverse proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Global state
session_manager: Optional[ProductionSessionManager] = None

def sanitize_watch_for_json(watch: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize watch data for JSON serialization."""
    sanitized = watch.copy()
    
    # Remove or convert numpy arrays
    for key, value in sanitized.items():
        if isinstance(value, np.ndarray):
            sanitized[key] = value.tolist()
        elif hasattr(value, 'dtype'):  # Other numpy types
            sanitized[key] = value.item()
    
    return sanitized

def initialize_system() -> bool:
    """Initialize the production session management system."""
    global session_manager
    
    try:
        logger.info("🚀 Initializing Production Watch Recommendation System...")
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Initialize production session manager
        session_manager = ProductionSessionManager(
            data_dir=str(DATA_DIR),
            session_timeout_minutes=60,
            max_concurrent_sessions=1000,
            cleanup_interval_seconds=300,
            enable_persistence=True,
            max_requests_per_minute=120  # Allow higher rate for production
        )
        
        logger.info("✅ Production system initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        import traceback
        traceback.print_exc()
        return False

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'status': 'error',
        'message': 'Bad request',
        'error_code': 'BAD_REQUEST'
    }), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'error_code': 'NOT_FOUND'
    }), 404

@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({
        'status': 'error',
        'message': 'Rate limit exceeded',
        'error_code': 'RATE_LIMIT_EXCEEDED'
    }), 429

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'error_code': 'INTERNAL_ERROR'
    }), 500

@app.errorhandler(503)
def service_unavailable(error):
    return jsonify({
        'status': 'error',
        'message': 'Service temporarily unavailable',
        'error_code': 'SERVICE_UNAVAILABLE'
    }), 503

# Request middleware
@app.before_request
def before_request():
    """Log requests and validate system state."""
    g.start_time = datetime.now()
    
    # Check if system is initialized
    if not session_manager and request.endpoint not in ['health', 'status']:
        return jsonify({
            'status': 'error',
            'message': 'System not initialized'
        }), 503

@app.after_request
def after_request(response):
    """Log response and update metrics."""
    if hasattr(g, 'start_time'):
        duration_ms = (datetime.now() - g.start_time).total_seconds() * 1000
        logger.info(f"{request.method} {request.path} - {response.status_code} - {duration_ms:.1f}ms")
    
    return response

# Core API endpoints
@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    if session_manager:
        metrics = session_manager.get_system_metrics()
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'system_metrics': metrics
        })
    else:
        return jsonify({
            'status': 'initializing',
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/api/status', methods=['GET'])
def status():
    """Detailed system status endpoint."""
    if not session_manager:
        return jsonify({
            'status': 'error',
            'message': 'System not initialized'
        }), 503
    
    try:
        return jsonify({
            'status': 'success',
            'system_status': session_manager.get_system_metrics(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/session', methods=['POST'])
def create_session():
    """Create a new recommendation session."""
    if not session_manager:
        return jsonify({
            'status': 'error',
            'message': 'System not initialized'
        }), 503
    
    try:
        # Extract client information
        user_agent = request.headers.get('User-Agent', '')
        ip_address = request.remote_addr or 'unknown'
        
        # Get optional user ID from request - handle JSON parsing errors
        try:
            data = request.get_json(force=True, silent=True) or {}
        except Exception as json_error:
            logger.warning(f"JSON parsing error in session creation: {json_error}")
            data = {}
        
        user_id = data.get('user_id')
        
        # Create session with initial recommendations
        session_id, recommendations = session_manager.create_session(
            user_agent=user_agent,
            ip_address=ip_address,
            user_id=user_id
        )
        
        # Sanitize recommendations for JSON serialization
        sanitized_recommendations = [sanitize_watch_for_json(rec) for rec in recommendations]
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'recommendations': sanitized_recommendations,
            'timestamp': datetime.now().isoformat()
        })
        
    except ValueError as e:
        # Handle specific validation errors (e.g., session limit)
        if "Maximum concurrent sessions" in str(e):
            return jsonify({
                'status': 'error',
                'message': str(e),
                'error_code': 'SESSION_LIMIT_EXCEEDED'
            }), 429
        else:
            return jsonify({
                'status': 'error',
                'message': str(e),
                'error_code': 'VALIDATION_ERROR'
            }), 400
            
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to create session',
            'error_code': 'SESSION_CREATION_FAILED'
        }), 500

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """Get recommendations for an existing session."""
    if not session_manager:
        return jsonify({
            'status': 'error',
            'message': 'System not initialized'
        }), 503
    
    try:
        # Get session ID from header
        session_id = request.headers.get('X-Session-ID')
        if not session_id:
            return jsonify({
                'status': 'error',
                'message': 'Session ID required in X-Session-ID header',
                'error_code': 'MISSING_SESSION_ID'
            }), 400
        
        # Get recommendations
        recommendations = session_manager.get_recommendations(session_id)
        
        # Sanitize for JSON serialization
        sanitized_recommendations = [sanitize_watch_for_json(rec) for rec in recommendations]
        
        return jsonify({
            'status': 'success',
            'recommendations': sanitized_recommendations,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        })
        
    except ValueError as e:
        # Handle specific validation errors
        error_msg = str(e)
        if "not found or expired" in error_msg:
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'error_code': 'SESSION_NOT_FOUND'
            }), 404
        elif "Rate limit exceeded" in error_msg:
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'error_code': 'RATE_LIMIT_EXCEEDED'
            }), 429
        else:
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'error_code': 'VALIDATION_ERROR'
            }), 400
            
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to get recommendations',
            'error_code': 'RECOMMENDATION_FAILED'
        }), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for a watch."""
    if not session_manager:
        return jsonify({
            'status': 'error',
            'message': 'System not initialized'
        }), 503
    
    try:
        # Get session ID from header
        session_id = request.headers.get('X-Session-ID')
        if not session_id:
            return jsonify({
                'status': 'error',
                'message': 'Session ID required in X-Session-ID header',
                'error_code': 'MISSING_SESSION_ID'
            }), 400
        
        # Parse request data
        try:
            data = request.get_json(force=True, silent=True)
            if not data:
                return jsonify({
                    'status': 'error',
                    'message': 'JSON data required',
                    'error_code': 'MISSING_DATA'
                }), 400
        except Exception as json_error:
            logger.warning(f"JSON parsing error in feedback submission: {json_error}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid JSON data',
                'error_code': 'INVALID_JSON'
            }), 400
        
        watch_id = data.get('watch_id')
        feedback = data.get('feedback')
        
        if watch_id is None:
            return jsonify({
                'status': 'error',
                'message': 'watch_id is required',
                'error_code': 'MISSING_WATCH_ID'
            }), 400
        
        if feedback not in ['like', 'dislike']:
            return jsonify({
                'status': 'error',
                'message': 'feedback must be "like" or "dislike"',
                'error_code': 'INVALID_FEEDBACK'
            }), 400
        
        # Submit feedback
        success = session_manager.add_feedback(session_id, watch_id, feedback)
        
        return jsonify({
            'status': 'success',
            'message': f'Feedback "{feedback}" recorded for watch {watch_id}',
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        })
        
    except ValueError as e:
        # Handle specific validation errors
        error_msg = str(e)
        if "not found or expired" in error_msg:
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'error_code': 'SESSION_NOT_FOUND'
            }), 404
        else:
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'error_code': 'VALIDATION_ERROR'
            }), 400
            
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to submit feedback',
            'error_code': 'FEEDBACK_FAILED'
        }), 500

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session_info(session_id: str):
    """Get detailed information about a session."""
    if not session_manager:
        return jsonify({
            'status': 'error',
            'message': 'System not initialized'
        }), 503
    
    try:
        session_info = session_manager.get_session_info(session_id)
        
        if session_info is None:
            return jsonify({
                'status': 'error',
                'message': 'Session not found or expired',
                'error_code': 'SESSION_NOT_FOUND'
            }), 404
        
        return jsonify({
            'status': 'success',
            'session_info': session_info,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to get session info',
            'error_code': 'SESSION_INFO_FAILED'
        }), 500

@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_session(session_id: str):
    """Manually cleanup a session."""
    if not session_manager:
        return jsonify({
            'status': 'error',
            'message': 'System not initialized'
        }), 503
    
    try:
        success = session_manager.manual_cleanup_session(session_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Session cleaned up successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Session not found',
                'error_code': 'SESSION_NOT_FOUND'
            }), 404
            
    except Exception as e:
        logger.error(f"Error cleaning up session: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to cleanup session',
            'error_code': 'CLEANUP_FAILED'
        }), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get system analytics (admin endpoint)."""
    if not session_manager:
        return jsonify({
            'status': 'error',
            'message': 'System not initialized'
        }), 503
    
    try:
        limit = request.args.get('limit', 100, type=int)
        analytics = session_manager.get_session_analytics(limit)
        
        return jsonify({
            'status': 'success',
            'analytics': analytics,
            'count': len(analytics),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to get analytics',
            'error_code': 'ANALYTICS_FAILED'
        }), 500

@app.route('/api/session/reset', methods=['POST'])
def reset_session():
    """Reset current session - creates new session and cleans up old one."""
    start_time = time.time()
    
    try:
        session_id = request.headers.get('X-Session-ID')
        if not session_id:
            return jsonify({
                'error': 'Missing X-Session-ID header',
                'error_code': 'MISSING_SESSION_ID'
            }), 400
        
        # Use the session manager's reset method
        new_session_id, initial_recommendations = session_manager.reset_session(session_id)
        
        response_time = (time.time() - start_time) * 1000
        
        logger.info(f"🔄 Session reset: {session_id} → {new_session_id} ({response_time:.1f}ms)")
        
        return jsonify({
            'success': True,
            'new_session_id': new_session_id,
            'recommendations': initial_recommendations,
            'response_time_ms': response_time
        })
        
    except ValueError as e:
        response_time = (time.time() - start_time) * 1000
        logger.error(f"❌ Reset session validation error: {e}")
        return jsonify({
            'error': str(e),
            'error_code': 'VALIDATION_ERROR',
            'response_time_ms': response_time
        }), 400
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        logger.error(f"❌ Reset session error: {e}")
        return jsonify({
            'error': 'Failed to reset session',
            'error_code': 'RESET_ERROR',
            'response_time_ms': response_time
        }), 500

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    shutdown_system()

def shutdown_system():
    """Graceful system shutdown."""
    global session_manager
    
    if session_manager:
        logger.info("🛑 Shutting down session manager...")
        session_manager.shutdown()
        session_manager = None
    
    logger.info("✅ Shutdown complete")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(shutdown_system)

if __name__ == '__main__':
    logger.info("🚀 Starting Production Watch Finder API...")
    
    # Initialize system
    if not initialize_system():
        logger.error("❌ Failed to initialize system, exiting...")
        sys.exit(1)
    
    logger.info(f"✅ System initialized, starting server on port {DEFAULT_PORT}...")
    
    try:
        # Run production server
        app.run(
            host='0.0.0.0',
            port=DEFAULT_PORT,
            debug=DEBUG,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
    finally:
        shutdown_system() 