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
import traceback

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.makedirs('logs', exist_ok=True)
# Import configuration
from config import (
    DEFAULT_PORT, DEBUG, DATA_DIR
)

# Import production session manager
try:
    from models.fast_linucb_engine import FastLinUCBEngine
    from models.production_session_manager import ProductionSessionManager
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

# Enhanced CORS configuration for production
CORS(app, 
     origins=[
         'https://watchrecomender.netlify.app',
         'https://www.watchrecomender.netlify.app',
         'https://deploy-preview-*--watchrecomender.netlify.app',
         'http://localhost:3000',
         'http://localhost:5173',
         'http://localhost:8080',
         'http://127.0.0.1:3000',
         'http://127.0.0.1:5173',
         'http://127.0.0.1:8080'
     ],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization', 'X-Session-ID'],
     supports_credentials=True)

# Handle proxy headers if behind reverse proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Global state
session_manager: Optional[ProductionSessionManager] = None
engine: Optional[FastLinUCBEngine] = None

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
    global session_manager, engine
    
    try:
        logger.info("🚀 Initializing Production Watch Recommendation System...")
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Initialize FAST engine with precomputed embeddings!
        engine = FastLinUCBEngine(
            alpha=0.15,
            batch_size=5,
            max_experts=4,
            similarity_threshold=0.95,
            data_dir='data'
        )
        
        # Initialize session manager with optimized engine
        session_manager = ProductionSessionManager(
            data_dir='data',
            session_timeout_minutes=60,
            max_requests_per_minute=60,
            enable_persistence=True,
            linucb_engine=engine
        )
        
        logger.info("✅ Optimized LinUCB engine and session manager initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        logger.error(traceback.format_exc())
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
    
    # Always allow OPTIONS requests (CORS preflight)
    if request.method == 'OPTIONS':
        return
    
    # Allow health, ready, and status endpoints even during initialization
    allowed_endpoints = ['health', 'ready', 'status']
    
    # Check if system is initialized for other endpoints
    if not session_manager and request.endpoint not in allowed_endpoints:
        return jsonify({
            'status': 'error',
            'message': 'System still initializing, please try again in a moment',
            'error_code': 'SYSTEM_INITIALIZING'
        }), 503

@app.after_request
def after_request(response):
    """Log response and update metrics."""
    if hasattr(g, 'start_time'):
        duration_ms = (datetime.now() - g.start_time).total_seconds() * 1000
        # Only log non-health check requests to reduce noise
        if request.path != '/api/health':
            logger.info(f"{request.method} {request.path} - {response.status_code} - {duration_ms:.1f}ms")
        else:
            # Log health checks at debug level
            logger.debug(f"HEALTH CHECK - {response.status_code} - {duration_ms:.1f}ms")
    
    return response

# Core API endpoints
@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint - always returns 200 for container health."""
    try:
        # Basic health check - always return 200 for container health
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'app_name': 'watch_finder_v2',
            'version': '2.0.0',
            'system_ready': session_manager is not None
        }), 200
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        # Even if there's an error, return 200 for basic container health
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 200

@app.route('/api/ready', methods=['GET'])
def ready():
    """Simple readiness check - returns 200 only when system is fully initialized."""
    if session_manager and engine:
        return jsonify({
            'status': 'ready',
            'timestamp': datetime.now().isoformat()
        }), 200
    else:
        return jsonify({
            'status': 'initializing',
            'timestamp': datetime.now().isoformat(),
            'system_initialized': session_manager is not None,
            'engine_initialized': engine is not None
        }), 503

@app.route('/api/status', methods=['GET'])
def status():
    """Detailed system status endpoint."""
    try:
        response_data = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'system_initialized': session_manager is not None,
            'engine_initialized': engine is not None
        }
        
        # Try to get metrics if system is available
        if session_manager:
            try:
                metrics = session_manager.get_system_metrics()
                response_data['system_metrics'] = metrics
            except Exception as e:
                logger.warning(f"Failed to get system metrics: {e}")
                response_data['metrics_error'] = str(e)
            
            # Add expert stats if engine is available
            if engine:
                try:
                    response_data['expert_stats'] = engine.get_expert_stats()
                except Exception as e:
                    logger.warning(f"Failed to get expert stats: {e}")
                    response_data['expert_stats_error'] = str(e)
        else:
            response_data['status'] = 'initializing'
            response_data['message'] = 'System not yet initialized'
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
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
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': 'Failed to create session',
            'error_code': 'SESSION_CREATION_FAILED'
        }), 500

@app.route('/api/recommendations', methods=['GET', 'POST'])
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
            # Try to get from query params
            session_id = request.args.get('session_id')
            if not session_id and request.method == 'POST':
                # Try to get from request body for POST requests
                try:
                    data = request.get_json(force=True, silent=True)
                    if data:
                        session_id = data.get('session_id')
                except:
                    pass
            
            if not session_id:
                return jsonify({
                    'status': 'error',
                    'message': 'Session ID required in X-Session-ID header, session_id query parameter, or request body',
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
        # Get session ID from header, then query params, then request body
        session_id = request.headers.get('X-Session-ID')
        if not session_id:
            session_id = request.args.get('session_id') # Check query parameters
            if not session_id:
                # Try to get from request body for POST requests
                try:
                    data_for_session_id = request.get_json(force=True, silent=True)
                    if data_for_session_id:
                        session_id = data_for_session_id.get('session_id')
                except Exception: # pylint: disable=broad-except
                    pass  # Ignore if body is not JSON or session_id is not there
        
        if not session_id:
            return jsonify({
                'status': 'error',
                'message': 'Session ID required in X-Session-ID header, session_id query parameter, or request body',
                'error_code': 'MISSING_SESSION_ID'
            }), 400
        
        # Parse request data (feedback data, not session_id data)
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
        
        # Convert feedback to boolean for the engine
        liked = (feedback == 'like')
        
        # Submit feedback
        success = session_manager.update_feedback(session_id, watch_id, liked)
        
        # Get updated expert stats after feedback
        expert_stats = engine.get_expert_stats() if engine else None
        
        # Log expert stats after feedback
        if expert_stats:
            logger.info(f"📊 Expert stats after feedback - Total experts: {expert_stats.get('total_experts', 0)}, " +
                       f"Total sessions: {expert_stats.get('total_sessions', 0)}, " +
                       f"Avg likes per expert: {expert_stats.get('avg_likes_per_expert', 0):.1f}")
        
        return jsonify({
            'status': 'success',
            'message': f'Feedback "{feedback}" recorded for watch {watch_id}',
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'expert_stats': expert_stats
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

@app.route('/api/liked-watches', methods=['GET'])
def get_liked_watches():
    """Get user's liked watches."""
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
        
        # Get liked watches from the session
        session_data = session_manager.get_session_info(session_id)
        if not session_data:
            return jsonify({
                'status': 'error',
                'message': 'Session not found or expired',
                'error_code': 'SESSION_NOT_FOUND'
            }), 404
        
        # Get liked watch IDs from the engine
        liked_watch_ids = []
        if engine and session_id in engine.session_liked_watches:
            liked_watch_ids = engine.session_liked_watches[session_id]
        
        # Get watch details for liked watches
        liked_watches = []
        for watch_id in liked_watch_ids:
            if watch_id in engine.watch_data:
                watch = sanitize_watch_for_json(engine.watch_data[watch_id])
                liked_watches.append(watch)
        
        return jsonify({
            'status': 'success',
            'liked_watches': liked_watches,
            'count': len(liked_watches),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting liked watches: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to get liked watches',
            'error_code': 'LIKED_WATCHES_FAILED'
        }), 500

@app.route('/api/filter-options', methods=['GET'])
def get_filter_options():
    """Get available filter options from watch metadata."""
    if not engine:
        return jsonify({
            'status': 'error',
            'message': 'System not initialized'
        }), 503
    
    try:
        # Extract unique values from watch metadata
        brands = set()
        case_materials = set()
        
        for watch_id, watch_data in engine.watch_data.items():
            # Extract brand
            if 'brand' in watch_data and watch_data['brand']:
                brands.add(watch_data['brand'])
            
            # Extract specs
            specs = watch_data.get('specs', {})
            
            # Case material
            if 'case_material' in specs and specs['case_material']:
                case_materials.add(specs['case_material'])
        
        # Convert sets to sorted lists with reasonable limits
        filter_options = {
            'brands': sorted(list(brands))[:50],  # Limit to top 50 brands
            'caseMaterials': sorted(list(case_materials)),
        }
        
        logger.info(f"✅ Generated filter options: {len(filter_options['brands'])} brands, {len(filter_options['caseMaterials'])} materials")
        
        return jsonify(filter_options)
        
    except Exception as e:
        logger.error(f"❌ Error getting filter options: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': 'Failed to get filter options',
            'error_code': 'FILTER_OPTIONS_FAILED'
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

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check system state."""
    try:
        debug_data = {
            'engine_type': type(engine).__name__ if engine else 'None',
            'engine_class': str(type(engine)) if engine else 'None',
            'session_manager_type': type(session_manager).__name__ if session_manager else 'None',
            'precomputed_file_exists': os.path.exists('data/precomputed_embeddings.pkl'),
            'precomputed_file_size_mb': round(os.path.getsize('data/precomputed_embeddings.pkl') / (1024*1024), 2) if os.path.exists('data/precomputed_embeddings.pkl') else 0,
            'data_dir_contents': os.listdir('data') if os.path.exists('data') else [],
            'current_working_dir': os.getcwd(),
            'python_path': sys.path[:3],  # First 3 entries
            'timestamp': datetime.now().isoformat()
        }
        
        # Add engine-specific info
        if engine:
            if hasattr(engine, 'dim'):
                debug_data['engine_dimension'] = engine.dim
            if hasattr(engine, 'available_watches'):
                debug_data['available_watches_count'] = len(engine.available_watches)
            if hasattr(engine, 'watch_data'):
                debug_data['watch_data_count'] = len(engine.watch_data)
        
        return jsonify({
            'status': 'success',
            'debug': debug_data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'debug': {
                'engine_exists': engine is not None,
                'session_manager_exists': session_manager is not None,
                'timestamp': datetime.now().isoformat()
            }
        })

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    shutdown_system()

def shutdown_system():
    """Graceful system shutdown."""
    global session_manager, engine
    
    if session_manager:
        logger.info("🛑 Shutting down session manager...")
        session_manager.shutdown()
        session_manager = None
    
    if engine:
        logger.info("🛑 Shutting down engine...")
        engine.shutdown()
        engine = None
    
    logger.info("✅ Shutdown complete")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(shutdown_system)

def initialize_system_background():
    """Initialize the system in the background after Flask starts."""
    import threading
    import time
    
    def init_worker():
        start_time = time.time()
        try:
            logger.info("🔄 Starting background system initialization...")
            if initialize_system():
                elapsed = time.time() - start_time
                logger.info(f"✅ Background system initialization completed successfully in {elapsed:.1f}s")
            else:
                elapsed = time.time() - start_time
                logger.error(f"❌ Background system initialization failed after {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ Background initialization error after {elapsed:.1f}s: {e}")
    
    # Start initialization in background thread
    init_thread = threading.Thread(target=init_worker, daemon=True)
    init_thread.start()
    logger.info("🚀 Background initialization thread started")
    return init_thread

if __name__ == '__main__':
    logger.info("🚀 Starting Production Watch Finder API...")
    logger.info("⚡ Quick start mode - initializing system in background...")
    
    # Start background initialization
    init_thread = initialize_system_background()
    
    logger.info(f"🌐 Starting server on port {DEFAULT_PORT}...")
    
    try:
        # Run production server immediately
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

# Initialize system when running in production (Railway/Gunicorn)
else:
    logger.info("🚀 Production mode detected - starting background initialization...")
    initialize_system_background() 