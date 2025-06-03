#!/usr/bin/env python3
"""
Modern Flask API for Watch Recommendation System
===============================================

A cutting-edge API server featuring:
- Async/await support with modern Python patterns
- Advanced error handling and monitoring
- Real-time performance metrics
- Comprehensive logging and debugging
- Production-ready security features
- Health monitoring and diagnostics

Author: Watch Finder AI Team  
Version: 3.0.0
"""

import os
import sys
import asyncio
import pickle
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Import our modern components
from models.modern_session_manager import ModernSessionManager

# Configure enhanced logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG for detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('watch_recommendation_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app with modern configuration
app = Flask(__name__)

# Enhanced CORS configuration for development and production
CORS(app, 
     origins=[
         'http://localhost:3000', 
         'http://localhost:3001', 
         'http://localhost:8080',
         'http://localhost:8081',
         'http://127.0.0.1:3000', 
         'http://127.0.0.1:3001',
         'http://127.0.0.1:8080',
         'http://127.0.0.1:8081',
         'http://192.168.0.209:3000',
         'http://192.168.0.209:3001', 
         'http://192.168.0.209:8080',
         'http://192.168.0.209:8081',
         'http://192.168.0.209:5173',
         'http://192.168.0.209:4173',
         'https://watchrecomender.netlify.app',  # Production Netlify domain
         'https://watch-finder.netlify.app',     # Alternative Netlify domain
         'https://watch-recommender.netlify.app', # Another possible Netlify domain
         '*'  # Allow all origins temporarily to debug
     ],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=True
)

# Global variables and async event loop
session_manager: Optional[ModernSessionManager] = None
app_start_time = time.time()
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

def get_project_root():
    """Get the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def load_watch_data():
    """Load watch embeddings and metadata with modern error handling."""
    project_root = get_project_root()
    embeddings_path = os.path.join(project_root, 'data', 'watch_embeddings.pkl')
    metadata_path = os.path.join(project_root, 'data', 'watch_metadata.pkl')
    
    logger.info(f"🔍 Loading embeddings from: {embeddings_path}")
    logger.info(f"🔍 Loading metadata from: {metadata_path}")
    
    # Check if files exist
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    logger.info("✅ Found deployment-ready embeddings and metadata files")
    
    # Load embeddings
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    # Load metadata
    with open(metadata_path, 'rb') as f:
        watch_data = pickle.load(f)
    
    # Handle different data formats
    if isinstance(embeddings_data, dict):
        logger.info("📊 Processing structured embeddings data")
        embeddings = embeddings_data.get('embeddings', embeddings_data)
        if hasattr(embeddings, 'shape'):
            embeddings = embeddings
        else:
            embeddings = np.array(embeddings)
    else:
        logger.info("📊 Using raw embeddings data")
        embeddings = embeddings_data
    
    # Validate data consistency
    if len(embeddings) != len(watch_data):
        logger.warning(f"⚠️  Dimension mismatch: {len(embeddings)} embeddings vs {len(watch_data)} watches")
        # Use the minimum length to avoid index errors
        min_length = min(len(embeddings), len(watch_data))
        embeddings = embeddings[:min_length]
        watch_data = watch_data[:min_length]
        logger.info(f"🔧 Adjusted to {min_length} consistent entries")
    
    logger.info(f"📊 Loaded {len(watch_data)} watches with {embeddings.shape[1]}D embeddings")
    
    return embeddings, watch_data

def initialize_system():
    """Initialize the modern recommendation system."""
    global session_manager
    
    try:
        logger.info("🚀 Initializing Modern Watch Recommendation System...")
        
        # Load data
        embeddings, watch_data = load_watch_data()
        
        # Pre-normalize embeddings for better performance
        logger.info("🔧 Pre-normalizing embeddings for optimal performance...")
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        logger.info(f"✅ Pre-normalized {len(embeddings)} embeddings")
        
        # Initialize modern session manager
        session_manager = ModernSessionManager(
            embeddings=embeddings_normalized,
            watch_data=watch_data,
            embeddings_pre_normalized=True,
            use_faiss=False  # Disable FAISS for Railway compatibility
        )
        
        logger.info("✅ Modern Session Manager initialized successfully!")
        logger.info(f"📊 Base dataset: {len(watch_data)} watches")
        logger.info(f"📐 Embedding dimension: {embeddings.shape[1]}")
        logger.info(f"🔗 Enhanced specifications: ✅")
        logger.info(f"📄 Modern ML algorithms: ✅")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        logger.error(traceback.format_exc())
        return False

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'error_type': '404_not_found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'error_type': '500_internal_error'
    }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    logger.error(traceback.format_exc())
    return jsonify({
        'status': 'error',
        'message': 'An unexpected error occurred',
        'error_type': 'unhandled_exception'
    }), 500

# Health and monitoring endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    """Enhanced health check with comprehensive system status."""
    try:
        if session_manager is None:
            return jsonify({
                'status': 'unhealthy',
                'message': 'Session manager not initialized',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        # Get comprehensive system status
        system_status = session_manager.get_system_status()
        
        return jsonify({
            'status': 'healthy',
            'system_status': system_status,
            'api_version': '3.0.0',
            'uptime_seconds': time.time() - app_start_time,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get detailed system metrics."""
    try:
        if session_manager is None:
            return jsonify({'error': 'Session manager not initialized'}), 503
        
        system_status = session_manager.get_system_status()
        popular_watches = session_manager.get_popular_watches(limit=10)
        session_analytics = session_manager.get_session_analytics(limit=50)
        
        return jsonify({
            'status': 'success',
            'system_metrics': system_status,
            'popular_watches': popular_watches,
            'recent_session_analytics': session_analytics,
            'api_performance': {
                'uptime_seconds': time.time() - app_start_time,
                'api_version': '3.0.0'
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Core API endpoints
@app.route('/api/start-session', methods=['POST'])
def start_session():
    """Start a new recommendation session with modern async support."""
    try:
        if session_manager is None:
            return jsonify({'status': 'error', 'message': 'System not initialized'}), 503
        
        # Get request data
        data = request.get_json() or {}
        num_seeds = data.get('num_seeds', 7)
        user_agent = request.headers.get('User-Agent', '')
        ip_address = request.remote_addr or 'unknown'
        
        # Create session asynchronously
        async def create_session_async():
            session_id = await session_manager.create_session(
                user_agent=user_agent,
                ip_address=ip_address,
                metadata={
                    'requested_seeds': num_seeds,
                    'client_info': {
                        'user_agent': user_agent,
                        'ip_address': ip_address
                    }
                }
            )
            
            # Get smart seeds
            seeds = await session_manager.get_smart_seeds(session_id, num_seeds)
            
            return session_id, seeds
        
        # Run async operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            session_id, seeds = loop.run_until_complete(create_session_async())
        finally:
            loop.close()
        
        logger.info(f"🌱 Started session {session_id} with {len(seeds)} seeds")
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'seeds': seeds,
            'algorithm_version': '3.0.0',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Failed to start session: {str(e)}',
            'error_type': 'session_creation_failed'
        }), 500

@app.route('/api/get-recommendations', methods=['POST'])
def get_recommendations():
    """Get personalized watch recommendations using modern ML techniques."""
    try:
        if not session_manager:
            return jsonify({
                'status': 'error',
                'message': 'Session manager not initialized'
            }), 500
        
        data = request.json or {}
        
        # Extract request parameters
        session_id = data.get('session_id')
        liked_indices = data.get('liked_indices', [])
        disliked_indices = data.get('disliked_indices', [])
        current_candidates = data.get('current_candidates', [])
        num_recommendations = data.get('num_recommendations', 7)
        step = data.get('step', 1)
        
        if not session_id:
            return jsonify({
                'status': 'error',
                'message': 'session_id is required'
            }), 400

        async def get_recommendations_async():
            try:
                # Create recommendation request
                from models.modern_recommendation_engine import RecommendationRequest
                request_obj = RecommendationRequest(
                    user_id=session_id,
                    liked_indices=liked_indices,
                    disliked_indices=disliked_indices,
                    current_candidates=current_candidates,
                    num_recommendations=num_recommendations,
                    exploration_factor=0.3,
                    diversity_threshold=0.7
                )
                
                # Get recommendations
                result = await session_manager.recommendation_engine.get_recommendations(request_obj)
                
                # 🆕 ENHANCED STRATEGY AND MODE INFORMATION
                # Get user state for detailed mode information
                user_state = session_manager.recommendation_engine._get_user_state(session_id)
                feedback_count = len(user_state['feedback_history'])
                likes_count = sum(1 for f in user_state['feedback_history'] if f['type'] == 'like')
                preference_clusters = user_state['preference_clusters']
                
                # Calculate cluster strength and details
                cluster_details = []
                total_cluster_strength = 0
                for i, cluster in enumerate(preference_clusters):
                    strength = cluster.get_strength()
                    total_cluster_strength += strength
                    cluster_details.append({
                        'cluster_id': i,
                        'strength': strength,
                        'preference_count': len(cluster.preferences),
                        'last_updated': cluster.last_updated.isoformat(),
                        'activation_count': cluster.activation_count
                    })
                
                # Determine current mode with detailed explanation
                current_strategy = result.algorithm_used
                strategy_explanation = ""
                
                if current_strategy == 'cold_start':
                    strategy_explanation = f"Cold start mode - building initial preference profile ({feedback_count} interactions)"
                elif current_strategy == 'preference_based':
                    strategy_explanation = f"Preference-based mode - using {len(preference_clusters)} preference clusters (strength: {total_cluster_strength:.2f})"
                elif current_strategy == 'exploration':
                    strategy_explanation = f"Exploration mode - discovering new styles and brands"
                elif current_strategy == 'hybrid':
                    strategy_explanation = f"Hybrid mode - balancing preferences ({likes_count} likes) with exploration"
                
                # Check if clustering is being used
                clustering_status = {
                    'is_using_clusters': len(preference_clusters) > 0,
                    'cluster_count': len(preference_clusters),
                    'total_cluster_strength': total_cluster_strength,
                    'cluster_details': cluster_details
                }
                
                # Check exploration level
                seen_watches = len(user_state['seen_watches'])
                total_watches = len(session_manager.recommendation_engine.watch_data)
                exploration_percentage = (seen_watches / total_watches * 100) if total_watches > 0 else 0
                
                exploration_status = {
                    'exploration_percentage': exploration_percentage,
                    'seen_watches': seen_watches,
                    'total_watches': total_watches,
                    'exploration_level': 'high' if exploration_percentage > 50 else 'medium' if exploration_percentage > 20 else 'low',
                    'engagement_level': user_state['engagement_level']
                }
                
                return {
                    'status': 'success',
                    'recommendations': result.watches,
                    'step': step,
                    'session_id': session_id,
                    'generated_at': datetime.now().isoformat(),
                    'processing_time': result.processing_time,
                    'confidence_scores': result.confidence_scores,
                    'diversity_score': result.diversity_score,
                    'exploration_rate': result.exploration_rate,
                    'user_profile_summary': result.user_profile_summary,
                    'next_exploration_suggestions': result.next_exploration_suggestions,
                    
                    # 🆕 ENHANCED MODE AND STRATEGY INFORMATION
                    'algorithm_used': current_strategy,
                    'strategy_explanation': strategy_explanation,
                    'clustering_status': clustering_status,
                    'exploration_status': exploration_status,
                    
                    # 🆕 DETAILED SYSTEM STATE
                    'system_state': {
                        'current_mode': current_strategy,
                        'is_clustering': len(preference_clusters) > 0,
                        'is_exploring': current_strategy in ['exploration', 'hybrid'] or exploration_percentage < 30,
                        'preference_strength': 'strong' if total_cluster_strength > 0.7 else 'medium' if total_cluster_strength > 0.4 else 'weak',
                        'recommendation_confidence': 'high' if current_strategy == 'preference_based' else 'medium' if current_strategy == 'hybrid' else 'exploratory',
                        'feedback_count': feedback_count,
                        'likes_count': likes_count,
                        'variant_filtering_active': True
                    },
                    
                    # 🆕 MODE TRANSITION INDICATORS
                    'mode_indicators': {
                        'ready_for_clustering': likes_count >= 3 and feedback_count >= 5,
                        'strong_preferences_detected': total_cluster_strength > 0.6,
                        'exploration_recommended': exploration_percentage < 15 or likes_count < 2,
                        'can_use_preferences': len(preference_clusters) > 0 and total_cluster_strength > 0.3
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in get_recommendations_async: {e}")
                logger.error(traceback.format_exc())
                raise e

        # Run async function
        result = asyncio.run(get_recommendations_async())
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/add-feedback', methods=['POST'])
def add_feedback():
    """Add user feedback with modern async processing."""
    try:
        if session_manager is None:
            return jsonify({'status': 'error', 'message': 'System not initialized'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
        
        session_id = data.get('session_id')
        watch_index = data.get('watch_index')
        feedback_type = data.get('feedback_type')
        confidence = data.get('confidence', 0.8)
        
        if not all([session_id, watch_index is not None, feedback_type]):
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields: session_id, watch_index, feedback_type'
            }), 400
        
        # Validate watch_index is an integer
        try:
            watch_index = int(watch_index)
        except (ValueError, TypeError):
            return jsonify({
                'status': 'error',
                'message': f'watch_index must be an integer, received: {type(watch_index).__name__}'
            }), 400
        
        if feedback_type not in ['like', 'dislike']:
            return jsonify({
                'status': 'error',
                'message': 'feedback_type must be "like" or "dislike"'
            }), 400
        
        # Add feedback synchronously for now to avoid event loop issues
        success = session_manager.add_feedback_sync(
            session_id=session_id,
            watch_index=watch_index,
            feedback_type=feedback_type,
            confidence=confidence
        )
        
        if success:
            logger.info(f"👍 Added {feedback_type} feedback for watch {watch_index} in session {session_id}")
            return jsonify({
                'status': 'success',
                'message': 'Feedback added successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to add feedback - invalid session'
            }), 400
            
    except Exception as e:
        logger.error(f"Error adding feedback: {e}")
        logger.error(traceback.format_exc())  # Add full traceback
        return jsonify({
            'status': 'error',
            'message': f'Failed to add feedback: {str(e)}'
        }), 500

@app.route('/api/session-info/<session_id>', methods=['GET'])
def get_session_info(session_id):
    """Get detailed session information."""
    try:
        if session_manager is None:
            return jsonify({'status': 'error', 'message': 'System not initialized'}), 503
        
        session_info = session_manager.get_session_info(session_id)
        
        if session_info is None:
            return jsonify({
                'status': 'error',
                'message': 'Session not found or expired'
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
            'message': str(e)
        }), 500

@app.route('/api/cleanup-session/<session_id>', methods=['DELETE'])
def cleanup_session(session_id):
    """Manually cleanup a session."""
    try:
        if session_manager is None:
            return jsonify({'status': 'error', 'message': 'System not initialized'}), 503
        
        async def cleanup_async():
            return await session_manager.cleanup_session(session_id)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            success = loop.run_until_complete(cleanup_async())
        finally:
            loop.close()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Session cleaned up successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Session not found'
            }), 404
            
    except Exception as e:
        logger.error(f"Error cleaning up session: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Legacy compatibility endpoints (for backward compatibility)
@app.route('/api/get-series/<int:watch_index>', methods=['GET'])
def get_series(watch_index):
    """Get all watches from the same series as the specified watch."""
    try:
        if not session_manager:
            return jsonify({
                'status': 'error',
                'message': 'Session manager not initialized'
            }), 500
        
        # Get watch data directly from the recommendation engine
        watch_data = session_manager.recommendation_engine.watch_data
        
        if watch_index >= len(watch_data):
            return jsonify({
                'status': 'error', 
                'message': 'Invalid watch index'
            }), 400
        
        target_watch = watch_data[watch_index]
        series_name = target_watch.get('specs', {}).get('serie', '')
        brand = target_watch.get('brand', '')
        
        if not series_name or series_name in ['-', 'All', '']:
            return jsonify({
                'status': 'success',
                'series_watches': [target_watch],
                'series_name': 'Individual Watch',
                'count': 1
            })
        
        # Find all watches in the same series and brand
        series_watches = []
        for i, watch in enumerate(watch_data):
            watch_series = watch.get('specs', {}).get('serie', '')
            watch_brand = watch.get('brand', '')
            
            if (watch_brand.lower() == brand.lower() and 
                watch_series.lower() == series_name.lower() and 
                watch_series not in ['-', 'All', '']):
                watch_copy = watch.copy()
                watch_copy['index'] = i
                series_watches.append(watch_copy)
        
        return jsonify({
            'status': 'success',
            'series_watches': series_watches,
            'series_name': series_name,
            'count': len(series_watches)
        })
    
    except Exception as e:
        logger.error(f"Error getting series: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/variant-filtering-stats', methods=['GET'])
def get_variant_filtering_stats():
    """Get variant filtering statistics from the modern recommendation engine."""
    try:
        if not session_manager:
            return jsonify({
                'status': 'error',
                'message': 'Session manager not initialized'
            }), 500
        
        variant_stats = session_manager.recommendation_engine.get_variant_filtering_stats()
        
        return jsonify({
            'status': 'success',
            'variant_filtering_stats': variant_stats,
            'info': 'Modern recommendation engine variant filtering statistics'
        })
    
    except Exception as e:
        logger.error(f"Error getting variant filtering stats: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/get-variants/<int:watch_index>', methods=['GET', 'OPTIONS'])
def get_variants(watch_index):
    """Get all variants of a specific watch (different colors, sizes, etc. of the same model)."""
    if request.method == 'OPTIONS':
        # Handle CORS preflight request
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
        return response
        
    try:
        if not session_manager:
            return jsonify({
                'status': 'error',
                'message': 'Session manager not initialized'
            }), 500
        
        watch_data = session_manager.recommendation_engine.watch_data
        
        if watch_index < 0 or watch_index >= len(watch_data):
            return jsonify({
                'status': 'error',
                'message': 'Invalid watch index'
            }), 400
        
        target_watch = watch_data[watch_index]
        target_brand = target_watch.get('brand', '')
        target_model = target_watch.get('model', target_watch.get('model_name', ''))
        
        # Get the signature for this watch
        if hasattr(session_manager.recommendation_engine, 'watch_signatures') and watch_index in session_manager.recommendation_engine.watch_signatures:
            target_signature = session_manager.recommendation_engine.watch_signatures[watch_index]
            target_brand_norm, target_model_norm = target_signature
        else:
            # Fallback to direct string matching
            target_brand_norm = session_manager.recommendation_engine._normalize_brand_name(target_brand)
            target_model_norm = session_manager.recommendation_engine._extract_base_model_name(target_model)
        
        # Find all watches with the same brand-model signature
        variants = []
        for i, watch in enumerate(watch_data):
            if hasattr(session_manager.recommendation_engine, 'watch_signatures') and i in session_manager.recommendation_engine.watch_signatures:
                watch_signature = session_manager.recommendation_engine.watch_signatures[i]
                if watch_signature == target_signature:
                    watch_copy = watch.copy()
                    watch_copy['index'] = i
                    watch_copy['is_target'] = (i == watch_index)
                    
                    # Add variant-specific metadata
                    watch_copy['variant_info'] = {
                        'brand_normalized': target_brand_norm,
                        'model_normalized': target_model_norm,
                        'signature': f"{target_brand_norm} - {target_model_norm}"
                    }
                    
                    # Extract what makes this variant unique (color, size, material, etc.)
                    variant_identifiers = []
                    
                    # Check specs for variant details
                    specs = watch.get('specs', {})
                    if specs:
                        # Case material variants
                        case_material = specs.get('case_material', '')
                        if case_material and case_material != '-':
                            variant_identifiers.append(f"Case: {case_material}")
                        
                        # Diameter variants
                        diameter = specs.get('diameter_mm', '')
                        if diameter and diameter != '-':
                            variant_identifiers.append(f"Size: {diameter}mm")
                        
                        # Dial color variants
                        dial_color = specs.get('dial_color', '')
                        if dial_color and dial_color != '-':
                            variant_identifiers.append(f"Dial: {dial_color}")
                        
                        # Reference/model variants
                        reference = specs.get('reference', '')
                        if reference and reference != '-':
                            variant_identifiers.append(f"Ref: {reference}")
                    
                    # Extract variant info from model name differences
                    model_variants = []
                    original_model = target_watch.get('model', '')
                    current_model = watch.get('model', '')
                    
                    if original_model != current_model:
                        model_variants.append(f"Model: {current_model}")
                    
                    watch_copy['variant_details'] = variant_identifiers + model_variants
                    variants.append(watch_copy)
        
        # Sort variants: target watch first, then by model name
        variants.sort(key=lambda w: (not w['is_target'], w.get('model', '')))
        
        response = jsonify({
            'status': 'success',
            'watch_index': watch_index,
            'target_watch': target_watch,
            'brand': target_brand,
            'model': target_model,
            'signature': f"{target_brand_norm} - {target_model_norm}",
            'variant_count': len(variants),
            'variants': variants,
            'info': f"Found {len(variants)} variants of {target_brand} {target_model_norm}"
        })
        
        # Add CORS headers to the response
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    
    except Exception as e:
        logger.error(f"Error getting variants for watch {watch_index}: {e}")
        logger.error(traceback.format_exc())
        error_response = jsonify({
            'status': 'error',
            'message': str(e)
        })
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        return error_response, 500

if __name__ == '__main__':
    print("🚀 Starting Modern Watch Recommendation API Server")
    print("=" * 60)
    
    # Initialize the system
    if not initialize_system():
        print("❌ Failed to initialize system. Exiting.")
        sys.exit(1)
    
    print("✅ System initialized successfully!")
    print("🌟 Modern ML algorithms active")
    print("🔄 Async/await processing enabled")
    print("📊 Advanced analytics active")
    print("=" * 60)
    print("🌐 Starting Flask server...")
    
    # Get port from environment (Railway sets PORT automatically)
    port = int(os.environ.get('PORT', 5001))
    
    # Determine if running in production
    is_production = os.environ.get('RAILWAY_ENVIRONMENT') == 'production' or \
                   os.environ.get('FLASK_ENV') == 'production'
    
    print(f"🌍 Port: {port}")
    print(f"⚙️  Environment: {'Production' if is_production else 'Development'}")
    
    # Start the Flask application
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=False  # Disabled debug to prevent double initialization
    ) 