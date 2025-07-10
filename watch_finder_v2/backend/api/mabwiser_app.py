"""
Watch Finder Backend API - MABWiser Edition
==========================================

Flask API server for watch recommendation engine with MABWiser Thompson Sampling algorithm.
"""

import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import traceback
import uuid
from datetime import datetime
import pickle
import numpy as np
from typing import Dict, Any, List

# Add the backend directory to the path for imports
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

# Import the MABWiser engine
from models.mabwiser_engine import MABWiserEngine
from utils.json_utils import convert_numpy_to_python
from utils.filter_utils import should_include_watch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('mabwiser_api.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Environment-specific CORS configuration
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

# Always allow your Netlify domain for simplicity
allowed_origins = [
    "https://watchrecomender.netlify.app",
    "https://www.watchrecomender.netlify.app",
    "https://deploy-preview-*--watchrecomender.netlify.app",
    "http://localhost:3000",  # Local development
    "http://localhost:5173",  # Vite dev server
    "http://localhost:8080",  # Vite dev server alt port
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080"
]

if ENVIRONMENT == 'production':
    # Production CORS - restrict to specific domains
    NETLIFY_DOMAIN = os.getenv('NETLIFY_DOMAIN', 'watchrecomender.netlify.app')
    CORS(app, origins=allowed_origins, methods=['GET', 'POST', 'OPTIONS'], 
         allow_headers=['Content-Type', 'Authorization', 'X-Session-ID'], supports_credentials=True)
    logger.info(f"🔒 Production CORS enabled for: {allowed_origins}")
else:
    # Development CORS - allow all origins plus specific ones
    CORS(app, origins=["*"] + allowed_origins, methods=['GET', 'POST', 'OPTIONS'], 
         allow_headers=['Content-Type', 'Authorization', 'X-Session-ID'], supports_credentials=True)
    logger.info("🔓 Development CORS enabled (all origins + specific domains)")

# Global engine instance
engine = None

def initialize():
    global engine
    logger.info("🚀 Starting MABWiser engine initialization...")
    import time
    init_start = time.time()
    
    try:
        # Initialize MABWiser engine
        logger.info("⚙️ Creating MABWiserEngine instance...")
        engine = MABWiserEngine(
            batch_size=5,
            max_experts=4,
            similarity_threshold=0.85,  # Slightly lower threshold for more diverse experts
            data_dir='data'
        )
        
        init_time = time.time() - init_start
        logger.info(f"✅ MABWiser engine initialized successfully in {init_time:.2f}s")
        logger.info(f"📊 Engine stats: {len(engine.watch_data)} watches, {len(engine.available_watches)} available")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize MABWiser engine: {e}")
        logger.error(traceback.format_exc())

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'engine_ready': engine is not None,
        'engine_type': 'MABWiser Thompson Sampling'
    })

@app.route('/api/filter-options', methods=['GET'])
def get_filter_options():
    """Get available filter options from watch metadata."""
    try:
        if engine is None:
            return jsonify({'error': 'MABWiser engine not initialized'}), 500
        
        # Extract unique values from watch metadata
        brands = set()
        case_materials = set()
        movements = set()
        dial_colors = set()
        watch_types = set()
        complications = set()
        price_range = [float('inf'), 0]
        diameter_range = [float('inf'), 0]
        thickness_range = [float('inf'), 0]
        water_resistance_values = set()
        
        for watch_id, watch_data in engine.watch_data.items():
            # Extract brand
            if 'brand' in watch_data and watch_data['brand']:
                brands.add(watch_data['brand'])
            
            # Extract specs
            specs = watch_data.get('specs', {})
            
            # Case material
            if 'case_material' in specs and specs['case_material']:
                case_materials.add(specs['case_material'])
            
            # Movement
            if 'movement' in specs and specs['movement']:
                movements.add(specs['movement'])
            elif 'winding' in specs and specs['winding']:
                movements.add(specs['winding'])
            
            # Dial color
            if 'dial_color' in specs and specs['dial_color']:
                dial_colors.add(specs['dial_color'])
            
            # Watch type
            if 'watch_type' in specs and specs['watch_type']:
                watch_types.add(specs['watch_type'])
            if 'second_watch_type' in specs and specs['second_watch_type']:
                watch_types.add(specs['second_watch_type'])
            
            # Extract complications from various fields
            complication_fields = [
                'complication_chronograph', 'complication_date', 'complication_dual_time',
                'complication_gmt', 'complication_moonphase', 'complication_power_reserve'
            ]
            for field in complication_fields:
                if field in specs and specs[field] and specs[field] not in ['No', 'None', '']:
                    # Convert field name to readable format
                    complication_name = field.replace('complication_', '').replace('_', ' ').title()
                    complications.add(complication_name)
            
            # Price range
            if 'price' in watch_data and watch_data['price']:
                try:
                    price = float(watch_data['price'])
                    price_range[0] = min(price_range[0], price)
                    price_range[1] = max(price_range[1], price)
                except:
                    pass
        
        # Convert to lists and sort
        filter_options = {
            'brands': sorted(list(brands)),
            'case_materials': sorted(list(case_materials)),
            'movements': sorted(list(movements)),
            'dial_colors': sorted(list(dial_colors)),
            'watch_types': sorted(list(watch_types)),
            'complications': sorted(list(complications)),
            'price_range': [int(price_range[0]) if price_range[0] != float('inf') else 0, int(price_range[1])],
            'diameter_range': [int(diameter_range[0]) if diameter_range[0] != float('inf') else 0, int(diameter_range[1])],
            'thickness_range': [int(thickness_range[0]) if thickness_range[0] != float('inf') else 0, int(thickness_range[1])],
            'water_resistance_values': sorted(list(water_resistance_values))
        }
        
        logger.info(f"📊 Filter options: {len(filter_options['brands'])} brands, {len(filter_options['watch_types'])} types")
        return jsonify(filter_options)
        
    except Exception as e:
        logger.error(f"❌ Error getting filter options: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/session', methods=['POST'])
def create_session():
    """Create a new user session."""
    try:
        if engine is None:
            return jsonify({'error': 'MABWiser engine not initialized'}), 500
        
        # Generate a new session ID
        session_id = str(uuid.uuid4())
        
        # Create session in the engine
        engine.create_session(session_id)
        
        logger.info(f"✅ Created new MABWiser session: {session_id}")
        
        return jsonify({
            'session_id': session_id,
            'status': 'created',
            'engine_type': 'MABWiser Thompson Sampling'
        })
        
    except Exception as e:
        logger.error(f"❌ Error creating session: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get watch recommendations using MABWiser Thompson Sampling."""
    try:
        if engine is None:
            return jsonify({'error': 'MABWiser engine not initialized'}), 500
        
        data = request.get_json()
        session_id = data.get('session_id')
        filter_preferences = data.get('filter_preferences', {})
        exclude_ids = set(data.get('exclude_ids', []))
        
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        
        logger.info(f"🎯 Getting MABWiser recommendations for session {session_id}")
        
        # Get recommendations from MABWiser engine
        recommendations = engine.get_recommendations(
            session_id=session_id,
            exclude_ids=exclude_ids
        )
        
        logger.info(f"📝 MABWiser engine returned {len(recommendations)} recommendations")
        
        # Apply user filters if provided
        if filter_preferences:
            recommendations = apply_user_filters(recommendations, filter_preferences)
            logger.info(f"🔍 After filters: {len(recommendations)} recommendations")
        
        # Convert numpy arrays to Python types for JSON serialization
        json_safe_recommendations = convert_numpy_to_python(recommendations)
        
        # Log algorithm distribution
        algorithm_counts = {}
        for rec in recommendations:
            algorithm = rec.get('algorithm', 'Unknown')
            algorithm_counts[algorithm] = algorithm_counts.get(algorithm, 0) + 1
        
        logger.info(f"📊 Algorithm distribution: {algorithm_counts}")
        
        return jsonify({
            'recommendations': json_safe_recommendations,
            'session_id': session_id,
            'count': len(json_safe_recommendations),
            'filter_preferences': filter_preferences,
            'algorithm_distribution': algorithm_counts,
            'engine_type': 'MABWiser Thompson Sampling'
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting recommendations: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def apply_user_filters(recommendations, filter_preferences):
    """Apply user-defined filters to recommendations."""
    if not filter_preferences:
        logger.info("🔍 No filter preferences provided, returning all recommendations")
        return recommendations
    
    logger.info(f"🔍 Applying filters: {filter_preferences}")
    filtered = []
    rejected_count = 0
    
    for watch in recommendations:
        if should_include_watch(watch, filter_preferences):
            filtered.append(watch)
        else:
            rejected_count += 1
            # Log a few examples of rejected watches for debugging
            if rejected_count <= 3:
                specs = watch.get('specs', {})
                diameter = specs.get('diameter_mm', 'Unknown')
                dial_color = specs.get('dial_color', 'Unknown')
                brand = watch.get('brand', 'Unknown')
                logger.info(f"🚫 Rejected: {brand} - diameter: {diameter}mm, color: {dial_color}")
    
    logger.info(f"🔍 Filter results: {len(filtered)} accepted, {rejected_count} rejected")
    return filtered

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for a watch."""
    try:
        if engine is None:
            return jsonify({'error': 'MABWiser engine not initialized'}), 500
        
        data = request.get_json()
        session_id = data.get('session_id')
        watch_id = int(data.get('watch_id'))
        feedback = data.get('feedback')  # 'like' or 'dislike'
        
        if not all([session_id, watch_id is not None, feedback]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Convert feedback to reward
        reward = 1.0 if feedback == 'like' else 0.0
        
        # Update the MABWiser engine
        engine.update(
            session_id=session_id,
            watch_id=watch_id,
            reward=reward
        )
        
        # Get expert stats after update
        expert_stats = engine.get_expert_stats()
        session_experts = len(engine.session_experts.get(session_id, []))
        
        logger.info(f"✅ Processed {feedback} feedback for watch {watch_id} in session {session_id}")
        logger.info(f"📊 Session now has {session_experts} experts")
        
        return jsonify({
            'status': 'success',
            'message': f'Feedback processed for watch {watch_id}',
            'session_experts': session_experts,
            'total_experts': expert_stats['total_experts'],
            'engine_type': 'MABWiser Thompson Sampling'
        })
        
    except Exception as e:
        logger.error(f"❌ Error processing feedback: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/watch/<int:watch_id>', methods=['GET'])
def get_watch_details(watch_id):
    """Get detailed information about a specific watch."""
    try:
        if engine is None:
            return jsonify({'error': 'MABWiser engine not initialized'}), 500
        
        watch_data = engine.watch_data.get(watch_id)
        
        if watch_data is None:
            return jsonify({'error': 'Watch not found'}), 404
        
        # Format the watch details
        watch_details = engine._format_recommendation(watch_id, 1.0, "Direct")
        
        return jsonify(watch_details)
        
    except Exception as e:
        logger.error(f"❌ Error getting watch details: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/stats', methods=['GET'])
def get_debug_stats():
    """Get debug statistics about the MABWiser engine."""
    try:
        if engine is None:
            return jsonify({'error': 'MABWiser engine not initialized'}), 500
        
        expert_stats = engine.get_expert_stats()
        
        stats = {
            'total_watches': len(engine.watch_data),
            'available_watches': len(engine.available_watches),
            'engine_config': {
                'dim': engine.dim,
                'batch_size': engine.batch_size,
                'max_experts': engine.max_experts,
                'similarity_threshold': engine.similarity_threshold,
                'engine_type': 'MABWiser Thompson Sampling'
            },
            'expert_stats': expert_stats
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"❌ Error getting debug stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a specific session."""
    try:
        if engine is None:
            return jsonify({'error': 'MABWiser engine not initialized'}), 500
        
        # Remove session data
        removed = False
        if session_id in engine.session_liked_watches:
            del engine.session_liked_watches[session_id]
            removed = True
        if session_id in engine.session_experts:
            del engine.session_experts[session_id]
            removed = True
        if session_id in engine.session_shown_watches:
            del engine.session_shown_watches[session_id]
            removed = True
        
        logger.info(f"🗑️ Deleted MABWiser session {session_id} (existed: {removed})")
        
        return jsonify({
            'status': 'success',
            'message': f'Session {session_id} deleted',
            'existed': removed,
            'engine_type': 'MABWiser Thompson Sampling'
        })
        
    except Exception as e:
        logger.error(f"❌ Error deleting session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/<session_id>/status', methods=['GET'])
def get_session_status(session_id):
    """Get session status including expert information."""
    try:
        if engine is None:
            return jsonify({'error': 'MABWiser engine not initialized'}), 500
        
        # Check if session exists and get its status
        session_exists = (
            session_id in engine.session_liked_watches or 
            session_id in engine.session_experts
        )
        
        if not session_exists:
            return jsonify({
                'session_id': session_id,
                'exists': False,
                'experts_count': 0,
                'likes_count': 0,
                'engine_type': 'MABWiser Thompson Sampling'
            })
        
        # Get session details
        experts_count = len(engine.session_experts.get(session_id, []))
        likes_count = len(engine.session_liked_watches.get(session_id, []))
        shown_count = len(engine.session_shown_watches.get(session_id, set()))
        
        return jsonify({
            'session_id': session_id,
            'exists': True,
            'experts_count': experts_count,
            'likes_count': likes_count,
            'shown_count': shown_count,
            'timestamp': datetime.now().isoformat(),
            'engine_type': 'MABWiser Thompson Sampling'
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting session status: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting MABWiser Watch Finder API...")
    
    # Initialize engine
    initialize()
    
    # Start the server
    port = int(os.environ.get('PORT', 5001))  # Use port 5001 for frontend compatibility
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🌐 Starting MABWiser server on port {port}, debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)

# Initialize on import for production
initialize() 