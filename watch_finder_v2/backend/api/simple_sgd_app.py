#!/usr/bin/env python3
"""
Simple SGD-Based Recommendation API Server

Flask API server for the Simple SGD Engine using scikit-learn's SGDClassifier.

Endpoints:
- POST /api/session - Create new session
- DELETE /api/session/<session_id> - Delete session 
- POST /api/recommendations - Get recommendations  
- POST /api/feedback - Submit user feedback
- GET /api/health - Health check
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import logging
import traceback
from models.simple_sgd_engine import SimpleSgdEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Simple SGD Engine
logger.info("🚀 Initializing Simple SGD Engine...")
try:
    engine = SimpleSgdEngine(
        batch_size=1,           # 1 recommendation per request for more responsive learning
        data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
        like_weight=5.0,
        alpha=0.0001,
        prior_like_rate=0.2
    )
    logger.info("✅ Simple SGD Engine initialized successfully with batch_size=1 for responsive learning!")
except Exception as e:
    logger.error(f"❌ Failed to initialize Simple SGD Engine: {e}")
    logger.error(f"❌ Error details: {traceback.format_exc()}")
    engine = None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if engine is None:
        return jsonify({
            'status': 'error',
            'message': 'Simple SGD Engine not initialized'
        }), 500
    
    stats = engine.get_stats()
    return jsonify({
        'status': 'healthy',
        'engine': 'Simple SGD (scikit-learn)',
        'stats': stats
    })

@app.route('/api/session', methods=['POST'])
def create_session():
    """Create a new recommendation session."""
    try:
        if engine is None:
            return jsonify({'error': 'Engine not initialized'}), 500
        
        session_id = str(uuid.uuid4())
        engine.create_session(session_id)
        
        logger.info(f"✅ Created new Simple SGD session: {session_id}")
        
        return jsonify({
            'session_id': session_id,
            'message': 'Session created successfully'
        })
        
    except Exception as e:
        logger.error(f"❌ Error creating session: {e}")
        logger.error(f"❌ Error details: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to create session'}), 500

@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session (frontend compatibility)."""
    try:
        if engine is None:
            return jsonify({'error': 'Engine not initialized'}), 500
        
        # Clean up session data if it exists
        if session_id in engine.session_models:
            del engine.session_models[session_id]
        if session_id in engine.session_shown_watches:
            del engine.session_shown_watches[session_id]
        if session_id in engine.session_interaction_counts:
            del engine.session_interaction_counts[session_id]
        if session_id in engine.session_timestamps:
            del engine.session_timestamps[session_id]
        
        logger.info(f"🗑️ Deleted Simple SGD session: {session_id}")
        
        return jsonify({
            'message': 'Session deleted successfully',
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"❌ Error deleting session: {e}")
        return jsonify({'error': 'Failed to delete session'}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get watch recommendations for a session."""
    try:
        if engine is None:
            return jsonify({'error': 'Engine not initialized'}), 500
        
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        
        logger.info(f"🎯 Getting Simple SGD recommendations for session {session_id}")
        
        # Get recommendations from Simple SGD Engine
        recommendations = engine.get_recommendations(session_id)
        
        logger.info(f"📝 Simple SGD engine returned {len(recommendations)} recommendations")
        
        # Apply any filters from the request (optional - frontend compatibility)
        filters = data.get('filters', {})
        if filters:
            logger.info(f"🔍 Applying filters: {filters}")
            filtered_recommendations = apply_filters(recommendations, filters)
            logger.info(f"🔍 After filters: {len(filtered_recommendations)} recommendations")
        else:
            filtered_recommendations = recommendations
        
        # Log algorithm distribution
        algorithm_counts = {}
        for rec in filtered_recommendations:
            alg = rec.get('algorithm', 'unknown')
            algorithm_counts[alg] = algorithm_counts.get(alg, 0) + 1
        logger.info(f"📊 Algorithm distribution: {algorithm_counts}")
        
        return jsonify({
            'recommendations': filtered_recommendations,
            'session_id': session_id,
            'total': len(filtered_recommendations)
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting recommendations: {e}")
        logger.error(f"❌ Error details: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

@app.route('/api/feedback', methods=['POST', 'OPTIONS'])
def submit_feedback():
    """Submit user feedback (like/dislike) for a watch."""
    
    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        logger.info("🔍 Received OPTIONS request (CORS preflight)")
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        if engine is None:
            return jsonify({'error': 'Engine not initialized'}), 500
        
        # Debug: Log raw request data
        logger.info(f"🔍 Raw request data: {request.get_data()}")
        logger.info(f"🔍 Request headers: {dict(request.headers)}")
        logger.info(f"🔍 Request content type: {request.content_type}")
        logger.info(f"🔍 Request method: {request.method}")
        
        # Try to parse JSON data
        try:
            data = request.get_json()
            logger.info(f"🔍 Parsed JSON data: {data}")
        except Exception as json_error:
            logger.error(f"❌ JSON parsing error: {json_error}")
            return jsonify({'error': f'Invalid JSON: {str(json_error)}'}), 400
        
        # Check if data is None or empty
        if not data:
            logger.error("❌ No JSON data received")
            return jsonify({'error': 'No JSON data received'}), 400
        
        # Log all keys in the data
        logger.info(f"🔍 Data keys: {list(data.keys())}")
        
        # Extract fields with better error handling
        session_id = data.get('session_id')
        watch_id = data.get('watch_id')
        feedback = data.get('feedback')
        
        logger.info(f"🔍 Parsed fields - session_id: '{session_id}', watch_id: {watch_id}, feedback: '{feedback}'")
        
        # Validate required fields
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        if not watch_id:
            return jsonify({'error': 'watch_id is required'}), 400
        if not feedback:
            return jsonify({'error': 'feedback is required'}), 400
        
        # Validate feedback value
        if feedback not in ['like', 'dislike']:
            return jsonify({'error': 'feedback must be "like" or "dislike"'}), 400
        
        # Convert watch_id to int if it's a string number
        try:
            if isinstance(watch_id, str) and watch_id.isdigit():
                watch_id = int(watch_id)
        except ValueError:
            pass  # Keep as string if it's not a number
        
        # Convert feedback to reward
        reward = 1.0 if feedback == 'like' else 0.0
        
        # Update the engine with the feedback
        engine.update(session_id, watch_id, reward)
        
        logger.info(f"✅ Processed {feedback} feedback for watch {watch_id} in session {session_id}")
        
        # Log total sessions for debugging
        logger.info(f"📊 Total sessions: {len(engine.session_models)}")
        
        return jsonify({
            'status': 'success',
            'message': f'Feedback recorded: {feedback} for watch {watch_id}',
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"❌ Error processing feedback: {e}")
        logger.error(f"❌ Error details: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to process feedback'}), 500

@app.route('/api/variants/<watch_id>', methods=['GET'])
def get_variants(watch_id):
    """Get all variants of a specific watch (same brand and model)."""
    try:
        if engine is None:
            return jsonify({'error': 'Engine not initialized'}), 500
        
        # Convert watch_id to int if it's a string number
        try:
            if isinstance(watch_id, str) and watch_id.isdigit():
                watch_id = int(watch_id)
        except ValueError:
            pass  # Keep as string if it's not a number
        
        # Get the watch data to verify it exists
        watch_data = engine.watch_data.get(watch_id)
        if not watch_data:
            return jsonify({'error': 'Watch not found'}), 404
        
        # Get all variants (similar watches with same brand+model)
        similar_watch_ids = engine._get_similar_watches(watch_id)
        
        # Convert to list and get watch data for each variant
        variants = []
        for variant_id in similar_watch_ids:
            variant_data = engine.watch_data.get(variant_id, {})
            if variant_data:
                # Format the variant data
                formatted_variant = engine._format_recommendation(
                    variant_id, 
                    0.0,  # No confidence score for variants
                    "variant"
                )
                variants.append(formatted_variant)
        
        logger.info(f"🔍 Found {len(variants)} variants for watch {watch_id}")
        
        return jsonify({
            'status': 'success',
            'watch_id': watch_id,
            'variants': variants,
            'count': len(variants)
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting variants: {e}")
        logger.error(f"❌ Error details: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to get variants'}), 500

def apply_filters(recommendations, filters):
    """Apply frontend filters to recommendations (for compatibility)."""
    filtered = []
    rejected_count = 0
    
    for rec in recommendations:
        # Price range filter
        price_range = filters.get('priceRange', [0, 50000])
        price = rec.get('price', 0)
        if not (price_range[0] <= price <= price_range[1]):
            rejected_count += 1
            continue
        
        # Diameter filter
        min_diameter = filters.get('minDiameter', 0)
        max_diameter = filters.get('maxDiameter', 100)
        diameter = rec.get('diameter', 0)
        if diameter and not (min_diameter <= diameter <= max_diameter):
            logger.info(f"🚫 Rejected: {rec.get('brand', 'Unknown')} - diameter: {diameter}mm")
            rejected_count += 1
            continue
        
        # Brand filter
        brands = filters.get('brands', [])
        if brands and rec.get('brand') not in brands:
            rejected_count += 1
            continue
        
        # Add more filters as needed...
        
        filtered.append(rec)
    
    if rejected_count > 0:
        logger.info(f"🔍 Filter results: {len(filtered)} accepted, {rejected_count} rejected")
    
    return filtered

if __name__ == '__main__':
    if engine is None:
        logger.error("❌ Cannot start server: Simple SGD Engine failed to initialize")
        sys.exit(1)
    
    logger.info("🌟 Starting Simple SGD API Server...")
    logger.info("🔗 Frontend should connect to: http://localhost:5001")
    logger.info("📊 Health check: http://localhost:5001/api/health")
    
    # Run the server on port 5001 to match frontend configuration
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=False,  # Set to True for development
        threaded=True
    ) # Force Railway redeploy - Thu Jul 10 17:59:54 CEST 2025
