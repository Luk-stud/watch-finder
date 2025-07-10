#!/usr/bin/env python3
"""
Simple Vector API Server

Flask API server using the Simple Vector Engine for watch recommendations.
Compatible with the existing frontend but much simpler than MABWiser.

Key endpoints:
- POST /api/session - Create new session
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
from models.simple_vector_engine import SimpleVectorEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Simple Vector Engine
logger.info("🚀 Initializing Simple Vector Engine...")
try:
    engine = SimpleVectorEngine(
        batch_size=3,           # 3 recommendations per request
        repel_strength=0.3,     # β=0.3 for negative feedback repulsion
        decay_factor=0.95,      # γ=0.95 for profile decay (5% decay per interaction)
        data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    )
    logger.info("✅ Simple Vector Engine initialized successfully!")
except Exception as e:
    logger.error(f"❌ Failed to initialize Simple Vector Engine: {e}")
    logger.error(f"❌ Error details: {traceback.format_exc()}")
    engine = None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if engine is None:
        return jsonify({
            'status': 'error',
            'message': 'Simple Vector Engine not initialized'
        }), 500
    
    stats = engine.get_stats()
    return jsonify({
        'status': 'healthy',
        'engine': 'Simple Vector (Cosine Similarity)',
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
        
        logger.info(f"✅ Created new Simple Vector session: {session_id}")
        
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
        if session_id in engine.session_user_vecs:
            del engine.session_user_vecs[session_id]
        if session_id in engine.session_user_sums:
            del engine.session_user_sums[session_id]
        if session_id in engine.session_shown_watches:
            del engine.session_shown_watches[session_id]
        if session_id in engine.session_interaction_counts:
            del engine.session_interaction_counts[session_id]
        if session_id in engine.session_timestamps:
            del engine.session_timestamps[session_id]
        
        logger.info(f"🗑️ Deleted Simple Vector session: {session_id}")
        
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
        
        logger.info(f"🎯 Getting Simple Vector recommendations for session {session_id}")
        
        # Get recommendations from Simple Vector Engine
        recommendations = engine.get_recommendations(session_id)
        
        logger.info(f"📝 Simple Vector engine returned {len(recommendations)} recommendations")
        
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

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback (like/dislike) for a watch."""
    try:
        if engine is None:
            return jsonify({'error': 'Engine not initialized'}), 500
        
        data = request.get_json()
        logger.info(f"🔍 Received feedback data: {data}")
        
        session_id = data.get('session_id')
        watch_id = data.get('watch_id')
        feedback_type = data.get('type')  # 'like' or 'dislike'
        
        logger.info(f"🔍 Parsed - session_id: {session_id}, watch_id: {watch_id}, type: {feedback_type}")
        
        if not session_id or watch_id is None or not feedback_type:
            logger.error(f"❌ Missing required fields: session_id={session_id}, watch_id={watch_id}, type={feedback_type}")
            return jsonify({'error': 'session_id, watch_id, and type are required'}), 400
        
        # Convert feedback to reward signal
        reward = 1.0 if feedback_type == 'like' else 0.0
        
        # Update engine
        engine.update(session_id, int(watch_id), reward)
        
        # Get session stats
        stats = engine.get_stats()
        
        feedback_msg = "like" if reward > 0 else "dislike"
        logger.info(f"✅ Processed {feedback_msg} feedback for watch {watch_id} in session {session_id}")
        logger.info(f"📊 Total sessions: {stats['total_sessions']}")
        
        return jsonify({
            'message': f'Feedback processed successfully',
            'feedback_type': feedback_type,
            'watch_id': watch_id,
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"❌ Error processing feedback: {e}")
        logger.error(f"❌ Error details: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to process feedback'}), 500

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
        logger.error("❌ Cannot start server: Simple Vector Engine failed to initialize")
        sys.exit(1)
    
    logger.info("🌟 Starting Simple Vector API Server...")
    logger.info("🔗 Frontend should connect to: http://localhost:5001")
    logger.info("📊 Health check: http://localhost:5001/api/health")
    
    # Run the server
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=False,  # Set to True for development
        threaded=True
    ) 