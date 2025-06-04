"""
Watch Finder Backend API
========================

Flask API server for watch recommendation engine with LinUCB algorithm.
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

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

# Import the engine
from linucb_engine import DynamicMultiExpertLinUCBEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('watch_finder_api.log')
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
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173"
]

if ENVIRONMENT == 'production':
    # Production CORS - restrict to specific domains
    NETLIFY_DOMAIN = os.getenv('NETLIFY_DOMAIN', 'watchrecomender.netlify.app')
    CORS(app, origins=allowed_origins, methods=['GET', 'POST', 'OPTIONS'], 
         allow_headers=['Content-Type', 'Authorization'], supports_credentials=True)
    logger.info(f"🔒 Production CORS enabled for: {allowed_origins}")
else:
    # Development CORS - allow all origins plus specific ones
    CORS(app, origins=["*"] + allowed_origins, methods=['GET', 'POST', 'OPTIONS'], 
         allow_headers=['Content-Type', 'Authorization'], supports_credentials=True)
    logger.info("🔓 Development CORS enabled (all origins + specific domains)")

# Global engine instance
engine = None

def initialize_engine():
    """Initialize the recommendation engine."""
    global engine
    try:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        logger.info(f"Initializing engine with data directory: {data_dir}")
        
        engine = DynamicMultiExpertLinUCBEngine(
            dim=50,
            alpha=0.15,
            batch_size=5,
            max_experts=6,
            data_dir=data_dir
        )
        logger.info("✅ Engine initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize engine: {e}")
        logger.error(traceback.format_exc())
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'engine_ready': engine is not None
    })

@app.route('/api/filter-options', methods=['GET'])
def get_filter_options():
    """Get available filter options from watch metadata."""
    try:
        if engine is None:
            return jsonify({'error': 'Engine not initialized'}), 500
        
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
            
            # Diameter range
            if 'diameter_mm' in specs and specs['diameter_mm']:
                try:
                    diameter = float(specs['diameter_mm'])
                    diameter_range[0] = min(diameter_range[0], diameter)
                    diameter_range[1] = max(diameter_range[1], diameter)
                except:
                    pass
            
            # Thickness range
            thickness_fields = ['thickness_with_crystal_mm', 'thickness_without_crystal_mm']
            for field in thickness_fields:
                if field in specs and specs[field]:
                    try:
                        thickness = float(specs[field])
                        thickness_range[0] = min(thickness_range[0], thickness)
                        thickness_range[1] = max(thickness_range[1], thickness)
                    except:
                        pass
            
            # Water resistance
            if 'waterproofing_meters' in specs and specs['waterproofing_meters']:
                try:
                    wr = int(specs['waterproofing_meters'])
                    water_resistance_values.add(wr)
                except:
                    pass
        
        # Convert sets to sorted lists
        filter_options = {
            'brands': sorted(list(brands))[:50],  # Limit to top 50 brands
            'caseMaterials': sorted(list(case_materials)),
            'movements': sorted(list(movements)),
            'dialColors': sorted(list(dial_colors)),
            'watchTypes': sorted(list(watch_types)),
            'complications': sorted(list(complications)),
            'priceRange': [max(0, int(price_range[0])), int(price_range[1])] if price_range[0] != float('inf') else [0, 50000],
            'diameterRange': [max(20, int(diameter_range[0])), min(60, int(diameter_range[1]))] if diameter_range[0] != float('inf') else [30, 50],
            'thicknessRange': [max(3, int(thickness_range[0])), min(25, int(thickness_range[1]))] if thickness_range[0] != float('inf') else [5, 20],
            'waterResistanceOptions': sorted(list(water_resistance_values))[:20]  # Limit to 20 common values
        }
        
        logger.info(f"✅ Generated filter options: {len(filter_options['brands'])} brands, {len(filter_options['caseMaterials'])} materials")
        
        return jsonify(filter_options)
        
    except Exception as e:
        logger.error(f"❌ Error getting filter options: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/session', methods=['POST'])
def create_session():
    """Create a new session or get existing session info."""
    try:
        data = request.get_json() or {}
        provided_session_id = data.get('session_id')
        
        # Generate new session ID if none provided or if provided one is None/empty
        if not provided_session_id:
            session_id = str(uuid.uuid4())
        else:
            session_id = provided_session_id
        
        # Initialize session in engine if needed
        if engine is None:
            return jsonify({'error': 'Engine not initialized'}), 500
        
        logger.info(f"✅ Created session: {session_id}")
        
        return jsonify({
            'session_id': session_id,
            'status': 'active',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error creating session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get watch recommendations with filter preferences."""
    try:
        if engine is None:
            return jsonify({'error': 'Engine not initialized'}), 500
        
        data = request.get_json()
        session_id = data.get('session_id', str(uuid.uuid4()))
        exclude_ids = set(data.get('exclude_ids', []))
        
        # Get filter preferences from request
        filter_preferences = data.get('filter_preferences', {})
        
        # Extract similarity weights (default to equal weighting)
        clip_weight = filter_preferences.get('clipSimilarityWeight', 50) / 100.0
        text_weight = filter_preferences.get('textSimilarityWeight', 50) / 100.0
        
        # Create context vector that includes filter preferences
        context = np.zeros(engine.dim)
        
        # Encode filter preferences into context (first few dimensions)
        context[0] = clip_weight
        context[1] = text_weight
        
        # Get raw recommendations from engine
        recommendations = engine.get_recommendations(
            session_id=session_id,
            context=context,
            exclude_ids=exclude_ids
        )
        
        logger.info(f"🔍 Raw recommendations from engine: {len(recommendations)}")
        
        # Apply additional filtering based on user preferences
        filtered_recommendations = apply_user_filters(recommendations, filter_preferences)
        
        logger.info(f"🔍 After user filters: {len(filtered_recommendations)}")
        
        # If filtering resulted in too few recommendations, relax filters and try again
        if len(filtered_recommendations) < 3 and len(recommendations) > 3:
            logger.info(f"⚠️  Only {len(filtered_recommendations)} recommendations after filtering, relaxing filters...")
            
            # First fallback: remove some filter restrictions
            relaxed_preferences = filter_preferences.copy()
            
            # Remove diameter restrictions if they exist
            if 'minDiameter' in relaxed_preferences:
                del relaxed_preferences['minDiameter']
            if 'maxDiameter' in relaxed_preferences:
                del relaxed_preferences['maxDiameter']
            
            # Try with relaxed filters
            filtered_recommendations = apply_user_filters(recommendations, relaxed_preferences)
            logger.info(f"🔍 After relaxed filters: {len(filtered_recommendations)}")
            
            # If still too few, try with even more relaxed filters
            if len(filtered_recommendations) < 3:
                # Second fallback: only keep brand and price filters
                minimal_preferences = {}
                if 'brands' in filter_preferences:
                    minimal_preferences['brands'] = filter_preferences['brands']
                if 'priceRange' in filter_preferences:
                    minimal_preferences['priceRange'] = filter_preferences['priceRange']
                
                filtered_recommendations = apply_user_filters(recommendations, minimal_preferences)
                logger.info(f"🔍 After minimal filters: {len(filtered_recommendations)}")
                
                # Final fallback: no filters if still insufficient
                if len(filtered_recommendations) < 3:
                    logger.info("📢 Using unfiltered recommendations - user filters too restrictive")
                    filtered_recommendations = recommendations
        
        # Ensure we always have some recommendations
        if len(filtered_recommendations) == 0:
            logger.error("❌ No recommendations available even after fallbacks!")
            return jsonify({
                'status': 'error',
                'message': 'No recommendations available',
                'recommendations': [],
                'session_id': session_id,
                'count': 0
            }), 404
        
        # Convert all numpy types to Python types for JSON serialization
        json_safe_recommendations = convert_numpy_to_python(filtered_recommendations)
        
        logger.info(f"✅ Generated {len(json_safe_recommendations)} filtered recommendations for session {session_id}")
        
        return jsonify({
            'status': 'success',
            'recommendations': json_safe_recommendations,
            'session_id': session_id,
            'count': len(json_safe_recommendations),
            'filter_preferences': filter_preferences
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting recommendations: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def apply_user_filters(recommendations, filter_preferences):
    """Apply user-defined filters to recommendations."""
    if not filter_preferences:
        return recommendations
    
    filtered = []
    
    for watch in recommendations:
        if should_include_watch(watch, filter_preferences):
            # Adjust score based on similarity preferences
            adjusted_watch = adjust_watch_score(watch, filter_preferences)
            filtered.append(adjusted_watch)
    
    return filtered

def should_include_watch(watch, filters):
    """Check if a watch passes the user's filter criteria."""
    try:
        # Brand filter
        if filters.get('brands') and watch.get('brand'):
            if watch['brand'] not in filters['brands']:
                return False
        
        # Price filter
        price_range = filters.get('priceRange', [0, 1000000])
        if watch.get('price'):
            try:
                price = float(watch['price'])
                if price < price_range[0] or price > price_range[1]:
                    return False
            except:
                pass
        
        # Watch type filter
        if filters.get('watchTypes'):
            specs = watch.get('specs', {})
            watch_type = specs.get('watch_type', '')
            second_type = specs.get('second_watch_type', '')
            
            if not any(wt in [watch_type, second_type] for wt in filters['watchTypes']):
                return False
        
        # Case material filter
        if filters.get('caseMaterials'):
            specs = watch.get('specs', {})
            case_material = specs.get('case_material', '')
            if not any(mat in case_material for mat in filters['caseMaterials']):
                return False
        
        # Movement filter
        if filters.get('movements'):
            specs = watch.get('specs', {})
            movement = specs.get('movement', '') or specs.get('winding', '')
            if not any(mov in movement for mov in filters['movements']):
                return False
        
        # Dial color filter
        if filters.get('dialColors'):
            specs = watch.get('specs', {})
            dial_color = specs.get('dial_color', '')
            if dial_color not in filters['dialColors']:
                return False
        
        # Diameter filter
        min_diameter = filters.get('minDiameter')
        max_diameter = filters.get('maxDiameter')
        if min_diameter is not None or max_diameter is not None:
            specs = watch.get('specs', {})
            diameter_str = specs.get('diameter_mm', '')
            if diameter_str:
                try:
                    diameter = float(diameter_str)
                    if min_diameter and diameter < min_diameter:
                        return False
                    if max_diameter and diameter > max_diameter:
                        return False
                except:
                    pass
        
        # Water resistance filter
        min_wr = filters.get('waterResistance', 0)
        if min_wr > 0:
            specs = watch.get('specs', {})
            wr_str = specs.get('waterproofing_meters', '') or specs.get('waterproofing', '')
            if wr_str:
                try:
                    wr = int(wr_str)
                    if wr < min_wr:
                        return False
                except:
                    pass
        
        return True
        
    except Exception as e:
        logger.warning(f"Error applying filters to watch {watch.get('watch_id', 'unknown')}: {e}")
        return True  # Include watch if filtering fails

def convert_numpy_to_python(obj):
    """Recursively convert numpy arrays and types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj

def adjust_watch_score(watch, filter_preferences):
    """Adjust watch score based on similarity preferences."""
    try:
        # Get similarity weights
        clip_weight = filter_preferences.get('clipSimilarityWeight', 50) / 100.0
        text_weight = filter_preferences.get('textSimilarityWeight', 50) / 100.0
        
        # Get current score
        current_score = watch.get('score', 0.5)
        
        # For now, we'll modify the algorithm field to indicate the weighting
        algorithm_info = f"LinUCB (Visual:{clip_weight*100:.0f}% + Vibe:{text_weight*100:.0f}%)"
        
        # Create a copy of the watch with updated info
        adjusted_watch = watch.copy()
        adjusted_watch['algorithm'] = algorithm_info
        adjusted_watch['filter_preferences_applied'] = True
        
        # Ensure similarity weights are Python floats, not numpy types
        adjusted_watch['similarity_weights'] = {
            'clip': float(clip_weight), 
            'text': float(text_weight)
        }
        
        # Adjust confidence based on how well defined the preferences are
        total_weight = clip_weight + text_weight
        if total_weight > 0:
            confidence_boost = min(0.1, total_weight * 0.1)
            adjusted_watch['confidence'] = min(0.95, watch.get('confidence', 0.5) + confidence_boost)
        
        # Convert any numpy types to Python types
        adjusted_watch = convert_numpy_to_python(adjusted_watch)
        
        return adjusted_watch
        
    except Exception as e:
        logger.warning(f"Error adjusting watch score: {e}")
        return convert_numpy_to_python(watch)

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for a watch."""
    try:
        if engine is None:
            return jsonify({'error': 'Engine not initialized'}), 500
        
        data = request.get_json()
        session_id = data.get('session_id')
        watch_id = int(data.get('watch_id'))
        feedback = data.get('feedback')  # 'like' or 'dislike'
        
        if not all([session_id, watch_id is not None, feedback]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Convert feedback to reward
        reward = 1.0 if feedback == 'like' else 0.0
        
        # Create a dummy context for now
        context = np.zeros(engine.dim)
        
        # Update the engine
        engine.update(
            session_id=session_id,
            watch_id=watch_id,
            reward=reward,
            context=context
        )
        
        logger.info(f"✅ Processed {feedback} feedback for watch {watch_id} in session {session_id}")
        
        return jsonify({
            'status': 'success',
            'message': f'Feedback processed for watch {watch_id}'
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
            return jsonify({'error': 'Engine not initialized'}), 500
        
        watch_details = engine.get_watch_details(watch_id)
        
        if watch_details is None:
            return jsonify({'error': 'Watch not found'}), 404
        
        return jsonify(watch_details)
        
    except Exception as e:
        logger.error(f"❌ Error getting watch details: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/stats', methods=['GET'])
def get_debug_stats():
    """Get debug statistics about the engine."""
    try:
        if engine is None:
            return jsonify({'error': 'Engine not initialized'}), 500
        
        stats = {
            'total_watches': len(engine.watch_data),
            'total_experts': len(engine.experts),
            'available_watches': len(engine.available_watches),
            'engine_config': {
                'dim': engine.dim,
                'alpha': engine.alpha,
                'batch_size': engine.batch_size,
                'max_experts': engine.max_experts,
                'similarity_threshold': engine.similarity_threshold
            }
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"❌ Error getting debug stats: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Watch Finder API...")
    
    # Initialize engine
    if initialize_engine():
        logger.info("✅ Engine initialized successfully")
    else:
        logger.error("❌ Failed to initialize engine")
        sys.exit(1)
    
    # Start the server
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🌐 Starting server on port {port}, debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug) 