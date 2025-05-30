from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import sys
import os

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.beam_search import WatchBeamSearch

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global variables to store the search engine
beam_search_engine = None

def load_embeddings_directly(embeddings_path):
    """Load precomputed embeddings directly from pickle file."""
    print(f"Loading embeddings from: {embeddings_path}")
    
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        # New format with separate embeddings and data
        embeddings = data['embeddings']
        watch_data = data['watch_data']
    else:
        # Old format - assume it's just embeddings and look for companion data file
        embeddings = data
        # Try to find companion data file
        data_file = embeddings_path.replace('_embeddings', '_data').replace('.pkl', '.pkl')
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                watch_data = pickle.load(f)
        else:
            raise ValueError(f"No watch data found for embeddings file: {embeddings_path}")
    
    print(f"Loaded {len(watch_data)} watches with {embeddings.shape[1]}D embeddings")
    return embeddings, watch_data

def initialize_search_engine():
    """Initialize the beam search engine with precomputed embeddings."""
    global beam_search_engine
    
    try:
        # Check for embeddings files in order of preference
        embeddings_complete_path = os.path.join(PROJECT_ROOT, "embeddings/watch_image_embeddings_v2_complete.pkl")
        embeddings_v2_path = os.path.join(PROJECT_ROOT, "embeddings/watch_image_embeddings_v2.pkl")
        embeddings_v1_path = os.path.join(PROJECT_ROOT, "embeddings/watch_image_embeddings.pkl")
        
        embeddings_path = None
        if os.path.exists(embeddings_complete_path):
            embeddings_path = embeddings_complete_path
            print(f"Using complete v2 embeddings: {embeddings_complete_path}")
        elif os.path.exists(embeddings_v2_path):
            embeddings_path = embeddings_v2_path
            print(f"Using v2 embeddings: {embeddings_v2_path}")
        elif os.path.exists(embeddings_v1_path):
            embeddings_path = embeddings_v1_path
            print(f"Using v1 embeddings: {embeddings_v1_path}")
        else:
            print("❌ No embeddings file found!")
            print("Available files in embeddings directory:")
            embeddings_dir = os.path.join(PROJECT_ROOT, "embeddings")
            if os.path.exists(embeddings_dir):
                for file in os.listdir(embeddings_dir):
                    print(f"  - {file}")
            else:
                print("  - embeddings directory doesn't exist")
            print("Please run generate_clip_embeddings.py to create embeddings")
            return False
        
        # Load precomputed embeddings directly
        embeddings, watch_data = load_embeddings_directly(embeddings_path)
        
        # Initialize beam search engine
        print("Initializing beam search engine...")
        beam_search_engine = WatchBeamSearch(
            embeddings=embeddings,
            watch_data=watch_data,
            beam_width=10
        )
        
        print(f"✅ Search engine initialized successfully!")
        print(f"📊 Dataset: {len(watch_data)} watches")
        print(f"📐 Embedding dimension: {embeddings.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing search engine: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'engine_initialized': beam_search_engine is not None
    })

@app.route('/api/start-session', methods=['POST'])
def start_session():
    """Start a new recommendation session with smart seed watches."""
    try:
        print("Starting new session...")
        
        # Reset exploration for new session
        beam_search_engine.reset_exploration()
        print("Reset exploration for new session")
        
        num_seeds = request.json.get('num_seeds', 3)
        print(f"Getting {num_seeds} smart seeds...")
        
        # Use smart seeds instead of random
        seeds = beam_search_engine.get_smart_seeds(num_seeds)
        print(f"Got smart seeds: {seeds}")
        
        response_data = {
            'status': 'success',
            'seeds': seeds,
            'session_id': 'session_' + str(hash(str(seeds)))[:8],
            'session_type': 'smart_seeds'
        }
        print(f"Sending response: {response_data}")
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error in start_session: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/get-recommendations', methods=['POST'])
def get_recommendations():
    """Get watch recommendations based on user feedback with enhanced processing."""
    try:
        data = request.json
        liked_indices = data.get('liked_indices', [])
        disliked_indices = data.get('disliked_indices', [])
        current_candidates = data.get('current_candidates', [])
        step = data.get('step', 0)
        
        print(f"Processing recommendations - Step {step}, Likes: {liked_indices}, Dislikes: {disliked_indices}")
        
        # Add feedback to the system for weighted preference calculation
        try:
            for idx in liked_indices:
                beam_search_engine.add_feedback(idx, 'like', confidence=1.0)
            
            for idx in disliked_indices:
                beam_search_engine.add_feedback(idx, 'dislike', confidence=1.0)
        except Exception as feedback_error:
            print(f"Error adding feedback: {feedback_error}")
            # Continue anyway, feedback is not critical
        
        # Check if we need to reset exploration to avoid getting stuck
        try:
            exploration_stats = beam_search_engine.get_exploration_stats()
            if exploration_stats['exploration_percentage'] > 60:
                beam_search_engine.reset_exploration()
                print(f"Auto-reset exploration at {exploration_stats['exploration_percentage']:.1f}% coverage")
        except Exception as exploration_error:
            print(f"Error checking exploration stats: {exploration_error}")
            exploration_stats = {'exploration_percentage': 0, 'seen_watches': 0, 'total_watches': len(beam_search_engine.watch_data)}
        
        # Calculate enhanced user preference vector with time weighting
        try:
            user_preferences = beam_search_engine.calculate_weighted_user_preference_vector(
                liked_indices, disliked_indices
            )
        except Exception as preference_error:
            print(f"Error calculating preferences: {preference_error}")
            # Fallback to empty preferences
            user_preferences = np.zeros(beam_search_engine.dimension)
        
        # Get recommendations using enhanced beam search
        try:
            recommendations = beam_search_engine.beam_search_step(
                current_candidates, user_preferences, step
            )
        except Exception as beam_error:
            print(f"Error in beam search step: {beam_error}")
            import traceback
            print(f"Beam search traceback: {traceback.format_exc()}")
            
            # Fallback: return some smart seeds if beam search fails
            try:
                recommendations = beam_search_engine.get_smart_seeds(3)
                print("Fallback: using smart seeds as recommendations")
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                return jsonify({
                    'status': 'error',
                    'message': f'Beam search failed: {beam_error}, Fallback failed: {fallback_error}'
                }), 500
        
        # Get comprehensive metrics
        try:
            metrics = beam_search_engine.get_comprehensive_metrics()
        except Exception as metrics_error:
            print(f"Error getting metrics: {metrics_error}")
            metrics = {'error': 'Failed to calculate metrics'}
        
        return jsonify({
            'status': 'success',
            'recommendations': recommendations,
            'step': step + 1,
            'exploration_stats': exploration_stats,
            'metrics': metrics,
            'feedback_processed': {
                'likes': len(liked_indices),
                'dislikes': len(disliked_indices),
                'total_feedback_history': len(beam_search_engine.feedback_history)
            }
        })
    
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/add-feedback', methods=['POST'])
def add_feedback():
    """Add explicit user feedback with confidence scoring."""
    try:
        data = request.json
        watch_index = data.get('watch_index')
        feedback_type = data.get('feedback_type')  # 'like' or 'dislike'
        confidence = data.get('confidence', 1.0)  # 0.5 to 1.0
        
        if watch_index is None or feedback_type is None:
            return jsonify({
                'status': 'error',
                'message': 'watch_index and feedback_type are required'
            }), 400
        
        if feedback_type not in ['like', 'dislike']:
            return jsonify({
                'status': 'error',
                'message': 'feedback_type must be "like" or "dislike"'
            }), 400
        
        # Add feedback to the system
        beam_search_engine.add_feedback(watch_index, feedback_type, confidence)
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback added successfully',
            'total_feedback': len(beam_search_engine.feedback_history)
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/get-variants', methods=['GET'])
def get_variants():
    """Get variant information for a specific watch."""
    try:
        watch_index = request.args.get('index', type=int)
        
        if watch_index is None:
            return jsonify({
                'status': 'error',
                'message': 'index parameter is required'
            }), 400
        
        if watch_index < 0 or watch_index >= len(beam_search_engine.watch_data):
            return jsonify({
                'status': 'error',
                'message': 'Invalid watch index'
            }), 400
        
        # Get variant information
        variant_group = beam_search_engine.variant_detector.get_variant_group(watch_index)
        representative = beam_search_engine.variant_detector.get_representative(watch_index)
        
        # Convert to watch objects
        variants = []
        for idx in variant_group:
            watch = beam_search_engine.get_watch_by_index(idx)
            watch['is_representative'] = (idx == representative)
            variants.append(watch)
        
        return jsonify({
            'status': 'success',
            'watch_index': watch_index,
            'variant_count': len(variant_group),
            'representative_index': representative,
            'variants': variants
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/variant-stats', methods=['GET'])
def get_variant_stats():
    """Get variant detection statistics."""
    try:
        if not beam_search_engine:
            return jsonify({
                'status': 'error',
                'message': 'Search engine not initialized'
            }), 500
        
        variant_stats = beam_search_engine.variant_detector.get_variant_stats()
        
        return jsonify({
            'status': 'success',
            'variant_stats': variant_stats
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get comprehensive recommendation system metrics."""
    try:
        if not beam_search_engine:
            return jsonify({
                'status': 'error',
                'message': 'Search engine not initialized'
            }), 500
        
        metrics = beam_search_engine.get_comprehensive_metrics()
        variant_stats = beam_search_engine.variant_detector.get_variant_stats()
        
        return jsonify({
            'status': 'success',
            'metrics': metrics,
            'system_info': {
                'total_watches': len(beam_search_engine.watch_data),
                'embedding_dimension': beam_search_engine.dimension,
                'style_clusters': len(beam_search_engine.cluster_to_watches),
                'beam_width': beam_search_engine.beam_width,
                'variant_detection': variant_stats
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/reset-session', methods=['POST'])
def reset_session():
    """Reset the current recommendation session."""
    try:
        beam_search_engine.reset_exploration()
        
        return jsonify({
            'status': 'success',
            'message': 'Session reset successfully'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/get-similar', methods=['POST'])
def get_similar():
    """Get watches similar to a specific watch."""
    try:
        data = request.json
        watch_index = data.get('watch_index')
        num_similar = data.get('num_similar', 5)
        
        if watch_index is None:
            return jsonify({
                'status': 'error',
                'message': 'watch_index is required'
            }), 400
        
        # Find similar watches
        similar = beam_search_engine.find_similar_watches([watch_index], num_similar)
        
        # Convert to watch objects
        similar_watches = []
        for idx, score in similar:
            watch = beam_search_engine.get_watch_by_index(idx)
            watch['similarity_score'] = score
            similar_watches.append(watch)
        
        return jsonify({
            'status': 'success',
            'similar_watches': similar_watches
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/get-watch', methods=['GET'])
def get_watch():
    """Get a specific watch by index."""
    try:
        watch_index = request.args.get('index', type=int)
        
        if watch_index is None:
            return jsonify({
                'status': 'error',
                'message': 'index parameter is required'
            }), 400
        
        if watch_index < 0 or watch_index >= len(beam_search_engine.watch_data):
            return jsonify({
                'status': 'error',
                'message': 'Invalid watch index'
            }), 400
        
        watch = beam_search_engine.get_watch_by_index(watch_index)
        
        return jsonify({
            'status': 'success',
            'watch': watch
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about the watch database."""
    try:
        if not beam_search_engine:
            return jsonify({
                'status': 'error',
                'message': 'Search engine not initialized'
            }), 500
        
        # Count watches by brand
        brand_counts = {}
        price_ranges = {'low': 0, 'mid': 0, 'high': 0, 'unknown': 0}
        
        for watch in beam_search_engine.watch_data:
            # Brand stats
            brand = watch.get('brand', 'unknown')
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
            
            # Price stats
            price = watch.get('price', '')
            if price:
                price_str = str(price).lower()
                if any(low in price_str for low in ['200', '300', '400']):
                    price_ranges['low'] += 1
                elif any(mid in price_str for mid in ['500', '600', '700', '800']):
                    price_ranges['mid'] += 1
                elif any(high in price_str for high in ['1000', '1500', '1600', '1700']):
                    price_ranges['high'] += 1
                else:
                    price_ranges['unknown'] += 1
            else:
                price_ranges['unknown'] += 1
        
        return jsonify({
            'status': 'success',
            'total_watches': len(beam_search_engine.watch_data),
            'brands': brand_counts,
            'price_ranges': price_ranges,
            'embedding_dimension': beam_search_engine.dimension
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    import os
    print("Initializing beam search watch finder...")
    
    if initialize_search_engine():
        print("Starting Flask server...")
        port = int(os.environ.get('PORT', 5001))
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        print("Failed to initialize search engine. Exiting.")
        sys.exit(1) 