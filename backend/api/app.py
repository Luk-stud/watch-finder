from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import sys
import os
import uuid
import time
from typing import Dict, Any

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 🆕 Import the new Enhanced Beam Search v2
from models.beam_search_v2 import EnhancedWatchBeamSearch

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global configuration
print(f"Project root: {PROJECT_ROOT}")

# Session management
class SessionManager:
    def __init__(self, base_embeddings, base_watch_data):
        self.base_embeddings = base_embeddings
        self.base_watch_data = base_watch_data
        
        # 🚀 PRE-NORMALIZE EMBEDDINGS AT STARTUP (once for all sessions)
        print("🔧 Pre-normalizing embeddings for all sessions...")
        self.normalized_embeddings = base_embeddings / np.linalg.norm(base_embeddings, axis=1, keepdims=True)
        print(f"✅ Pre-normalized {len(base_embeddings)} embeddings at startup")
        
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = 3600  # 1 hour timeout
        
    def create_session(self) -> str:
        """Create a new user session with isolated enhanced beam search engine."""
        session_id = str(uuid.uuid4())
        
        # 🆕 Pass PRE-NORMALIZED embeddings to session engine
        session_engine = EnhancedWatchBeamSearch(
            embeddings=self.normalized_embeddings,  # Already normalized!
            watch_data=[watch.copy() for watch in self.base_watch_data],  # Fresh copy
            initial_beam_width=15,  # Enhanced default
            max_beam_width=30,
            embeddings_pre_normalized=True  # Signal that embeddings are already normalized
        )
        
        self.sessions[session_id] = {
            'engine': session_engine,
            'created_at': time.time(),
            'last_activity': time.time()
        }
        
        print(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> EnhancedWatchBeamSearch:
        """Get enhanced beam search engine for a specific session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        session['last_activity'] = time.time()
        return session['engine']
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions to free memory."""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session_data in self.sessions.items():
            if current_time - session_data['last_activity'] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            print(f"Cleaned up expired session: {session_id}")
    
    def get_session_count(self) -> int:
        """Get number of active sessions."""
        self.cleanup_expired_sessions()
        return len(self.sessions)

# Global session manager
session_manager = None

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

def initialize_session_manager():
    """Initialize the session manager with base data."""
    global session_manager
    
    try:
        # Check for embeddings files in deployment-ready locations
        embeddings_path = os.path.join(os.path.dirname(__file__), '../data/watch_embeddings.pkl')
        metadata_path = os.path.join(os.path.dirname(__file__), '../data/watch_metadata.pkl')
        
        print(f"Looking for embeddings at: {embeddings_path}")
        print(f"Looking for metadata at: {metadata_path}")
        
        if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            print("✅ Found deployment-ready embeddings and metadata files")
            # Load embeddings and metadata separately
            with open(embeddings_path, 'rb') as f:
                embeddings_data = pickle.load(f)
                
            # Handle different embeddings formats
            if isinstance(embeddings_data, dict):
                if 'embeddings' in embeddings_data:
                    embeddings = embeddings_data['embeddings']
                    print(f"Extracted embeddings from dictionary format")
                else:
                    # If it's a dict but no 'embeddings' key, assume it's the embeddings itself
                    embeddings = embeddings_data
                    print(f"Using dictionary as embeddings directly")
            else:
                embeddings = embeddings_data
                print(f"Using raw embeddings data")
                
            with open(metadata_path, 'rb') as f:
                watch_data = pickle.load(f)
                
            print(f"Loaded {len(watch_data)} watches with {embeddings.shape[1]}D embeddings")
        else:
            # Fallback: try original locations for local development (prioritize final scrape)
            embeddings_final_path = os.path.join(PROJECT_ROOT, "embeddings/watch_image_embeddings_final_scrape.pkl")
            embeddings_v3_path = os.path.join(PROJECT_ROOT, "embeddings/watch_image_embeddings_v3_detailed.pkl")
            embeddings_complete_path = os.path.join(PROJECT_ROOT, "embeddings/watch_image_embeddings_v2_complete.pkl")
            embeddings_v2_path = os.path.join(PROJECT_ROOT, "embeddings/watch_image_embeddings_v2.pkl")
            embeddings_v1_path = os.path.join(PROJECT_ROOT, "embeddings/watch_image_embeddings.pkl")
            
            embeddings_path = None
            if os.path.exists(embeddings_final_path):
                embeddings_path = embeddings_final_path
                print(f"🎉 Using NEW final scrape embeddings: {embeddings_final_path}")
            elif os.path.exists(embeddings_v3_path):
                embeddings_path = embeddings_v3_path
                print(f"🎉 Using v3 detailed embeddings: {embeddings_v3_path}")
            elif os.path.exists(embeddings_complete_path):
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
                print("Available files in backend/data directory:")
                data_dir = os.path.join(os.path.dirname(__file__), '../data')
                if os.path.exists(data_dir):
                    for file in os.listdir(data_dir):
                        print(f"  - {file}")
                else:
                    print("  - backend/data directory doesn't exist")
                print("Available files in embeddings directory:")
                embeddings_dir = os.path.join(PROJECT_ROOT, "embeddings")
                if os.path.exists(embeddings_dir):
                    for file in os.listdir(embeddings_dir):
                        print(f"  - {file}")
                else:
                    print("  - embeddings directory doesn't exist")
                return False
            
            # Load precomputed embeddings using original method
            embeddings, watch_data = load_embeddings_directly(embeddings_path)
        
        # Initialize session manager with base data
        session_manager = SessionManager(embeddings, watch_data)
        
        print(f"✅ Session manager initialized successfully!")
        print(f"📊 Base dataset: {len(watch_data)} watches")
        print(f"📐 Embedding dimension: {embeddings.shape[1]}")
        
        # Show data version info
        if watch_data and len(watch_data) > 0:
            sample_watch = watch_data[0]
            has_specs = 'specs' in sample_watch and sample_watch['specs']
            has_series = has_specs and sample_watch['specs'].get('serie', '') and sample_watch['specs']['serie'] != '-'
            print(f"📋 Enhanced specifications: {'✅' if has_specs else '❌'}")
            print(f"🔗 Series information: {'✅' if has_series else '❌'}")
            print(f"📄 Data source: {sample_watch.get('source', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing session manager: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    session_count = session_manager.get_session_count() if session_manager else 0
    return jsonify({
        'status': 'healthy',
        'session_manager_initialized': session_manager is not None,
        'active_sessions': session_count
    })

@app.route('/api/start-session', methods=['POST'])
def start_session():
    """Start a new recommendation session with smart seed watches."""
    try:
        print("Starting new session...")
        
        if not session_manager:
            return jsonify({
                'status': 'error',
                'message': 'Session manager not initialized'
            }), 500
        
        # Create new session
        session_id = session_manager.create_session()
        beam_search_engine = session_manager.get_session(session_id)
        
        num_seeds = request.json.get('num_seeds', 7) if request.json else 7
        print(f"Getting {num_seeds} smart seeds for session {session_id}...")
        
        # Use smart seeds
        seeds = beam_search_engine.get_smart_seeds(num_seeds)
        print(f"Got smart seeds for session {session_id}: {[s['index'] for s in seeds]}")
        
        response_data = {
            'status': 'success',
            'seeds': seeds,
            'session_id': session_id,
            'session_type': 'smart_seeds'
        }
        print(f"Sending response for session {session_id}")
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error in start_session: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/get-recommendations', methods=['POST'])
def get_recommendations():
    """Get watch recommendations using enhanced multi-modal beam search."""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                'status': 'error',
                'message': 'session_id is required'
            }), 400
        
        # Get session-specific enhanced beam search engine
        beam_search_engine = session_manager.get_session(session_id)
        
        liked_indices = data.get('liked_indices', [])
        disliked_indices = data.get('disliked_indices', [])
        current_candidates = data.get('current_candidates', [])
        step = data.get('step', 0)
        num_recommendations = data.get('num_recommendations', 10)
        
        print(f"Processing recommendations for session {session_id} - Step {step}, Likes: {liked_indices}, Dislikes: {disliked_indices}")
        
        # 🆕 LOG EXPLORATION STATS BEFORE RECOMMENDATIONS
        try:
            exploration_stats = beam_search_engine.get_exploration_stats()
            print(f"📊 Session {session_id}: {exploration_stats['seen_watches']}/{exploration_stats['total_watches']} watches seen ({exploration_stats['exploration_percentage']:.1f}%)")
            
            # Auto-suggest reset if too much explored
            if exploration_stats['exploration_percentage'] > 80:
                print(f"⚠️  Session {session_id}: High exploration percentage, may need reset soon")
        except Exception as stats_error:
            print(f"Error getting exploration stats for session {session_id}: {stats_error}")
            exploration_stats = {'exploration_percentage': 0, 'seen_watches': 0, 'total_watches': len(beam_search_engine.watch_data)}
        
        # 🆕 Add feedback to the enhanced multi-modal system
        try:
            for idx in liked_indices:
                beam_search_engine.add_feedback(idx, 'like', confidence=1.0)
            
            for idx in disliked_indices:
                beam_search_engine.add_feedback(idx, 'dislike', confidence=1.0)
        except Exception as feedback_error:
            print(f"Error adding feedback for session {session_id}: {feedback_error}")
        
        # 🆕 Get multi-modal recommendations using the enhanced beam search
        try:
            if step == 0 and not liked_indices and not disliked_indices:
                # First step: get smart seeds
                recommendations = beam_search_engine.get_smart_seeds(num_recommendations)
                print(f"First step: providing {len(recommendations)} smart seeds")
            else:
                # Use multi-modal beam search for subsequent steps
                recommendations = beam_search_engine.multi_modal_beam_search_step(
                    current_candidates, liked_indices, disliked_indices, step
                )
                print(f"Multi-modal search: provided {len(recommendations)} recommendations")
                
            # 🆕 VALIDATE NO DUPLICATES IN RESPONSE
            seen_indices_in_response = set()
            unique_recommendations = []
            for watch in recommendations:
                watch_idx = watch.get('index')
                if watch_idx is not None and watch_idx not in seen_indices_in_response:
                    seen_indices_in_response.add(watch_idx)
                    unique_recommendations.append(watch)
                else:
                    print(f"⚠️  Filtered duplicate in response: watch {watch_idx}")
            
            recommendations = unique_recommendations
            print(f"✅ Final response: {len(recommendations)} unique recommendations for session {session_id}")
            
        except Exception as beam_error:
            print(f"Error in enhanced beam search for session {session_id}: {beam_error}")
            try:
                # Fallback to smart seeds
                recommendations = beam_search_engine.get_smart_seeds(num_recommendations)
                print(f"Fallback: using smart seeds for session {session_id}")
            except Exception as fallback_error:
                print(f"Fallback also failed for session {session_id}: {fallback_error}")
                return jsonify({
                    'status': 'error',
                    'message': f'Enhanced beam search failed: {beam_error}, Fallback failed: {fallback_error}'
                }), 500
        
        # 🆕 Get enhanced performance summary with exploration stats
        try:
            performance_summary = beam_search_engine.get_performance_summary()
            preference_modes = beam_search_engine.get_preference_modes_summary()
            final_exploration_stats = beam_search_engine.get_exploration_stats()
        except Exception as metrics_error:
            print(f"Error getting enhanced metrics for session {session_id}: {metrics_error}")
            performance_summary = {'error': 'Failed to calculate performance metrics'}
            preference_modes = {'error': 'Failed to get preference modes'}
            final_exploration_stats = exploration_stats
        
        return jsonify({
            'status': 'success',
            'recommendations': recommendations,
            'step': step + 1,
            'session_id': session_id,
            'enhanced_metrics': {
                'performance_summary': performance_summary,
                'preference_modes': preference_modes,
                'exploration_stats': final_exploration_stats,
            'feedback_processed': {
                'likes': len(liked_indices),
                'dislikes': len(disliked_indices),
                'total_feedback_history': len(beam_search_engine.feedback_history)
                }
            },
            'system_info': {
                'beam_search_version': 'Enhanced v2',
                'multi_modal': True,
                'current_beam_width': beam_search_engine.current_beam_width,
                'exploration_factor': beam_search_engine.exploration_factor,
                'duplicate_prevention': 'enabled'
            }
        })
    
    except ValueError as ve:
        print(f"Session error: {str(ve)}")
        return jsonify({
            'status': 'error',
            'message': str(ve),
            'error_type': 'session_expired'
        }), 404
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
        session_id = data.get('session_id')
        watch_index = data.get('watch_index')
        feedback_type = data.get('feedback_type')  # 'like' or 'dislike'
        confidence = data.get('confidence', 1.0)  # 0.5 to 1.0
        
        if not session_id:
            return jsonify({
                'status': 'error',
                'message': 'session_id is required'
            }), 400
        
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
        
        # Get session-specific enhanced beam search engine
        beam_search_engine = session_manager.get_session(session_id)
        
        # Add feedback to the enhanced multi-modal system
        beam_search_engine.add_feedback(watch_index, feedback_type, confidence)
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback added successfully',
            'session_id': session_id,
            'total_feedback': len(beam_search_engine.feedback_history)
        })
    
    except ValueError as ve:
        return jsonify({
            'status': 'error',
            'message': str(ve),
            'error_type': 'session_expired'
        }), 404
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/preference-modes', methods=['GET'])
def get_preference_modes():
    """Get detailed information about user's preference modes."""
    try:
        session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({
                'status': 'error',
                'message': 'session_id parameter is required'
            }), 400
        
        # Get session-specific enhanced beam search engine
        beam_search_engine = session_manager.get_session(session_id)
        
        # Get comprehensive preference modes information
        preference_modes = beam_search_engine.get_preference_modes_summary()
        performance_summary = beam_search_engine.get_performance_summary()
        
        # Get recent feedback for context
        recent_feedback = list(beam_search_engine.feedback_history)[-10:]
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'preference_modes': preference_modes,
            'performance_summary': performance_summary,
            'recent_feedback': [
                {
                    'watch_index': f['watch_index'],
                    'type': f['type'],
                    'confidence': f['confidence'],
                    'timestamp': f['timestamp'],
                    'context': f.get('context', {})
                } for f in recent_feedback
            ],
            'system_status': {
                'total_feedback': len(beam_search_engine.feedback_history),
                'current_beam_width': beam_search_engine.current_beam_width,
                'exploration_factor': beam_search_engine.exploration_factor,
                'objective_weights': beam_search_engine.objective_weights,
                'user_engagement_level': beam_search_engine.user_profile['engagement_level']
            }
        })
    
    except ValueError as ve:
        return jsonify({
            'status': 'error',
            'message': str(ve),
            'error_type': 'session_expired'
        }), 404
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/get-variants', methods=['GET'])
def get_variants():
    """Get variant information for a specific watch."""
    try:
        session_id = request.args.get('session_id')
        watch_index = request.args.get('index', type=int)
        
        if not session_id:
            return jsonify({
                'status': 'error',
                'message': 'session_id parameter is required'
            }), 400
        
        if watch_index is None:
            return jsonify({
                'status': 'error',
                'message': 'index parameter is required'
            }), 400
        
        # Get session-specific beam search engine
        beam_search_engine = session_manager.get_session(session_id)
        
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
            'session_id': session_id,
            'watch_index': watch_index,
            'variant_count': len(variant_group),
            'representative_index': representative,
            'variants': variants
        })
    
    except ValueError as ve:
        return jsonify({
            'status': 'error',
            'message': str(ve),
            'error_type': 'session_expired'
        }), 404
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/variant-stats', methods=['GET'])
def get_variant_stats():
    """Get variant detection statistics (global, not session-specific)."""
    try:
        if not session_manager:
            return jsonify({
                'status': 'error',
                'message': 'Session manager not initialized'
            }), 500
        
        # Create a temporary engine to get variant stats (uses base data)
        temp_engine = EnhancedWatchBeamSearch(
            embeddings=session_manager.base_embeddings,
            watch_data=session_manager.base_watch_data,
            initial_beam_width=15,
            max_beam_width=30
        )
        variant_stats = temp_engine.variant_detector.get_variant_stats()
        
        return jsonify({
            'status': 'success',
            'variant_stats': variant_stats
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/modern-variant-stats', methods=['GET'])
def get_modern_variant_stats():
    """Get variant filtering statistics from the modern recommendation engine."""
    try:
        if not session_manager:
            return jsonify({
                'status': 'error',
                'message': 'Session manager not initialized'
            }), 500
        
        # Get modern engine instance (if available)
        if hasattr(session_manager, 'modern_engine'):
            variant_stats = session_manager.modern_engine.get_variant_filtering_stats()
        else:
            # Create temporary modern engine to get stats
            from models.modern_recommendation_engine import ModernRecommendationEngine
            temp_modern_engine = ModernRecommendationEngine(
                embeddings=session_manager.base_embeddings,
                watch_data=session_manager.base_watch_data,
                embeddings_pre_normalized=True
            )
            variant_stats = temp_modern_engine.get_variant_filtering_stats()
        
        return jsonify({
            'status': 'success',
            'variant_filtering_stats': variant_stats,
            'info': 'Modern recommendation engine variant filtering statistics'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get comprehensive recommendation system metrics for a specific session."""
    try:
        session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({
                'status': 'error',
                'message': 'session_id parameter is required'
            }), 400
        
        if not session_manager:
            return jsonify({
                'status': 'error',
                'message': 'Session manager not initialized'
            }), 500
        
        # Get session-specific beam search engine
        beam_search_engine = session_manager.get_session(session_id)
        
        metrics = beam_search_engine.get_comprehensive_metrics()
        variant_stats = beam_search_engine.variant_detector.get_variant_stats()
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'metrics': metrics,
            'system_info': {
                'total_watches': len(beam_search_engine.watch_data),
                'embedding_dimension': beam_search_engine.dimension,
                'style_clusters': len(beam_search_engine.cluster_to_watches),
                'initial_beam_width': beam_search_engine.initial_beam_width,
                'max_beam_width': beam_search_engine.max_beam_width,
                'variant_detection': variant_stats
            }
        })
    
    except ValueError as ve:
        return jsonify({
            'status': 'error',
            'message': str(ve),
            'error_type': 'session_expired'
        }), 404
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/reset-session', methods=['POST'])
def reset_session():
    """Reset the current recommendation session."""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                'status': 'error',
                'message': 'session_id is required'
            }), 400
        
        # Get session-specific enhanced beam search engine
        beam_search_engine = session_manager.get_session(session_id)
        
        # 🆕 Enhanced reset with seen watches tracking
        exploration_stats_before = beam_search_engine.get_exploration_stats()
        beam_search_engine.reset_seen_watches()
        exploration_stats_after = beam_search_engine.get_exploration_stats()
        
        return jsonify({
            'status': 'success',
            'message': 'Session reset successfully',
            'session_id': session_id,
            'reset_info': {
                'watches_seen_before': exploration_stats_before['seen_watches'],
                'watches_seen_after': exploration_stats_after['seen_watches'],
                'exploration_percentage_before': exploration_stats_before['exploration_percentage'],
                'brands_explored_before': exploration_stats_before['brands_explored'],
                'styles_explored_before': exploration_stats_before['styles_explored']
            }
        })
    
    except ValueError as ve:
        return jsonify({
            'status': 'error',
            'message': str(ve),
            'error_type': 'session_expired'
        }), 404
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/exploration-stats', methods=['GET'])
def get_exploration_stats():
    """Get exploration statistics for the current session."""
    try:
        session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({
                'status': 'error',
                'message': 'session_id parameter is required'
            }), 400
        
        # Get session-specific enhanced beam search engine
        beam_search_engine = session_manager.get_session(session_id)
        
        # Get exploration statistics
        exploration_stats = beam_search_engine.get_exploration_stats()
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'exploration_stats': exploration_stats
        })
    
    except ValueError as ve:
        return jsonify({
            'status': 'error',
            'message': str(ve),
            'error_type': 'session_expired'
        }), 404
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
        session_id = data.get('session_id')
        watch_index = data.get('watch_index')
        num_similar = data.get('num_similar', 5)
        
        if not session_id:
            return jsonify({
                'status': 'error',
                'message': 'session_id is required'
            }), 400
        
        if watch_index is None:
            return jsonify({
                'status': 'error',
                'message': 'watch_index is required'
            }), 400
        
        # Get session-specific beam search engine
        beam_search_engine = session_manager.get_session(session_id)
        
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
            'session_id': session_id,
            'similar_watches': similar_watches
        })
    
    except ValueError as ve:
        return jsonify({
            'status': 'error',
            'message': str(ve),
            'error_type': 'session_expired'
        }), 404
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/get-watch', methods=['GET'])
def get_watch():
    """Get a specific watch by index."""
    try:
        session_id = request.args.get('session_id')
        watch_index = request.args.get('index', type=int)
        
        if not session_id:
            return jsonify({
                'status': 'error',
                'message': 'session_id parameter is required'
            }), 400
        
        if watch_index is None:
            return jsonify({
                'status': 'error',
                'message': 'index parameter is required'
            }), 400
        
        # Get session-specific beam search engine
        beam_search_engine = session_manager.get_session(session_id)
        
        if watch_index < 0 or watch_index >= len(beam_search_engine.watch_data):
            return jsonify({
                'status': 'error',
                'message': 'Invalid watch index'
            }), 400
        
        watch = beam_search_engine.get_watch_by_index(watch_index)
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'watch': watch
        })
    
    except ValueError as ve:
        return jsonify({
            'status': 'error',
            'message': str(ve),
            'error_type': 'session_expired'
        }), 404
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about the watch database (global stats)."""
    try:
        if not session_manager:
            return jsonify({
                'status': 'error',
                'message': 'Session manager not initialized'
            }), 500
        
        # Use base data for global stats
        watch_data = session_manager.base_watch_data
        
        # Count watches by brand
        brand_counts = {}
        price_ranges = {'low': 0, 'mid': 0, 'high': 0, 'unknown': 0}
        
        for watch in watch_data:
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
            'total_watches': len(watch_data),
            'brands': brand_counts,
            'price_ranges': price_ranges,
            'embedding_dimension': session_manager.base_embeddings.shape[1],
            'active_sessions': session_manager.get_session_count()
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/session-info', methods=['GET'])
def get_session_info():
    """Get enhanced information about active sessions (admin endpoint)."""
    try:
        if not session_manager:
            return jsonify({
                'status': 'error',
                'message': 'Session manager not initialized'
            }), 500
        
        session_manager.cleanup_expired_sessions()
        
        session_info = []
        for session_id, session_data in session_manager.sessions.items():
            engine = session_data['engine']
            
            # Get enhanced metrics
            try:
                performance_summary = engine.get_performance_summary()
                preference_modes = engine.get_preference_modes_summary()
            except:
                performance_summary = {'error': 'metrics unavailable'}
                preference_modes = {'error': 'modes unavailable'}
            
            session_info.append({
                'session_id': session_id,
                'created_at': session_data['created_at'],
                'last_activity': session_data['last_activity'],
                'seen_watches': len(engine.seen_watches),
                'feedback_count': len(engine.feedback_history),
                'current_beam_width': engine.current_beam_width,
                'exploration_factor': engine.exploration_factor,
                'user_engagement_level': engine.user_profile['engagement_level'],
                'preference_modes_count': len(engine.preference_modes),
                'performance_summary': performance_summary,
                'preference_modes': preference_modes
            })
        
        return jsonify({
            'status': 'success',
            'active_sessions': len(session_info),
            'sessions': session_info,
            'system_info': {
                'beam_search_version': 'Enhanced v2',
                'multi_modal_support': True,
                'total_watches': len(session_manager.base_watch_data),
                'embedding_dimension': session_manager.base_embeddings.shape[1]
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/get-series', methods=['GET'])
def get_series():
    """Get watches from the same series as a specific watch."""
    try:
        session_id = request.args.get('session_id')
        watch_index = request.args.get('index', type=int)
        
        if not session_id:
            return jsonify({
                'status': 'error',
                'message': 'session_id parameter is required'
            }), 400
        
        if watch_index is None:
            return jsonify({
                'status': 'error',
                'message': 'index parameter is required'
            }), 400
        
        # Get session-specific beam search engine
        beam_search_engine = session_manager.get_session(session_id)
        
        if watch_index < 0 or watch_index >= len(beam_search_engine.watch_data):
            return jsonify({
                'status': 'error',
                'message': 'Invalid watch index'
            }), 400
        
        # Get series information
        series_watches = beam_search_engine.get_series_watches(watch_index)
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'watch_index': watch_index,
            'series_count': len(series_watches),
            'series_watches': series_watches
        })
    
    except ValueError as ve:
        return jsonify({
            'status': 'error',
            'message': str(ve),
            'error_type': 'session_expired'
        }), 404
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    import os
    print("Initializing beam search watch finder...")
    
    if initialize_session_manager():
        print("Starting Flask server...")
        port = int(os.environ.get('PORT', 5001))
        app.run(host='0.0.0.0', port=port, debug=False)  # Disable debug mode for production
    else:
        print("Failed to initialize session manager. Exiting.")
        sys.exit(1) 