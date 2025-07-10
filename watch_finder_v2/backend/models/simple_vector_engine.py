#!/usr/bin/env python3
"""
Simple Vector-Based Recommendation Engine

A clean, efficient approach using normalized embeddings and cosine similarity.
Much simpler than MABWiser but potentially just as effective.

Key features:
- Single user profile vector that evolves with feedback
- Cosine similarity for recommendations  
- Optional repulsion for negative feedback
- Fast matrix operations with argpartition
- No complex multi-armed bandit logic
"""

import os
import pickle
import time
import logging
import numpy as np
from typing import Dict, List, Set, Optional, Any
import traceback

logger = logging.getLogger(__name__)

class SimpleVectorEngine:
    """Simple vector-based recommendation engine using cosine similarity."""
    
    def __init__(self, 
                 batch_size: int = 3,
                 repel_strength: float = 0.3,  # β ∈ [0,1] for negative feedback
                 decay_factor: float = 0.95,  # γ ∈ [0,1] for profile decay
                 data_dir: Optional[str] = None):
        
        self.batch_size = batch_size
        self.repel_strength = repel_strength  # How much to repel from disliked items
        self.decay_factor = decay_factor  # How much to decay profile with each update
        self.data_dir = data_dir or "data"
        
        # Data storage
        self.watch_data = {}
        self.items_matrix = None  # (N×D) normalized embeddings matrix
        self.watch_id_to_idx = {}  # Map watch_id -> matrix row index
        self.idx_to_watch_id = {}  # Map matrix row index -> watch_id
        self.available_watches = set()
        self.global_centroid = None
        self.dim = 200
        
        # Session management
        self.session_user_sums = {}   # session_id -> user_sum vector
        self.session_user_vecs = {}   # session_id -> normalized user_vec
        self.session_shown_watches = {}
        self.session_interaction_counts = {}  # session_id -> number of interactions
        self.session_timestamps = {}  # session_id -> list of interaction timestamps
        
        # Load and initialize
        self._load_and_initialize()
        
    def _load_and_initialize(self) -> None:
        """Load precomputed embeddings and create items matrix."""
        logger.info("🚀 Initializing Simple Vector Engine...")
        total_start = time.time()
        
        try:
            precomputed_path = os.path.join(self.data_dir, 'precomputed_embeddings.pkl')
            
            if not os.path.exists(precomputed_path):
                logger.error(f"❌ Precomputed file not found: {precomputed_path}")
                self._create_fallback_data()
                return
                
            # Load precomputed data
            with open(precomputed_path, 'rb') as f:
                precomputed_data = pickle.load(f)
            
            self.watch_data = precomputed_data['watch_data']
            final_embeddings = precomputed_data['final_embeddings']
            self.dim = precomputed_data['embedding_dim']
            
            # Build items matrix and mappings
            watch_ids = list(final_embeddings.keys())
            n_items = len(watch_ids)
            
            self.items_matrix = np.zeros((n_items, self.dim), dtype=np.float32)
            
            for idx, watch_id in enumerate(watch_ids):
                self.items_matrix[idx] = final_embeddings[watch_id]
                self.watch_id_to_idx[watch_id] = idx
                self.idx_to_watch_id[idx] = watch_id
            
            # Embeddings are already normalized from preprocessing
            
            # Compute global centroid
            self.global_centroid = np.mean(self.items_matrix, axis=0)
            self.global_centroid = self._normalize_vector(self.global_centroid)
            
            self.available_watches = set(watch_ids)
            
            total_time = time.time() - total_start
            file_size = os.path.getsize(precomputed_path) / (1024 * 1024)
            
            logger.info(f"✅ Simple Vector Engine initialized in {total_time:.2f}s:")
            logger.info(f"   • File size: {file_size:.1f}MB")
            logger.info(f"   • Items matrix: {self.items_matrix.shape}")
            logger.info(f"   • Available watches: {len(self.available_watches)}")
            logger.info(f"   • Global centroid computed")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}")
            logger.error(f"❌ Error details: {traceback.format_exc()}")
            self._create_fallback_data()
    
    def _normalize_vector(self, v: np.ndarray) -> np.ndarray:
        """Normalize a single vector."""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm
    

    
    def create_session(self, session_id: str) -> None:
        """Initialize a new session with global centroid as starting point."""
        # Start with global centroid as user profile
        self.session_user_sums[session_id] = np.zeros(self.dim, dtype=np.float32)
        self.session_user_vecs[session_id] = self.global_centroid.copy()
        self.session_shown_watches[session_id] = set()
        self.session_interaction_counts[session_id] = 0
        self.session_timestamps[session_id] = []
        
        logger.info(f"✅ Created Simple Vector session {session_id}")
    
    def get_recommendations(self,
                          session_id: str,
                          exclude_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """Get recommendations using cosine similarity to user profile."""
        exclude_ids = exclude_ids or set()
        
        # Ensure session exists
        if session_id not in self.session_user_vecs:
            self.create_session(session_id)
        
        # Get session state
        user_vec = self.session_user_vecs[session_id]
        session_shown = self.session_shown_watches[session_id]
        all_excludes = exclude_ids | session_shown
        
        # Create availability mask
        available_indices = []
        available_watch_ids = []
        
        for watch_id in self.available_watches:
            if watch_id not in all_excludes:
                idx = self.watch_id_to_idx[watch_id]
                available_indices.append(idx)
                available_watch_ids.append(watch_id)
        
        if not available_indices:
            return []
        
        available_indices = np.array(available_indices)
        
        # Compute cosine similarities: efficient matrix multiplication
        similarities = self.items_matrix[available_indices] @ user_vec
        
        # Find top-k using argpartition (faster than full sort)
        k = min(self.batch_size, len(similarities))
        if k == len(similarities):
            top_k_local = np.arange(len(similarities))
        else:
            top_k_local = np.argpartition(similarities, -k)[-k:]
        
        # Sort the top-k by similarity (descending)
        top_k_similarities = similarities[top_k_local]
        sort_order = np.argsort(top_k_similarities)[::-1]  # descending
        top_k_local = top_k_local[sort_order]
        
        # Convert back to watch IDs and format recommendations
        recommendations = []
        for local_idx in top_k_local:
            global_idx = available_indices[local_idx]
            watch_id = self.idx_to_watch_id[global_idx]
            similarity = similarities[local_idx]
            
            # Track shown watches
            session_shown.add(watch_id)
            
            recommendations.append(
                self._format_recommendation(watch_id, float(similarity), "simple_vector")
            )
        
        similarities_str = [f"{r['confidence']:.3f}" for r in recommendations]
        logger.info(f"🎯 Simple Vector recommendations for session {session_id}: "
                   f"{len(recommendations)} watches with similarities {similarities_str}")
        
        return recommendations
    
    def update(self, session_id: str, watch_id: int, reward: float) -> None:
        """Update user profile with feedback and decay."""
        # Ensure session exists
        if session_id not in self.session_user_vecs:
            self.create_session(session_id)
        
        # Get watch embedding
        if watch_id not in self.watch_id_to_idx:
            logger.warning(f"Watch {watch_id} not found in embeddings")
            return
        
        idx = self.watch_id_to_idx[watch_id]
        item_vec = self.items_matrix[idx]
        
        # Get current user state
        user_sum = self.session_user_sums[session_id]
        old_norm = np.linalg.norm(user_sum)
        
        # Apply decay to existing profile (before adding new feedback)
        if self.session_interaction_counts[session_id] > 0:  # Don't decay on first interaction
            user_sum *= self.decay_factor
            logger.debug(f"🕰️ Applied decay (γ={self.decay_factor}) to session {session_id}")
        
        # Update user sum based on feedback
        if reward > 0:  # Like: +1
            user_sum += item_vec
            logger.debug(f"👍 Added watch {watch_id} to user profile")
        else:  # Dislike: repel with strength β
            user_sum -= self.repel_strength * item_vec
            logger.debug(f"👎 Repelled watch {watch_id} from user profile (β={self.repel_strength})")
        
        # Update session tracking
        self.session_interaction_counts[session_id] += 1
        self.session_timestamps[session_id].append(time.time())
        
        # Update normalized profile
        self.session_user_vecs[session_id] = self._normalize_vector(user_sum)
        
        # Store updated sum
        self.session_user_sums[session_id] = user_sum
        
        # Log profile evolution
        new_norm = np.linalg.norm(user_sum)
        interaction_count = self.session_interaction_counts[session_id]
        logger.info(f"📊 Updated session {session_id} (interaction #{interaction_count}): "
                   f"profile norm {old_norm:.3f} → {new_norm:.3f}")
    
    def _format_recommendation(self, watch_id: int, confidence: float, algorithm: str) -> Dict[str, Any]:
        """Format a recommendation with watch data."""
        watch_data = self.watch_data.get(watch_id, {})
        
        # Convert any numpy types to Python native types for JSON serialization
        formatted_data = {}
        for key, value in watch_data.items():
            if isinstance(value, np.ndarray):
                if value.size == 1:  # Single element array
                    formatted_data[key] = value.item()
                else:  # Multi-element array
                    formatted_data[key] = value.tolist()
            elif hasattr(value, 'item') and hasattr(value, 'dtype'):  # numpy scalar
                formatted_data[key] = value.item()
            else:
                formatted_data[key] = value
        
        return {
            **formatted_data,
            'watch_id': int(watch_id),
            'confidence': float(confidence),
            'algorithm': str(algorithm)
        }
    
    def _create_fallback_data(self) -> None:
        """Create minimal fallback data if precomputed file is missing."""
        logger.warning("⚠️ Creating fallback data for Simple Vector Engine")
        
        # Create minimal data
        n_items = 10
        self.watch_data = {i: {'watch_id': i, 'brand': 'Sample', 'model': f'Watch {i}'} for i in range(n_items)}
        
        # Random normalized embeddings for fallback
        self.items_matrix = np.random.randn(n_items, 200).astype(np.float32)
        # Normalize rows manually for fallback data
        norms = np.linalg.norm(self.items_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.items_matrix = self.items_matrix / norms
        
        # Mappings
        for i in range(n_items):
            self.watch_id_to_idx[i] = i
            self.idx_to_watch_id[i] = i
        
        self.available_watches = set(range(n_items))
        self.global_centroid = np.mean(self.items_matrix, axis=0)
        self.global_centroid = self._normalize_vector(self.global_centroid)
        self.dim = 200
        
        logger.info("✅ Created fallback data with 10 sample watches")
    
    def shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("🔄 Shutting down Simple Vector Engine...")
        self.session_user_sums.clear()
        self.session_user_vecs.clear()
        self.session_shown_watches.clear()
        self.session_interaction_counts.clear()
        self.session_timestamps.clear()
        logger.info("✅ Simple Vector Engine shutdown complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        total_sessions = len(self.session_user_vecs)
        
        # Compute average profile norms
        profile_norms = []
        for session_id in self.session_user_sums:
            norm = np.linalg.norm(self.session_user_sums[session_id])
            profile_norms.append(norm)
        
        # Compute interaction statistics
        interaction_counts = list(self.session_interaction_counts.values())
        
        return {
            'total_sessions': total_sessions,
            'avg_profile_norm': float(np.mean(profile_norms)) if profile_norms else 0.0,
            'max_profile_norm': float(max(profile_norms)) if profile_norms else 0.0,
            'total_items': len(self.available_watches),
            'repel_strength': float(self.repel_strength),
            'decay_factor': float(self.decay_factor),
            'avg_interactions_per_session': float(np.mean(interaction_counts)) if interaction_counts else 0.0,
            'max_interactions_per_session': max(interaction_counts) if interaction_counts else 0,
            'algorithm': 'Simple Vector (Cosine Similarity + Decay)'
        } 