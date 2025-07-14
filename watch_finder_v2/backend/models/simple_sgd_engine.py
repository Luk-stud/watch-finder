#!/usr/bin/env python3
"""
Simple SGD-Based Recommendation Engine

A drop-in replacement for SimpleVectorEngine that uses
scikit-learn's SGDClassifier for online learning.

Key features:
- Online learning with SGDClassifier (logistic regression)
- Session-based models that learn from user feedback
- Same interface as SimpleVectorEngine for easy replacement
- Scikit-learn's robust preprocessing and optimization
"""

import os
import pickle
import time
import logging
import numpy as np
from typing import Dict, List, Set, Optional, Any
import traceback

# Scikit-learn imports
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class SimpleSgdEngine:
    """
    A drop-in replacement for SimpleVectorEngine that uses
    scikit-learn's SGDClassifier for online learning.
    """

    def __init__(self,
                 batch_size: int = 3,
                 data_dir: Optional[str] = None,
                 like_weight: float = 5.0,
                 alpha: float = 0.0001,
                 prior_like_rate: float = 0.2):
        
        self.batch_size = batch_size
        self.data_dir = data_dir or "data"
        
        # --- Model Tuning Parameters ---
        self.like_weight = like_weight
        self.alpha = alpha
        self.prior_like_rate = prior_like_rate
        # ---
        
        # Data storage
        self.watch_data = {}
        self.items_matrix = None  # (N×D) normalized embeddings matrix
        self.watch_id_to_idx = {}  # Map watch_id -> matrix row index
        self.idx_to_watch_id = {}  # Map matrix row index -> watch_id
        self.available_watches = set()
        self.dim = 200
        
        # Load precomputed embeddings just like before
        self._load_embeddings()
        
        # Build and "warm up" a global scaler
        self.scaler = StandardScaler()
        self.scaler.partial_fit(self.items_matrix)   # fit on all items
        
        # Session-state stores one SGDClassifier per user
        self.session_models: Dict[str, SGDClassifier] = {}
        self.session_shown_watches: Dict[str, Set[int]] = {}
        self.session_interaction_counts: Dict[str, int] = {}
        self.session_timestamps: Dict[str, List[float]] = {}
        self.session_initialized: Dict[str, bool] = {}
        # Track liked embeddings per session to build a user preference vector.
        self.session_liked_embeddings: Dict[str, List[np.ndarray]] = {}

        logger.info("🔧 SGD Engine configured with the following parameters:")
        logger.info(f"   - Like Weight: {self.like_weight}")
        logger.info(f"   - Alpha (Regularization): {self.alpha}")
        logger.info(f"   - Prior Like Rate: {self.prior_like_rate}")

    def _load_embeddings(self) -> None:
        """Load precomputed embeddings and create items matrix."""
        logger.info("🚀 Initializing Simple SGD Engine...")
        total_start = time.time()
        
        try:
            precomputed_path = os.path.join(self.data_dir, 'precomputed_embeddings.pkl')
            logger.info(f"🔍 Looking for embeddings at: {precomputed_path}")
            logger.info(f"🔍 Absolute path: {os.path.abspath(precomputed_path)}")
            
            if not os.path.exists(precomputed_path):
                logger.error(f"❌ Precomputed file not found: {precomputed_path}")
                self._create_fallback_data()
                return
                
            # Load precomputed data
            logger.info(f"📖 Loading embeddings from: {precomputed_path}")
            with open(precomputed_path, 'rb') as f:
                precomputed_data = pickle.load(f)
            
            logger.info(f"📊 Loaded data keys: {list(precomputed_data.keys())}")
            logger.info(f"📊 Embedding dim from file: {precomputed_data.get('embedding_dim', 'NOT_FOUND')}")
            logger.info(f"📊 Number of watches from file: {len(precomputed_data.get('watch_data', {}))}")
            logger.info(f"📊 Number of embeddings from file: {len(precomputed_data.get('final_embeddings', {}))}")
            
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
            
            self.available_watches = set(watch_ids)
            
            # Build brand+model grouping for variant exclusion
            self._build_brand_model_groups()
            
            total_time = time.time() - total_start
            file_size = os.path.getsize(precomputed_path) / (1024 * 1024)
            
            logger.info(f"✅ Simple SGD Engine initialized in {total_time:.2f}s:")
            logger.info(f"   • File size: {file_size:.1f}MB")
            logger.info(f"   • Items matrix: {self.items_matrix.shape}")
            logger.info(f"   • Available watches: {len(self.available_watches)}")
            logger.info(f"   • Brand+model groups: {len(self.brand_model_groups)}")
            logger.info(f"   • StandardScaler fitted on all items")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}")
            logger.error(f"❌ Error details: {traceback.format_exc()}")
            self._create_fallback_data()

    def _build_brand_model_groups(self) -> None:
        """Build mapping of brand+model combinations to watch IDs for variant exclusion."""
        self.brand_model_groups = {}
        self.watch_to_brand_model = {}
        
        for watch_id, watch_data in self.watch_data.items():
            brand = watch_data.get('brand', '').strip()
            model = watch_data.get('model', '').strip()
            
            # Create a unique key for brand+model combination
            brand_model_key = f"{brand}|{model}"
            
            if brand_model_key not in self.brand_model_groups:
                self.brand_model_groups[brand_model_key] = set()
            
            self.brand_model_groups[brand_model_key].add(watch_id)
            self.watch_to_brand_model[watch_id] = brand_model_key
        
        # Log some statistics about the grouping
        group_sizes = [len(watch_ids) for watch_ids in self.brand_model_groups.values()]
        multi_variant_groups = [size for size in group_sizes if size > 1]
        
        logger.info(f"📊 Brand+model grouping statistics:")
        logger.info(f"   • Total unique brand+model combinations: {len(self.brand_model_groups)}")
        logger.info(f"   • Groups with multiple variants: {len(multi_variant_groups)}")
        if multi_variant_groups:
            logger.info(f"   • Average variants per group: {sum(multi_variant_groups)/len(multi_variant_groups):.1f}")
            logger.info(f"   • Max variants in a group: {max(multi_variant_groups)}")
        
        # Show some examples of multi-variant groups
        multi_variant_examples = [(key, len(watch_ids)) for key, watch_ids in self.brand_model_groups.items() if len(watch_ids) > 1][:3]
        if multi_variant_examples:
            logger.info(f"   • Example multi-variant groups:")
            for key, count in multi_variant_examples:
                brand, model = key.split('|', 1)
                logger.info(f"     - {brand} {model}: {count} variants")

    def _get_similar_watches(self, watch_id) -> Set[int]:
        """Get all watches with the same brand+model as the given watch."""
        brand_model_key = self.watch_to_brand_model.get(watch_id)
        if brand_model_key:
            return self.brand_model_groups.get(brand_model_key, set())
        return {watch_id}  # If not found, just return the watch itself

    def _create_blank_model(self) -> SGDClassifier:
        """Creates a new SGDClassifier with our desired default settings."""
        # Based on expert recommendations for online learning:
        # 1. Initialize the intercept (bias) to a reasonable prior.
        initial_intercept = np.log(self.prior_like_rate / (1 - self.prior_like_rate))

        # 2. Use ASGD (average=True) for smoother, more stable updates.
        # Use a small constant learning rate to prevent large coefficient swings
        model = SGDClassifier(
            loss="log_loss",            # Logistic regression
            learning_rate="constant",   # Fixed step size instead of "optimal"
            eta0=0.05,                   # Smaller step → gentler updates
            alpha=self.alpha,            # Regularization strength
            average=True,                # Averaged SGD for smoother convergence
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            warm_start=True
        )
        
        # 3. Correctly initialize the model with our prior (as per user's excellent insight).
        #    First, do a dummy fit to create the model's internal attributes.
        dummy_X = np.zeros((1, self.dim))
        model.partial_fit(dummy_X, [0], classes=[0, 1])

        #    Now, overwrite the initialized coef_/intercept_ with our desired cold start values.
        model.coef_ = np.zeros(model.coef_.shape)
        model.intercept_ = np.array([initial_intercept])
        
        return model

    def create_session(self, session_id: str) -> None:
        """Initialize a new session with a fresh SGD model."""
        model = self._create_blank_model()
        
        # Store session state
        self.session_models[session_id] = model
        self.session_shown_watches[session_id] = set()
        self.session_interaction_counts[session_id] = 0
        self.session_timestamps[session_id] = []
        self.session_initialized[session_id] = False  # Track if model has seen any real data
        
        logger.info(f"✅ Created Simple SGD session {session_id} with robust prior initialization")

        # Initialize liked embeddings list for preference centroid
        self.session_liked_embeddings[session_id] = []

    def get_recommendations(self,
                          session_id: str,
                          exclude_ids: Optional[Set[int]] = None
                          ) -> List[Dict[str, Any]]:
        """Rank all unseen watches by the session's raw model score."""
        exclude_ids = exclude_ids or set()
        
        if session_id not in self.session_models:
            self.create_session(session_id)

        model = self.session_models[session_id]
        shown = self.session_shown_watches[session_id]
        
        # Expand shown watches to include all variants of the same brand+model
        expanded_shown = set()
        for watch_id in shown:
            similar_watches = self._get_similar_watches(watch_id)
            expanded_shown.update(similar_watches)
        
        all_excludes = exclude_ids | expanded_shown

        # Build list of candidate indices
        candidates = [
            self.watch_id_to_idx[w]
            for w in self.available_watches
            if w not in all_excludes
        ]
        if not candidates:
            return []

        # Prepare feature matrix for candidates
        X_cand = self.scaler.transform(self.items_matrix[candidates])

        # IMPROVED COLD START: Use similarity-based recommendations instead of random
        if not self.session_initialized[session_id]:
            # Before any training, use global centroid similarity for initial recommendations
            # This gives much better cold start than random scores
            global_centroid = np.mean(self.items_matrix, axis=0)
            global_centroid = global_centroid / np.linalg.norm(global_centroid)
            
            # Compute cosine similarities to global centroid
            scores = self.items_matrix[candidates] @ global_centroid
            logger.debug(f"🎯 SGD using global centroid similarity scores (no training data yet)")
        else:
            # Compute calibrated probabilities (exploitation term)
            proba = model.predict_proba(X_cand)[:, 1]

            # --- Preference centroid similarity (stability term) ---
            pref_vector = self._get_preference_vector(session_id)
            cosine_sims = self.items_matrix[candidates] @ pref_vector

            # Blend: 70% model probability, 30% cosine similarity to preference
            scores = 0.7 * proba + 0.3 * cosine_sims
            logger.debug(
                "🎲 SGD probability scores: min={:.2f}, max={:.2f}, mean={:.2f}".format(
                    np.min(scores), np.max(scores), np.mean(scores)
                )
            )
            # Debug: Show top scores before selection
            top_10_scores = np.sort(scores)[-10:][::-1]
            logger.debug(f"🎯 SGD top-10 P(like) scores: {[f'{s:.2f}' for s in top_10_scores]}")

        # Select top-k by raw score, but ensure we don't return multiple watches from the same brand+model group
        k = min(self.batch_size, len(scores))
        if k == len(scores):
            top_k_local = np.arange(len(scores))
        else:
            top_k_local = self._argpartition_descending(scores, k)[:k]
        
        sorted_idx = top_k_local[self._argsort_descending(scores[top_k_local])]

        # Format recommendations, ensuring no duplicates from same brand+model group
        recommendations = []
        seen_groups = set()
        
        for loc in sorted_idx:
            global_idx = candidates[loc]
            watch_id = self.idx_to_watch_id[global_idx]
            score = float(scores[loc])
            
            # Check if this watch's brand+model group is already in this batch
            watch_data = self.watch_data.get(watch_id, {})
            brand = watch_data.get('brand', '').lower().strip()
            model = watch_data.get('model', '').lower().strip()
            group_key = f"{brand}|{model}"
            
            if group_key in seen_groups:
                # Skip this watch as we already have one from this group in this batch
                continue
            
            # Add this watch and all its variants to shown set
            similar_watches = self._get_similar_watches(watch_id)
            shown.update(similar_watches)
            
            # Mark this group as seen in this batch
            seen_groups.add(group_key)
            
            if not self.session_initialized[session_id]:
                algorithm = "sgd_centroid_similarity"
            else:
                algorithm = "sgd_blend_prob_cosine"
            recommendations.append(
                self._format_recommendation(watch_id, score, algorithm)
            )
            
            # Stop if we have enough recommendations
            if len(recommendations) >= self.batch_size:
                break

        scores_str = [f"{r['confidence']:.2f}" for r in recommendations]
        logger.info(f"🎯 SGD recommendations for session {session_id}: "
                   f"{len(recommendations)} watches with scores {scores_str}")

        return recommendations

    def update(self, session_id: str, watch_id: int, reward: float) -> None:
        """Stream the new example into the session's SGD model."""
        if session_id not in self.session_models:
            self.create_session(session_id)

        model = self.session_models[session_id]
        idx = self.watch_id_to_idx.get(watch_id)
        if idx is None:
            logger.warning(f"Watch {watch_id} not found in embeddings")
            return

        # Prepare the feature vector
        x = self.items_matrix[idx]
        x_scaled = self.scaler.transform([x])

        # Label: 1 for "yes", 0 for "no"
        y = 1 if reward > 0 else 0

        # Expert recommendation: Use sample_weight to handle imbalance
        # Give more weight to the rare "like" events
        sample_weight = np.array([self.like_weight if y == 1 else 1.0])

        # Online update
        try:
            # For the first example, we need to specify classes
            if not self.session_initialized[session_id]:
                model.partial_fit(x_scaled, [y], classes=[0, 1], sample_weight=sample_weight)
                self.session_initialized[session_id] = True
                logger.debug(f"🎯 SGD initialized with first real example: {'like' if y == 1 else 'dislike'}")
            else:
                model.partial_fit(x_scaled, [y], sample_weight=sample_weight)
            
            feedback_type = "like" if y == 1 else "dislike"
            logger.debug(f"📚 SGD learned from {feedback_type} on watch {watch_id} with weight {sample_weight[0]}")
        except Exception as e:
            logger.error(f"SGD partial_fit failed: {e}")

        # Track session stats
        self.session_interaction_counts[session_id] += 1
        self.session_timestamps[session_id].append(time.time())
        
        # Debug: Track class balance
        if not hasattr(self, 'session_likes'):
            self.session_likes = {}
            self.session_dislikes = {}
        
        if session_id not in self.session_likes:
            self.session_likes[session_id] = 0
            self.session_dislikes[session_id] = 0
        
        if y == 1:
            self.session_likes[session_id] += 1

            # Store embedding of liked watch for preference vector
            self.session_liked_embeddings[session_id].append(x)
        else:
            self.session_dislikes[session_id] += 1
        
        interaction_count = self.session_interaction_counts[session_id]
        likes = self.session_likes[session_id]
        dislikes = self.session_dislikes[session_id]
        
        logger.info(f"📊 Updated SGD session {session_id} (interaction #{interaction_count}): "
                   f"L={likes}, D={dislikes}, ratio={likes/(likes+dislikes):.2f}")

    def _argpartition_descending(self, array: np.ndarray, k: int) -> np.ndarray:
        """Returns indices of top-k largest values."""
        return np.argpartition(array, -k)[-k:]

    def _argsort_descending(self, array: np.ndarray) -> np.ndarray:
        """Returns indices that would sort array descending."""
        return np.argsort(array)[::-1]

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
        
        # Map DINO image fields to frontend-expected fields
        # Always check for image_url, regardless of image_path
        if formatted_data.get('image_url'):
            formatted_data['main_image'] = formatted_data['image_url']
        else:
            formatted_data['main_image'] = None
        formatted_data['local_image_path'] = None
        
        return {
            **formatted_data,
            # Preserve original watch_id type; attempt int cast only if possible
            'watch_id': (int(watch_id) if isinstance(watch_id, (int, np.integer)) or (isinstance(watch_id, str) and watch_id.isdigit()) else str(watch_id)),
            'confidence': float(confidence),
            'algorithm': str(algorithm)
        }

    def _create_fallback_data(self) -> None:
        """Create minimal fallback data if precomputed file is missing."""
        logger.warning("⚠️ Creating fallback data for Simple SGD Engine")
        
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
        self.dim = 200
        
        logger.info("✅ Created fallback data with 10 sample watches")

    def shutdown(self) -> None:
        """Clean up any session state."""
        logger.info("🔄 Shutting down Simple SGD Engine...")
        self.session_models.clear()
        self.session_shown_watches.clear()
        self.session_interaction_counts.clear()
        self.session_timestamps.clear()
        self.session_initialized.clear()
        if hasattr(self, 'session_likes'):
            self.session_likes.clear()
            self.session_dislikes.clear()
        logger.info("✅ Simple SGD Engine shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """Return engine statistics."""
        # Compute interaction statistics
        interaction_counts = list(self.session_interaction_counts.values())
        
        return {
            "total_sessions": len(self.session_models),
            "total_items": len(self.available_watches),
            "batch_size": self.batch_size,
            "avg_interactions_per_session": float(np.mean(interaction_counts)) if interaction_counts else 0.0,
            "max_interactions_per_session": max(interaction_counts) if interaction_counts else 0,
            "algorithm": "SGDClassifier (logistic)"
        } 

    def _get_preference_vector(self, session_id: str) -> np.ndarray:
        """Return a unit vector representing the centroid of liked embeddings for the session.

        Falls back to global centroid if the user has no likes yet.
        """
        liked = self.session_liked_embeddings.get(session_id, [])
        if not liked:
            # Global centroid (already normalised rows) for cold start / no likes yet
            if not hasattr(self, '_global_centroid'):
                gc = np.mean(self.items_matrix, axis=0)
                self._global_centroid = gc / np.linalg.norm(gc)
            return self._global_centroid

        centroid = np.mean(liked, axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            return centroid  # rare degenerate case
        return centroid / norm 