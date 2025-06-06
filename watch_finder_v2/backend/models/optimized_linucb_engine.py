"""
Optimized Multi-Expert LinUCB Recommendation Engine
================================================

Implements multiple specialized LinUCB experts with performance optimizations:
- Pre-computed session embeddings
- Cached matrix operations
- Vectorized batch processing
- Memory reuse
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import logging
import random

# Add sklearn for clustering
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except ImportError:
    print("⚠️  sklearn not installed. Run: pip install scikit-learn")
    KMeans = None
    StandardScaler = None
    PCA = None

logger = logging.getLogger(__name__)

class OptimizedArm:
    """Represents a single arm (watch) with optimized matrix operations."""
    def __init__(self, dim: int):
        self.A: np.ndarray = np.identity(dim)  # d x d matrix
        self.b: np.ndarray = np.zeros(dim)     # d-dimensional vector
        self.theta: Optional[np.ndarray] = None # Coefficient vector
        self._cached_A_inv: Optional[np.ndarray] = None
        
        # Watch-specific data
        self.total_pulls: int = 0
        self.positive_feedback: int = 0
        self.last_pull: Optional[datetime] = None
        self.features: Optional[np.ndarray] = None
    
    def update(self, context: np.ndarray, reward: float) -> None:
        """Update arm parameters with new observation."""
        self.A += np.outer(context, context)
        self.b += reward * context
        self.theta = None  # Reset cached theta
        self._cached_A_inv = None  # Reset cached inverse
        
        self.total_pulls += 1
        if reward > 0:
            self.positive_feedback += 1
        self.last_pull = datetime.now()
    
    def get_theta(self) -> np.ndarray:
        """Get coefficient vector, computing if necessary."""
        if self.theta is None:
            try:
                if self._cached_A_inv is None:
                    # Add regularization for numerical stability
                    A_reg = self.A + 0.01 * np.identity(self.A.shape[0])
                    self._cached_A_inv = np.linalg.inv(A_reg)
                self.theta = self._cached_A_inv @ self.b
            except np.linalg.LinAlgError:
                # Fallback for numerical issues
                self.theta = np.zeros(self.b.shape)
        return self.theta
    
    def get_ucb(self, context: np.ndarray, alpha: float) -> float:
        """Calculate UCB score with cached matrix operations."""
        theta = self.get_theta()
        mean = np.dot(theta, context)
        
        try:
            if self._cached_A_inv is None:
                A_reg = self.A + 1e-6 * np.identity(self.A.shape[0])
                self._cached_A_inv = np.linalg.inv(A_reg)
            
            confidence_width = alpha * np.sqrt(np.dot(context.T, np.dot(self._cached_A_inv, context)))
            
            # Ensure confidence width is finite and positive
            if not np.isfinite(confidence_width) or confidence_width < 0:
                confidence_width = alpha * 0.1  # Safe fallback
                
        except (np.linalg.LinAlgError, ValueError):
            # Fallback for numerical issues
            confidence_width = alpha * np.sqrt(np.dot(context, context) / max(1, self.total_pulls))
            
        return mean + confidence_width

    def batch_ucb(self, contexts: np.ndarray, alpha: float) -> np.ndarray:
        """Calculate UCB scores for multiple contexts at once."""
        theta = self.get_theta()
        means = contexts @ theta  # Vectorized dot product
        
        try:
            if self._cached_A_inv is None:
                A_reg = self.A + 1e-6 * np.identity(self.A.shape[0])
                self._cached_A_inv = np.linalg.inv(A_reg)
            
            # Vectorized confidence width calculation
            confidence_widths = alpha * np.sqrt(np.sum(contexts @ self._cached_A_inv * contexts, axis=1))
            confidence_widths = np.clip(confidence_widths, 0, alpha)  # Ensure positive and bounded
            
        except (np.linalg.LinAlgError, ValueError):
            # Fallback using simpler confidence calculation
            confidence_widths = alpha * np.sqrt(np.sum(contexts * contexts, axis=1) / max(1, self.total_pulls))
            
        return means + confidence_widths

class OptimizedExpertLinUCB:
    """Single LinUCB expert with optimized operations."""
    def __init__(self, expert_id: int, dim: int, alpha: float):
        self.expert_id = expert_id
        self.dim = dim
        self.alpha = alpha
        self.arms: Dict[int, OptimizedArm] = {}
        self.centroid: Optional[np.ndarray] = None
        self.liked_watches: List[int] = []
        self._context_buffer: np.ndarray = np.zeros(dim)  # Pre-allocated buffer
        
    def add_liked_watch(self, watch_id: int, embedding: np.ndarray):
        """Add a liked watch to this expert's learned preferences."""
        self.liked_watches.append(watch_id)
        
        # Update centroid
        if self.centroid is None:
            self.centroid = embedding.copy()
        else:
            # Running average of liked watch embeddings
            alpha = 1.0 / len(self.liked_watches)
            self.centroid = (1 - alpha) * self.centroid + alpha * embedding
        
        # Normalize centroid for proper cosine similarity
        norm = np.linalg.norm(self.centroid)
        if norm > 0:
            self.centroid = self.centroid / norm
        
        # Create an arm for this watch if it doesn't exist, using combined context
        if watch_id not in self.arms:
            arm = OptimizedArm(self.dim)
            arm.features = embedding
            self.arms[watch_id] = arm
            logger.debug(f"🎯 Expert {self.expert_id}: Added arm for watch {watch_id} | Total arms: {len(self.arms)}")
    
    def update(self, watch_id: int, reward: float, watch_embedding: np.ndarray) -> None:
        """Update expert with feedback using consistent combined context."""
        # Create combined context for both training and scoring consistency
        if self.centroid is not None:
            combined_context = self._combine_context(self.centroid, watch_embedding)
        else:
            # If no centroid yet, use the embedding itself (padded/truncated to dim)
            combined_context = self._prepare_context(watch_embedding)
        
        # Create arm if it doesn't exist
        if watch_id not in self.arms:
            arm = OptimizedArm(self.dim)
            arm.features = watch_embedding  # Store original embedding as features
            self.arms[watch_id] = arm
        
        # Update arm with feedback using combined context
        arm = self.arms[watch_id]
        arm.update(combined_context, reward)
        
        # Update liked watches list and centroid if positive feedback
        if reward > 0 and watch_id not in self.liked_watches:
            self.add_liked_watch(watch_id, watch_embedding)
    
    def _prepare_context(self, embedding: np.ndarray) -> np.ndarray:
        """Prepare context when no centroid exists yet."""
        if len(embedding) >= self.dim:
            return embedding[:self.dim] / np.linalg.norm(embedding[:self.dim])
        else:
            padded = np.zeros(self.dim)
            padded[:len(embedding)] = embedding
            norm = np.linalg.norm(padded)
            return padded / norm if norm > 0 else padded
    
    def _combine_context(self, centroid: np.ndarray, watch_embedding: np.ndarray) -> np.ndarray:
        """Combine expert centroid with watch embedding creating a unique context."""
        half_dim = self.dim // 2
        
        # Combine expert preferences with watch features
        # Use expert centroid (text) + watch's unique CLIP features for differentiation
        expert_text = centroid[:half_dim]
        watch_text = watch_embedding[:half_dim]  
        watch_clip = watch_embedding[half_dim:]
        
        # Create context that balances expert preference with watch uniqueness
        # Weighted blend of expert preference and watch text + full watch CLIP
        blended_text = 0.8 * expert_text + 0.2 * watch_text  # Mostly expert preference
        
        # Create NEW array instead of reusing buffer to avoid reference issues
        combined_context = np.concatenate([blended_text, watch_clip])
        
        # Normalize combined context
        norm = np.linalg.norm(combined_context)
        if norm > 0:
            combined_context = combined_context / norm
        
        return combined_context
    
    def get_ucb_score(self, watch_id: int, watch_embedding: np.ndarray) -> float:
        """Get UCB score for a single watch using its individual arm."""
        if self.centroid is None:
            return 0.0
            
        # Create combined context (same as used in training)
        combined_context = self._combine_context(self.centroid, watch_embedding)
        
        # Get or create arm for this specific watch
        if watch_id not in self.arms:
            arm = OptimizedArm(self.dim)
            arm.features = watch_embedding
            
            # Initialize arm with watch-specific bias for differentiation
            # Use similarity to expert centroid to give relevant watches higher initial scores
            np.random.seed(watch_id)  # Deterministic but unique per watch
            similarity = np.dot(combined_context, self.centroid) if self.centroid is not None else 0.0
            base_bias = 0.1 * similarity  # Bias based on relevance to expert
            random_bias = np.random.normal(0, 0.05, self.dim)  # Random component for exploration
            arm.b = base_bias * combined_context + random_bias
            
            self.arms[watch_id] = arm
        else:
            arm = self.arms[watch_id]
        
        # Use this watch's specific arm for scoring
        return arm.get_ucb(combined_context, self.alpha)
    
    def batch_get_ucb_scores(self, watch_ids: List[int], embeddings: np.ndarray) -> np.ndarray:
        """Get UCB scores for multiple watches using their individual arms."""
        if self.centroid is None:
            return np.zeros(len(embeddings))
            
        scores = np.zeros(len(embeddings))
        
        for i, (watch_id, embedding) in enumerate(zip(watch_ids, embeddings)):
            # Create combined context for this watch
            combined_context = self._combine_context(self.centroid, embedding)
            
            # Get or create arm for this specific watch
            if watch_id not in self.arms:
                arm = OptimizedArm(self.dim)
                arm.features = embedding
                self.arms[watch_id] = arm
            else:
                arm = self.arms[watch_id]
            
            # Use this watch's specific arm for scoring
            scores[i] = arm.get_ucb(combined_context, self.alpha)
        
        return scores

class OptimizedLinUCBEngine:
    """Optimized Multi-Expert LinUCB engine with performance improvements."""
    def __init__(self, 
                 dim: int = 50,  # Optimal: Lower dimension with PCA works better
                 alpha: float = 0.1,  # Optimal: Lower exploration for better convergence
                 batch_size: int = 5,
                 max_experts: int = 4,  # Optimal: 4 experts for balanced specialization
                 similarity_threshold: float = 0.7,  # Optimal: Lower threshold for broader learning
                 data_dir: Optional[str] = None):
        """Initialize optimized engine."""
        self.dim = dim
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_experts = max_experts
        self.similarity_threshold = similarity_threshold
        
        # Expert management
        self.experts: Dict[int, OptimizedExpertLinUCB] = {}
        self.next_expert_id = 0
        
        # Session management
        self.session_experts: Dict[str, List[int]] = {}
        self.session_liked_watches: Dict[str, List[int]] = {}  # Used by API for liked watches endpoint
        self.session_embeddings: Dict[str, Dict[int, np.ndarray]] = {}
        self.session_shown_watches: Dict[str, Set[int]] = {}
        
        # Watch data
        self.watch_data: Dict[int, Dict[str, Any]] = {}
        self.watch_text_reduced: Dict[int, np.ndarray] = {}
        self.watch_clip_reduced: Dict[int, np.ndarray] = {}
        self.available_watches: Set[int] = set()
        
        # Load data
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data'
        )
        self._load_data()
        
        # Pre-allocate reusable arrays
        self._embedding_buffer = np.zeros(dim)
        self._batch_context_buffer = np.zeros((batch_size, dim))
        
        logger.info(f"✅ Optimized LinUCB engine initialized with {len(self.watch_data)} watches")
    
    def _load_data(self) -> None:
        """Load and preprocess watch data."""
        try:
            # Load watch metadata
            metadata_path = os.path.join(self.data_dir, 'watch_text_metadata.pkl')
            with open(metadata_path, 'rb') as f:
                metadata_list = pickle.load(f)
            
            # Load text embeddings
            text_embeddings_path = os.path.join(self.data_dir, 'watch_text_embeddings.pkl')
            with open(text_embeddings_path, 'rb') as f:
                text_embeddings_array = pickle.load(f)
            
            # Load CLIP embeddings
            clip_embeddings_path = os.path.join(self.data_dir, 'watch_clip_embeddings.pkl')
            try:
                with open(clip_embeddings_path, 'rb') as f:
                    clip_embeddings_array = pickle.load(f)
            except FileNotFoundError:
                clip_embeddings_array = np.zeros((len(metadata_list), 512))
            
            # Initialize PCA for both text and CLIP embeddings
            if not hasattr(self, '_text_pca_reducer'):
                # Collect sample embeddings for PCA fitting
                sample_size = min(1000, len(metadata_list))
                
                # Fit PCA for text embeddings
                text_samples = []
                for idx in range(sample_size):
                    if idx < len(text_embeddings_array):
                        text_samples.append(text_embeddings_array[idx])
                
                if text_samples:
                    # Fit text PCA
                    self._text_scaler = StandardScaler()
                    text_scaled = self._text_scaler.fit_transform(text_samples)
                    self._text_pca_reducer = PCA(n_components=self.dim // 2)
                    self._text_pca_reducer.fit(text_scaled)
                
                # Fit PCA for CLIP embeddings
                clip_samples = []
                for idx in range(sample_size):
                    if idx < len(clip_embeddings_array):
                        clip_samples.append(clip_embeddings_array[idx])
                
                if clip_samples:
                    # Fit CLIP PCA
                    self._clip_scaler = StandardScaler()
                    clip_scaled = self._clip_scaler.fit_transform(clip_samples)
                    self._clip_pca_reducer = PCA(n_components=self.dim // 2)
                    self._clip_pca_reducer.fit(clip_scaled)
            
            # Process all watches
            for idx, watch_dict in enumerate(metadata_list):
                try:
                    watch_id = watch_dict.get('index', idx)
                    
                    # Store watch data
                    self.watch_data[watch_id] = {
                        **watch_dict,
                        'watch_id': watch_id,
                        'index': watch_id
                    }
                    
                    # Get and reduce text embedding with PCA
                    if idx < len(text_embeddings_array):
                        text_emb = text_embeddings_array[idx]
                        text_reduced = self._reduce_text_features(text_emb)
                        self.watch_text_reduced[watch_id] = text_reduced
                    
                    # Get and reduce CLIP embedding with PCA
                    if idx < len(clip_embeddings_array):
                        clip_emb = clip_embeddings_array[idx]
                        clip_reduced = self._reduce_clip_features(clip_emb)
                        self.watch_clip_reduced[watch_id] = clip_reduced
                    
                    self.available_watches.add(watch_id)
                    
                except Exception as e:
                    logger.error(f"Error processing watch {idx}: {e}")
                    continue
            
            logger.info(f"✅ Loaded {len(self.watch_data)} watches with PCA-reduced embeddings")
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            self._create_fallback_data()
    
    def _reduce_text_features(self, embedding: np.ndarray) -> np.ndarray:
        """Reduce text embedding dimensionality with PCA."""
        if embedding is None or len(embedding) == 0:
            return np.zeros(self.dim // 2)
            
        if len(embedding) <= self.dim // 2:
            return np.pad(embedding, (0, self.dim // 2 - len(embedding)))
        
        try:
            if hasattr(self, '_text_pca_reducer'):
                embedding_scaled = self._text_scaler.transform(embedding.reshape(1, -1))
                return self._text_pca_reducer.transform(embedding_scaled).flatten()
            else:
                return embedding[:self.dim // 2]
        except:
            return embedding[:self.dim // 2]
    
    def _reduce_clip_features(self, embedding: np.ndarray) -> np.ndarray:
        """Reduce CLIP embedding dimensionality with PCA."""
        if embedding is None or len(embedding) == 0:
            return np.zeros(self.dim // 2)
            
        if len(embedding) <= self.dim // 2:
            return np.pad(embedding, (0, self.dim // 2 - len(embedding)))
        
        try:
            if hasattr(self, '_clip_pca_reducer'):
                embedding_scaled = self._clip_scaler.transform(embedding.reshape(1, -1))
                return self._clip_pca_reducer.transform(embedding_scaled).flatten()
            else:
                return embedding[:self.dim // 2]
        except:
            return embedding[:self.dim // 2]
    
    def create_session(self, session_id: str) -> None:
        """Initialize a new session with pre-computed embeddings."""
        
        # Pre-compute concatenated embeddings for all watches (no user scaling)
        self.session_embeddings[session_id] = {}
        
        # Process in batches to manage memory
        batch_size = 1000
        watch_ids = list(self.available_watches)
        
        for i in range(0, len(watch_ids), batch_size):
            batch_ids = watch_ids[i:i + batch_size]
            for watch_id in batch_ids:
                text_emb = self.watch_text_reduced[watch_id]
                clip_emb = self.watch_clip_reduced.get(watch_id, np.zeros(self.dim // 2))
                
                # Simply concatenate without user scaling - let LinUCB learn the weights
                combined = np.concatenate([text_emb, clip_emb])
                
                # Normalize the final combined embedding
                combined_norm = np.linalg.norm(combined)
                if combined_norm > 0:
                    combined = combined / combined_norm
                
                # Store normalized combined embedding
                self.session_embeddings[session_id][watch_id] = combined
        
        # Initialize session tracking
        self.session_experts[session_id] = []
        self.session_liked_watches[session_id] = []
        self.session_shown_watches[session_id] = set()  # Initialize empty set for shown watches
        
        logger.info(f"✅ Created session {session_id} with {len(self.session_embeddings[session_id])} pre-computed embeddings")
    
    def get_recommendations(self,
                          session_id: str,
                          exclude_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """Get recommendations using pre-computed embeddings."""
        exclude_ids = exclude_ids or set()
        
        # Ensure session exists
        if session_id not in self.session_embeddings:
            self.create_session(session_id)
        
        # Exclude session-specific shown watches and provided excludes
        session_shown_watches = self.session_shown_watches.setdefault(session_id, set())
        all_excludes = exclude_ids | session_shown_watches
        
        # Get available watches with deduplication by brand+model
        available_watches = self._get_unique_watches(all_excludes)
        
        if not available_watches:
            return []
        
        session_experts = self.session_experts[session_id]
        
        if not session_experts:
            # Random exploration
            selected_watches = np.random.choice(
                available_watches,
                size=min(self.batch_size, len(available_watches)),
                replace=False
            ).tolist()
            
            # Track the watches we're actually showing
            session_shown_watches.update(selected_watches)
            
            logger.info(f"🎲 Session {session_id}: No experts yet, using random exploration ({len(selected_watches)} watches)")
            return [self._format_recommendation(watch_id, 0.5, "Random") for watch_id in selected_watches]
        
        # Get all scores from all experts
        all_scores = []  # List of (watch_id, score, expert_id)
        expert_best_scores = {}  # Track best score per expert
        
        logger.info(f"🤔 Session {session_id}: Querying {len(session_experts)} experts for {len(available_watches)} watches")
        
        # Process available watches in batches
        batch_size = 1000
        for expert_id in session_experts:
            if expert_id not in self.experts:
                continue
                
            expert = self.experts[expert_id]
            expert_scores = []  # Scores for this expert
            
            # Process watches in batches
            for i in range(0, len(available_watches), batch_size):
                batch_ids = available_watches[i:i + batch_size]
                
                # Get pre-computed embeddings for this batch
                batch_embeddings = np.array([
                    self.session_embeddings[session_id][watch_id]
                    for watch_id in batch_ids
                ])
                
                # Get scores for this batch
                batch_scores = expert.batch_get_ucb_scores(batch_ids, batch_embeddings)
                
                # Store scores for this expert
                for watch_id, score in zip(batch_ids, batch_scores):
                    expert_scores.append((watch_id, score, expert_id))
                
                # Add to global scores
                all_scores.extend((watch_id, score, expert_id) 
                                for watch_id, score in zip(batch_ids, batch_scores))
            
            # Track best scores per expert for balanced selection
            if expert_scores:
                expert_scores.sort(key=lambda x: x[1], reverse=True)
                expert_best_scores[expert_id] = expert_scores[:3]  # Top 3 from each expert
        
        # BALANCED EXPERT RECOMMENDATION STRATEGY
        final_recommendations = []
        seen_watches = set()
        
        # Phase 1: Ensure each expert gets at least 1 recommendation (balanced representation)
        for expert_id in session_experts:
            if expert_id in expert_best_scores and len(final_recommendations) < self.batch_size:
                for watch_id, score, exp_id in expert_best_scores[expert_id]:
                    if watch_id not in seen_watches:
                        seen_watches.add(watch_id)
                        session_expert_number = session_experts.index(expert_id) + 1
                        final_recommendations.append(
                            self._format_recommendation(watch_id, score, f"expert_{session_expert_number}")
                        )
                        break  # Only take 1 per expert in phase 1
        
        # Phase 2: Fill remaining slots with best overall scores
        if len(final_recommendations) < self.batch_size:
            # Sort all remaining scores
            all_scores.sort(key=lambda x: x[1], reverse=True)
            
            for watch_id, score, expert_id in all_scores:
                if watch_id not in seen_watches and len(final_recommendations) < self.batch_size:
                    seen_watches.add(watch_id)
                    session_expert_number = session_experts.index(expert_id) + 1
                    final_recommendations.append(
                        self._format_recommendation(watch_id, score, f"expert_{session_expert_number}")
                    )
        
        # Debug: Log expert ID mapping for session
        if session_experts:
            expert_mapping = {expert_id: i+1 for i, expert_id in enumerate(session_experts)}
            logger.debug(f"Session {session_id} expert mapping: {expert_mapping}")
            
            # Log expert contributions
            expert_contributions = {}
            for rec in final_recommendations:
                alg = rec.get('algorithm', 'unknown')
                expert_contributions[alg] = expert_contributions.get(alg, 0) + 1
            logger.info(f"📊 Expert contributions: {expert_contributions}")
        
        # Track all watches we're actually showing
        shown_watch_ids = [rec.get('watch_id') for rec in final_recommendations if rec.get('watch_id') is not None]
        session_shown_watches.update(shown_watch_ids)
        
        return final_recommendations
    
    def update(self, session_id: str, watch_id: int, reward: float) -> None:
        """Update system with feedback."""
        # Ensure session exists
        if session_id not in self.session_embeddings:
            self.create_session(session_id)
        
        # Get pre-computed embedding
        if watch_id not in self.session_embeddings[session_id]:
            logger.warning(f"Watch {watch_id} not found in session embeddings")
            return
        
        watch_embedding = self.session_embeddings[session_id][watch_id]
        
        # Track likes
        if reward > 0:
            if watch_id not in self.session_liked_watches[session_id]:
                self.session_liked_watches[session_id].append(watch_id)
        
        session_experts = self.session_experts[session_id]
        
        # Create first expert on first like
        if len(session_experts) == 0:
            if reward > 0:
                expert_id = self._create_new_expert()
                self.session_experts[session_id].append(expert_id)
                expert = self.experts[expert_id]
                expert.add_liked_watch(watch_id, watch_embedding)
                expert.update(watch_id, reward, watch_embedding)
                logger.info(f"👤 Session {session_id}: First expert {expert_id} initialized with watch {watch_id}")
            return
        
        # Update existing experts or create new one
        if reward > 0:
            # Find best existing expert
            best_expert_id = None
            best_similarity = -1.0
            
            for expert_id in session_experts:
                if expert_id in self.experts:
                    expert = self.experts[expert_id]
                    if expert.centroid is not None:
                        similarity = np.dot(watch_embedding, expert.centroid)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_expert_id = expert_id
            
            if best_similarity >= self.similarity_threshold:
                # Add to existing expert
                expert = self.experts[best_expert_id]
                expert.add_liked_watch(watch_id, watch_embedding)
                expert.update(watch_id, reward, watch_embedding)
                logger.info(f"✅ Expert {best_expert_id}: Added watch {watch_id} (similarity: {best_similarity:.3f} ≥ {self.similarity_threshold}) | Liked watches: {len(expert.liked_watches)}")
            elif len(session_experts) < self.max_experts:
                # Create new expert
                expert_id = self._create_new_expert()
                self.session_experts[session_id].append(expert_id)
                expert = self.experts[expert_id]
                expert.add_liked_watch(watch_id, watch_embedding)
                expert.update(watch_id, reward, watch_embedding)
                logger.info(f"🆕 Expert {expert_id}: New expert created for watch {watch_id} (similarity: {best_similarity:.3f} < {self.similarity_threshold})")
            else:
                # Add to best expert if at limit
                expert = self.experts[best_expert_id]
                expert.add_liked_watch(watch_id, watch_embedding)
                expert.update(watch_id, reward, watch_embedding)
                logger.info(f"🔄 Expert {best_expert_id}: Added watch {watch_id} (at max experts, best similarity: {best_similarity:.3f}) | Liked watches: {len(expert.liked_watches)}")
        else:
            # Update all experts with negative feedback
            negative_count = 0
            for expert_id in session_experts:
                if expert_id in self.experts:
                    expert = self.experts[expert_id]
                    expert.update(watch_id, reward, watch_embedding)
                    negative_count += 1
            logger.info(f"👎 Negative feedback: Updated {negative_count} experts with watch {watch_id} (reward: {reward})")
    
    def _get_unique_watches(self, exclude_ids: Set[int]) -> List[int]:
        """Get unique watches by complete signature, excluding specified IDs."""
        seen_signatures = set()
        unique_watches = []
        
        for watch_id in self.available_watches:
            if watch_id in exclude_ids:
                continue
                
            watch = self.watch_data.get(watch_id, {})
            specs = watch.get('specs', {})
            
            # Create comprehensive signature to distinguish true duplicates from variations
            signature_parts = []
            
            # Core identity
            brand = watch.get('brand', '').strip().lower()
            model = watch.get('model', '').strip().lower()
            signature_parts.append(f"brand:{brand}")
            signature_parts.append(f"model:{model}")
            
            # Key differentiating specs (only add if not empty/null)
            for key in ['dial_color', 'case_material', 'movement', 'case_size', 'strap_material']:
                value = specs.get(key, '').strip().lower()
                if value and value != '-' and value != 'null':
                    signature_parts.append(f"{key}:{value}")
            
            # Product URL as final differentiator (same watch shouldn't have different URLs)
            product_url = watch.get('product_url', '').strip().lower()
            if product_url:
                signature_parts.append(f"url:{product_url}")
            
            # Create complete signature
            full_signature = "|".join(signature_parts)
            
            if full_signature not in seen_signatures:
                seen_signatures.add(full_signature)
                unique_watches.append(watch_id)
        
        return unique_watches
    
    def _create_new_expert(self) -> int:
        """Create a new expert and return its ID."""
        expert_id = self.next_expert_id
        self.next_expert_id += 1
        self.experts[expert_id] = OptimizedExpertLinUCB(expert_id, self.dim, self.alpha)
        logger.info(f"🧠 Created Expert {expert_id} | Total experts: {len(self.experts)}/{self.max_experts}")
        return expert_id
    
    def _format_recommendation(self, watch_id: int, confidence: float, algorithm: str) -> Dict[str, Any]:
        """Format a recommendation as a dictionary."""
        if watch_id in self.watch_data:
            watch = self.watch_data[watch_id].copy()
            watch.update({
                'score': confidence,
                'confidence': confidence,
                'algorithm': algorithm,
                'watch_id': watch_id
            })
            return watch
        else:
            return {
                'watch_id': watch_id,
                'confidence': confidence,
                'score': confidence,
                'algorithm': algorithm,
                'error': 'Watch not found'
            }
    
    def _create_fallback_data(self) -> None:
        """Create minimal fallback data for testing."""
        logger.warning("Creating fallback data for testing")
        
        for i in range(20):
            watch_id = i
            self.watch_data[watch_id] = {
                'watch_id': watch_id,
                'index': watch_id,
                'brand': f'TestBrand{i%5}',
                'model': f'Model{i}',
                'price': 1000 + i * 200,
                'description': f'Test watch {i}'
            }
            
            # Create test embeddings
            text_reduced = np.random.randn(self.dim // 2)
            clip_reduced = np.random.randn(self.dim // 2)
            
            self.watch_text_reduced[watch_id] = text_reduced
            self.watch_clip_reduced[watch_id] = clip_reduced
            self.available_watches.add(watch_id)

    def shutdown(self) -> None:
        """Clean up resources and prepare for shutdown."""
        logger.info("🛑 Shutting down OptimizedLinUCBEngine...")
        # Clear memory
        self.experts.clear()
        self.watch_data.clear()
        self.available_watches.clear()
        self.session_embeddings.clear()
        logger.info("✅ OptimizedLinUCBEngine shutdown complete")

    def get_expert_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the experts."""
        stats = {
            'num_experts': len(self.experts),
            'max_experts': self.max_experts,
            'experts': []
        }
        
        total_pulls = 0
        total_positive_feedback = 0
        
        for expert_id, expert in self.experts.items():
            expert_stats = {
                'expert_id': expert_id,
                'num_liked_watches': len(expert.liked_watches),
                'arms': {
                    'total': len(expert.arms),
                    'most_pulled': max((arm.total_pulls for arm in expert.arms.values()), default=0),
                    'most_positive': max((arm.positive_feedback for arm in expert.arms.values()), default=0)
                }
            }
            
            # Sum up pulls and feedback across all arms
            expert_total_pulls = sum(arm.total_pulls for arm in expert.arms.values())
            expert_total_positive = sum(arm.positive_feedback for arm in expert.arms.values())
            
            expert_stats['total_pulls'] = expert_total_pulls
            expert_stats['total_positive_feedback'] = expert_total_positive
            
            total_pulls += expert_total_pulls
            total_positive_feedback += expert_total_positive
            
            stats['experts'].append(expert_stats)
        
        # Add global stats
        stats['global'] = {
            'total_pulls': total_pulls,
            'total_positive_feedback': total_positive_feedback,
            'feedback_rate': total_positive_feedback / total_pulls if total_pulls > 0 else 0
        }
        
        return stats

# For backward compatibility
MultiExpertLinUCBEngine = OptimizedLinUCBEngine
LinUCBEngine = OptimizedLinUCBEngine 