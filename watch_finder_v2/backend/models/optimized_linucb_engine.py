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
        
        # Create an arm for this watch if it doesn't exist
        if watch_id not in self.arms:
            arm = OptimizedArm(self.dim)
            arm.features = embedding
            self.arms[watch_id] = arm
    
    def update(self, watch_id: int, reward: float, context: np.ndarray) -> None:
        """Update expert with feedback."""
        # Create arm if it doesn't exist
        if watch_id not in self.arms:
            arm = OptimizedArm(self.dim)
            arm.features = context
            self.arms[watch_id] = arm
        
        # Update arm with feedback
        arm = self.arms[watch_id]
        arm.update(context, reward)
        
        # Update liked watches list and centroid if positive feedback
        if reward > 0 and watch_id not in self.liked_watches:
            self.add_liked_watch(watch_id, context)
    
    def _combine_context(self, centroid: np.ndarray, watch_embedding: np.ndarray) -> np.ndarray:
        """Combine expert centroid with watch embedding using pre-allocated buffer."""
        half_dim = self.dim // 2
        self._context_buffer[:half_dim] = centroid[:half_dim]
        self._context_buffer[half_dim:] = watch_embedding[half_dim:]
        return self._context_buffer
    
    def batch_get_ucb_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """Get UCB scores for multiple watches at once."""
        if self.centroid is None:
            return np.zeros(len(embeddings))
            
        # Create combined contexts for all embeddings at once
        half_dim = self.dim // 2
        batch_size = len(embeddings)
        
        # Pre-allocate combined contexts array
        combined_contexts = np.zeros((batch_size, self.dim))
        combined_contexts[:, :half_dim] = self.centroid[:half_dim]
        combined_contexts[:, half_dim:] = embeddings[:, half_dim:]
        
        # Normalize combined contexts
        norms = np.linalg.norm(combined_contexts, axis=1, keepdims=True)
        norms[norms == 0] = 1
        combined_contexts /= norms
        
        # Get or create a default arm for scoring
        if not self.arms:
            default_arm = OptimizedArm(self.dim)
            default_arm.features = np.zeros(self.dim)
            return default_arm.batch_ucb(combined_contexts, self.alpha)
        
        # Use the most updated arm for scoring
        latest_arm = max(self.arms.values(), key=lambda a: a.total_pulls)
        return latest_arm.batch_ucb(combined_contexts, self.alpha)

class OptimizedLinUCBEngine:
    """Optimized Multi-Expert LinUCB engine with performance improvements."""
    def __init__(self, 
                 dim: int = 100,
                 alpha: float = 0.15,
                 batch_size: int = 5,
                 max_experts: int = 6,
                 similarity_threshold: float = 0.45,
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
        self.expert_centroids: Dict[int, np.ndarray] = {}
        
        # Session management
        self.session_experts: Dict[str, List[int]] = {}
        self.session_liked_watches: Dict[str, List[int]] = {}
        self.session_embeddings: Dict[str, Dict[int, np.ndarray]] = {}
        self.session_embedding_weights: Dict[str, Tuple[float, float]] = {}
        
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
            
            # Initialize PCA for dimensionality reduction
            if not hasattr(self, '_pca_reducer'):
                # Collect sample embeddings for PCA fitting
                sample_size = min(1000, len(metadata_list))
                sample_embeddings = []
                for idx in range(sample_size):
                    if idx < len(text_embeddings_array):
                        sample_embeddings.append(text_embeddings_array[idx])
                
                if sample_embeddings:
                    # Fit PCA
                    self._scaler = StandardScaler()
                    embeddings_scaled = self._scaler.fit_transform(sample_embeddings)
                    self._pca_reducer = PCA(n_components=self.dim // 2)
                    self._pca_reducer.fit(embeddings_scaled)
            
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
                    
                    # Get and reduce text embedding
                    if idx < len(text_embeddings_array):
                        text_emb = text_embeddings_array[idx]
                        text_reduced = self._reduce_features(text_emb)
                        self.watch_text_reduced[watch_id] = text_reduced
                    
                    # Get and reduce CLIP embedding
                    if idx < len(clip_embeddings_array):
                        clip_emb = clip_embeddings_array[idx]
                        clip_reduced = self._reduce_clip_embedding(clip_emb)
                        self.watch_clip_reduced[watch_id] = clip_reduced
                    
                    self.available_watches.add(watch_id)
                    
                except Exception as e:
                    logger.error(f"Error processing watch {idx}: {e}")
                    continue
            
            logger.info(f"✅ Loaded {len(self.watch_data)} watches with reduced embeddings")
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            self._create_fallback_data()
    
    def _reduce_features(self, embedding: np.ndarray) -> np.ndarray:
        """Reduce embedding dimensionality efficiently."""
        if embedding is None or len(embedding) == 0:
            return np.zeros(self.dim // 2)
            
        if len(embedding) <= self.dim // 2:
            return np.pad(embedding, (0, self.dim // 2 - len(embedding)))
        
        try:
            if hasattr(self, '_pca_reducer'):
                embedding_scaled = self._scaler.transform(embedding.reshape(1, -1))
                return self._pca_reducer.transform(embedding_scaled).flatten()
            else:
                return embedding[:self.dim // 2]
        except:
            return embedding[:self.dim // 2]
    
    def _reduce_clip_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Reduce CLIP embedding efficiently."""
        if len(embedding) == self.dim // 2:
            return embedding
        elif len(embedding) > self.dim // 2:
            return embedding[:self.dim // 2]
        else:
            return np.pad(embedding, (0, self.dim // 2 - len(embedding)))
    
    def create_session(self, session_id: str, clip_weight: float = 0.5, text_weight: float = 0.5) -> None:
        """Initialize a new session with pre-computed embeddings."""
        # Normalize weights
        total = clip_weight + text_weight
        if total > 0:
            clip_weight /= total
            text_weight /= total
        else:
            clip_weight = text_weight = 0.5
        
        # Store session weights
        self.session_embedding_weights[session_id] = (clip_weight, text_weight)
        
        # Pre-compute weighted embeddings for all watches
        self.session_embeddings[session_id] = {}
        
        # Process in batches to manage memory
        batch_size = 1000
        watch_ids = list(self.available_watches)
        
        for i in range(0, len(watch_ids), batch_size):
            batch_ids = watch_ids[i:i + batch_size]
            for watch_id in batch_ids:
                text_emb = self.watch_text_reduced[watch_id]
                clip_emb = self.watch_clip_reduced.get(watch_id, np.zeros(self.dim // 2))
                
                # Create weighted combination
                text_norm = text_emb / (np.linalg.norm(text_emb) + 1e-8)
                clip_norm = clip_emb / (np.linalg.norm(clip_emb) + 1e-8)
                
                combined = np.concatenate([
                    text_weight * text_norm,
                    clip_weight * clip_norm
                ])
                
                # Store normalized combined embedding
                self.session_embeddings[session_id][watch_id] = combined
        
        # Initialize session tracking
        self.session_experts[session_id] = []
        self.session_liked_watches[session_id] = []
        
        logger.info(f"✅ Created session {session_id} with {len(self.session_embeddings[session_id])} pre-computed embeddings")
    
    def get_recommendations(self,
                          session_id: str,
                          context: np.ndarray,
                          exclude_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """Get recommendations using pre-computed embeddings."""
        exclude_ids = exclude_ids or set()
        
        # Ensure session exists
        if session_id not in self.session_embeddings:
            clip_weight = context[0] if len(context) > 0 else 0.5
            text_weight = context[1] if len(context) > 1 else 0.5
            self.create_session(session_id, clip_weight, text_weight)
        
        # Get available watches
        available_watches = [
            watch_id for watch_id in self.available_watches 
            if watch_id not in exclude_ids
        ]
        
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
            return [self._format_recommendation(watch_id, 0.5, "Random") for watch_id in selected_watches]
        
        # Get all scores from all experts
        all_scores = []  # List of (watch_id, score, expert_id)
        
        # Process available watches in batches
        batch_size = 1000
        for expert_id in session_experts:
            if expert_id not in self.experts:
                continue
                
            expert = self.experts[expert_id]
            
            # Process watches in batches
            for i in range(0, len(available_watches), batch_size):
                batch_ids = available_watches[i:i + batch_size]
                
                # Get pre-computed embeddings for this batch
                batch_embeddings = np.array([
                    self.session_embeddings[session_id][watch_id]
                    for watch_id in batch_ids
                ])
                
                # Get scores for this batch
                batch_scores = expert.batch_get_ucb_scores(batch_embeddings)
                
                # Add scores with expert ID
                all_scores.extend((watch_id, score, expert_id) 
                                for watch_id, score in zip(batch_ids, batch_scores))
        
        # Sort all scores from all experts
        all_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Debug: Log expert ID mapping for session
        if session_experts:
            expert_mapping = {expert_id: i+1 for i, expert_id in enumerate(session_experts)}
            logger.debug(f"Session {session_id} expert mapping: {expert_mapping}")
        
        # Take top scores while avoiding duplicates
        seen_watches = set()
        final_recommendations = []
        
        for watch_id, score, expert_id in all_scores:
            if watch_id not in seen_watches and len(final_recommendations) < self.batch_size:
                seen_watches.add(watch_id)
                # Convert global expert ID to session-specific expert number (1-6)
                session_expert_number = session_experts.index(expert_id) + 1
                final_recommendations.append(
                    self._format_recommendation(watch_id, score, f"expert_{session_expert_number}")
                )
        
        # Fill remaining slots with random recommendations if needed
        if len(final_recommendations) < self.batch_size:
            remaining_watches = [w for w in available_watches if w not in seen_watches]
            if remaining_watches:
                additional = np.random.choice(
                    remaining_watches,
                    size=min(self.batch_size - len(final_recommendations), len(remaining_watches)),
                    replace=False
                ).tolist()
                
                for watch_id in additional:
                    final_recommendations.append(
                        self._format_recommendation(watch_id, 0.4, "random_fill")
                    )
        
        return final_recommendations
    
    def update(self, session_id: str, watch_id: int, reward: float, context: np.ndarray) -> None:
        """Update system with feedback."""
        # Ensure session exists
        if session_id not in self.session_embeddings:
            clip_weight = context[0] if len(context) > 0 else 0.5
            text_weight = context[1] if len(context) > 1 else 0.5
            self.create_session(session_id, clip_weight, text_weight)
        
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
            elif len(session_experts) < self.max_experts:
                # Create new expert
                expert_id = self._create_new_expert()
                self.session_experts[session_id].append(expert_id)
                expert = self.experts[expert_id]
                expert.add_liked_watch(watch_id, watch_embedding)
                expert.update(watch_id, reward, watch_embedding)
            else:
                # Add to best expert if at limit
                expert = self.experts[best_expert_id]
                expert.add_liked_watch(watch_id, watch_embedding)
                expert.update(watch_id, reward, watch_embedding)
        else:
            # Update all experts with negative feedback
            for expert_id in session_experts:
                if expert_id in self.experts:
                    expert = self.experts[expert_id]
                    expert.update(watch_id, reward, watch_embedding)
    
    def _create_new_expert(self) -> int:
        """Create a new expert and return its ID."""
        expert_id = self.next_expert_id
        self.next_expert_id += 1
        self.experts[expert_id] = OptimizedExpertLinUCB(expert_id, self.dim, self.alpha)
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
        self.expert_centroids.clear()
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