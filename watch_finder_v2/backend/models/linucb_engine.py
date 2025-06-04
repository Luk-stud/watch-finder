"""
Multi-Expert LinUCB Recommendation Engine
========================================

Implements multiple specialized LinUCB experts based on embedding clustering:
- Clusters watches by embedding similarity (luxury, sports, dress, etc.)
- Each expert specializes in its watch cluster
- Routes feedback and recommendations to appropriate experts
- Handles diverse user preferences better than single expert
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

class Arm:
    """Represents a single arm (watch) in the LinUCB algorithm."""
    def __init__(self, dim: int):
        self.A: np.ndarray = np.identity(dim)  # d x d matrix
        self.b: np.ndarray = np.zeros(dim)     # d-dimensional vector
        self.theta: Optional[np.ndarray] = None # Coefficient vector
        
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
        
        self.total_pulls += 1
        if reward > 0:
            self.positive_feedback += 1
        self.last_pull = datetime.now()
    
    def get_theta(self) -> np.ndarray:
        """Get coefficient vector, computing if necessary."""
        if self.theta is None:
            try:
                self.theta = np.linalg.solve(self.A, self.b)
            except np.linalg.LinAlgError:
                # Handle singular matrix with regularization
                regularized_A = self.A + 0.01 * np.identity(self.A.shape[0])
                self.theta = np.linalg.solve(regularized_A, self.b)
        return self.theta
    
    def get_ucb(self, context: np.ndarray, alpha: float) -> float:
        """Calculate UCB score for this arm with numerical stability."""
        theta = self.get_theta()
        mean = np.dot(theta, context)
        
        try:
            # Add small regularization for numerical stability
            A_reg = self.A + 1e-6 * np.identity(self.A.shape[0])
            A_inv_context = np.linalg.solve(A_reg, context)
            confidence_width = alpha * np.sqrt(np.dot(context.T, A_inv_context))
            
            # Ensure confidence width is finite and positive
            if not np.isfinite(confidence_width) or confidence_width < 0:
                confidence_width = alpha * 0.1  # Safe fallback
                
        except (np.linalg.LinAlgError, ValueError) as e:
            # Fallback for numerical issues
            confidence_width = alpha * np.sqrt(np.dot(context, context) / max(1, self.total_pulls))
            
        return mean + confidence_width

class ExpertLinUCB:
    """Single LinUCB expert that learns a preference centroid and can score any watch."""
    def __init__(self, expert_id: int, dim: int, alpha: float):
        self.expert_id = expert_id
        self.dim = dim
        self.alpha = alpha
        self.arms: Dict[int, Arm] = {}  # Only tracks watches this expert has received feedback on
        self.centroid: Optional[np.ndarray] = None  # Learned preference centroid
        self.liked_watches: List[int] = []  # Watches that define this expert's preferences
        
    def add_liked_watch(self, watch_id: int, embedding: np.ndarray):
        """Add a liked watch to this expert's learned preferences."""
        self.liked_watches.append(watch_id)
        
        # Update centroid
        if self.centroid is None:
            self.centroid = embedding.copy()
        else:
            # Running average of liked watch embeddings
            alpha = 1.0 / len(self.liked_watches)  # Give equal weight to all likes
            self.centroid = (1 - alpha) * self.centroid + alpha * embedding
        
        # Create an arm for this watch if it doesn't exist
        if watch_id not in self.arms:
            arm = Arm(self.dim)
            arm.features = embedding
            self.arms[watch_id] = arm
        
    def get_recommendations(self, all_watches: Dict[int, np.ndarray], exclude_ids: Set[int], num_recs: int) -> List[Tuple[int, float]]:
        """Score ALL watches against this expert's learned preferences."""
        if self.centroid is None:
            return []  # No preferences learned yet
            
        scores: List[Tuple[int, float]] = []
        
        for watch_id, embedding in all_watches.items():
            if watch_id in exclude_ids:
                continue
                
            # Get or create arm for this watch
            if watch_id not in self.arms:
                arm = Arm(self.dim)
                arm.features = embedding
                self.arms[watch_id] = arm
            else:
                arm = self.arms[watch_id]
            
            # Combine expert's centroid with watch embedding as context
            combined_context = self._combine_context(self.centroid, embedding)
            
            # Calculate UCB score
            ucb = arm.get_ucb(combined_context, self.alpha)
            scores.append((watch_id, ucb))
        
        # Sort by UCB score and return top recommendations
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:num_recs]
    
    def update(self, watch_id: int, reward: float, embedding: np.ndarray) -> bool:
        """Update this expert with feedback."""
        # Get or create arm for this watch
        if watch_id not in self.arms:
            arm = Arm(self.dim)
            arm.features = embedding
            self.arms[watch_id] = arm
        else:
            arm = self.arms[watch_id]
        
        # Combine expert's centroid with watch embedding as context
        if self.centroid is not None:
            combined_context = self._combine_context(self.centroid, embedding)
        else:
            # If no centroid yet, use the embedding itself
            combined_context = embedding[:self.dim] if len(embedding) > self.dim else np.pad(embedding, (0, max(0, self.dim - len(embedding))))
        
        # Update arm with feedback
        arm.update(combined_context, reward)
        
        # If this is a like, add to our learned preferences
        if reward > 0:
            self.add_liked_watch(watch_id, embedding)
        
        return True
    
    def _combine_context(self, centroid: np.ndarray, watch_embedding: np.ndarray) -> np.ndarray:
        """Combine expert centroid with watch embedding as context for LinUCB."""
        # Create fixed-size context
        combined_context = np.zeros(self.dim)
        
        # Split dimension evenly between centroid and embedding
        half_dim = self.dim // 2
        
        # Fill first half with expert centroid (truncated/padded as needed)
        centroid_size = min(len(centroid), half_dim)
        combined_context[:centroid_size] = centroid[:centroid_size]
        
        # Fill second half with watch embedding (truncated/padded as needed)  
        embedding_size = min(len(watch_embedding), self.dim - half_dim)
        combined_context[half_dim:half_dim + embedding_size] = watch_embedding[:embedding_size]
        
        # Normalize to prevent magnitude issues
        norm = np.linalg.norm(combined_context)
        if norm > 0:
            combined_context = combined_context / norm
            
        return combined_context

class DynamicMultiExpertLinUCBEngine:
    def __init__(self, 
                 dim: int = 200,  # INCREASED: Better feature representation for LinUCB
                 alpha: float = 0.15,  
                 batch_size: int = 5,
                 num_experts: int = 0,  # Always start with 0 - create dynamically
                 max_experts: int = 6,  
                 similarity_threshold: float = 0.85,  # High threshold for distinct preferences
                 min_expert_size: int = 1,  # Only need 1 like to create expert
                 unassigned_ratio: float = 1.0,  
                 max_exploration_rounds: int = 50,  
                 pre_cluster_method: str = 'dynamic',  # Changed to dynamic
                 initialization_strategy: str = 'clean_dynamic',  # NEW: Clean approach
                 data_dir: Optional[str] = None,
                 min_likes_for_first_expert: int = 1,  # CHANGED: First like = first expert
                 min_likes_for_new_expert: int = 1,   # CHANGED: One like per expert
                 like_clustering_threshold: float = 0.85):  # High threshold for distinct clusters
        """Initialize Like-Driven Dynamic Multi-Expert LinUCB engine."""
        self.dim = dim
        self.alpha = alpha
        self.batch_size = batch_size
        self.initial_experts = num_experts
        self.max_experts = max_experts
        self.similarity_threshold = similarity_threshold
        self.min_expert_size = min_expert_size
        self.unassigned_ratio = unassigned_ratio
        self.max_exploration_rounds = max_exploration_rounds
        self.pre_cluster_method = pre_cluster_method
        self.initialization_strategy = initialization_strategy
        
        # Like-driven clustering parameters
        self.min_likes_for_first_expert = min_likes_for_first_expert
        self.min_likes_for_new_expert = min_likes_for_new_expert
        self.like_clustering_threshold = like_clustering_threshold
        
        # Dynamic expert state
        self.experts: Dict[int, ExpertLinUCB] = {}
        self.next_expert_id = 0
        self.expert_centroids: Dict[int, np.ndarray] = {}  # Expert ID -> centroid embedding
        self.unassigned_watches: Set[int] = set()  # Watches not yet assigned to experts
        
        # Clean Dynamic Clustering state - SIMPLIFIED
        self.session_liked_watches: Dict[str, List[int]] = {}  # session_id -> [liked_watch_ids]
        self.session_experts: Dict[str, List[int]] = {}  # session_id -> [expert_ids] 
        self.session_interaction_counts: Dict[str, int] = {}  # session_id -> interaction count
        
        # Watch data
        self.watch_data: Dict[int, Dict[str, Any]] = {}
        self.watch_embeddings: Dict[int, np.ndarray] = {}  # watch_id -> reduced embedding (dim size)
        
        # Load data and initialize
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data'
        )
        self._load_data()
        
        # All watches start available for selection
        self.available_watches: Set[int] = set(self.watch_data.keys())
        
        logger.info("🎯 CLEAN DYNAMIC CLUSTERING INITIALIZATION:")
        logger.info(f"   📊 Total watches: {len(self.watch_data)}")
        logger.info(f"   🔍 Random exploration until first like")
        logger.info(f"   💝 First like creates first expert")
        logger.info(f"   🎯 Similarity threshold: {self.similarity_threshold}")
        logger.info(f"   📈 Max experts per session: {self.max_experts}")
        logger.info("🚀 CLEAN APPROACH ACTIVATED!")
    
    def _load_data(self) -> None:
        """Load watch data and both text and CLIP embeddings."""
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
                logger.info(f"✅ Loaded CLIP embeddings shape {clip_embeddings_array.shape}")
            except FileNotFoundError:
                logger.warning("⚠️  CLIP embeddings not found, using zero embeddings")
                clip_embeddings_array = np.zeros((len(metadata_list), 512))
            
            logger.info(f"✅ Loaded {len(metadata_list)} watches, text embeddings shape {text_embeddings_array.shape}, CLIP embeddings shape {clip_embeddings_array.shape}")
            
            # Store raw embeddings for PCA fitting
            self._raw_embeddings = []
            valid_raw_embeddings = []
            for idx, watch_dict in enumerate(metadata_list):
                if 'text_embedding' in watch_dict and isinstance(watch_dict['text_embedding'], np.ndarray):
                    raw_embedding = watch_dict['text_embedding']
                elif idx < len(text_embeddings_array):
                    raw_embedding = text_embeddings_array[idx]
                else:
                    continue
                    
                if len(raw_embedding) > self.dim:  # Only collect embeddings that need reduction
                    valid_raw_embeddings.append(raw_embedding)
            
            # Store raw embeddings for PCA fitting
            if len(valid_raw_embeddings) > 0:
                self._raw_embeddings = valid_raw_embeddings[:min(1000, len(valid_raw_embeddings))]  # Limit for memory
                logger.info(f"📊 Collected {len(self._raw_embeddings)} raw embeddings for PCA fitting")
            
            # Convert metadata list to dictionary and store embeddings
            for idx, watch_dict in enumerate(metadata_list):
                try:
                    watch_id = watch_dict.get('index', idx)
                    
                    # Enhanced watch data
                    enhanced_watch = {
                        **watch_dict,
                        'watch_id': watch_id,
                        'index': watch_id,
                        'algorithm': 'dynamic_multi_expert_linucb',
                        'confidence': 0.5,
                        'score': 0.0
                    }
                    
                    self.watch_data[watch_id] = enhanced_watch
                    
                    # Store text embedding with proper dimension reduction
                    if 'text_embedding' in watch_dict and isinstance(watch_dict['text_embedding'], np.ndarray):
                        raw_embedding = watch_dict['text_embedding']
                        self.watch_embeddings[watch_id] = self._reduce_features(raw_embedding, target_dim=self.dim)
                    elif watch_id < len(text_embeddings_array):
                        raw_embedding = text_embeddings_array[watch_id]
                        self.watch_embeddings[watch_id] = self._reduce_features(raw_embedding, target_dim=self.dim)
                    
                    # Store CLIP embedding separately (keep original size for weighted combination)
                    if watch_id < len(clip_embeddings_array):
                        if not hasattr(self, 'watch_clip_embeddings'):
                            self.watch_clip_embeddings = {}
                        self.watch_clip_embeddings[watch_id] = clip_embeddings_array[watch_id]
                    
                    # All watches start unassigned
                    self.unassigned_watches.add(watch_id)
                    
                except Exception as e:
                    logger.error(f"Error processing watch {idx}: {e}")
                    continue
            
            logger.info(f"✅ Processed {len(self.watch_data)} watches with text and CLIP embeddings")
            logger.info(f"📊 Text embeddings: {len(self.watch_embeddings)} watches")
            logger.info(f"🖼️ CLIP embeddings: {len(getattr(self, 'watch_clip_embeddings', {})) if hasattr(self, 'watch_clip_embeddings') else 0} watches")
            
            logger.info("✅ LinUCB engine initialized successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            self._create_fallback_data()
    
    def _create_new_expert(self) -> int:
        """Create a new expert and return its ID."""
        expert_id = self.next_expert_id
        self.next_expert_id += 1
        
        expert = ExpertLinUCB(expert_id, self.dim, self.alpha)
        self.experts[expert_id] = expert
        
        return expert_id
    
    def _add_watch_to_expert(self, expert_id: int, watch_id: int) -> None:
        """Add a watch to an expert and update the expert's centroid."""
        if expert_id not in self.experts:
            return
        
        expert = self.experts[expert_id]
        
        # Get watch embedding and features
        if watch_id in self.watch_embeddings:
            embedding = self.watch_embeddings[watch_id]
            expert.add_liked_watch(watch_id, embedding)
            
            # Update expert centroid
            self._update_expert_centroid(expert_id)
    
    def _update_expert_centroid(self, expert_id: int) -> None:
        """Update the centroid embedding for an expert."""
        expert = self.experts[expert_id]
        
        # Collect all embeddings for this expert
        embeddings = []
        for watch_id in expert.liked_watches:
            if watch_id in self.watch_embeddings:
                embeddings.append(self.watch_embeddings[watch_id])
        
        if embeddings:
            # Calculate centroid as mean of all embeddings
            centroid = np.mean(embeddings, axis=0)
            self.expert_centroids[expert_id] = centroid
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors - FAST VERSION."""
        try:
            # Use numpy's optimized operations
            dot_product = np.dot(a, b)
            norm_product = np.linalg.norm(a) * np.linalg.norm(b)
            
            if norm_product == 0:
                return 0.0
            
            return dot_product / norm_product
        except:
            return 0.0
    
    def _find_best_expert_for_watch(self, watch_id: int) -> Tuple[int, float]:
        """Find the best expert for a watch based on similarity with full embeddings."""
        if watch_id not in self.watch_embeddings or len(self.expert_centroids) == 0:
            # Return first available expert
            return list(self.experts.keys())[0] if self.experts else (0, 0.0)
        
        watch_embedding = self.watch_embeddings[watch_id]
        
        # Use full embeddings for accurate similarity calculation
        best_expert_id = None
        best_similarity = -1.0
        
        for expert_id, centroid in self.expert_centroids.items():
            if centroid is not None and len(centroid) > 0:
                try:
                    # Use full embeddings for accurate similarity
                    similarity = self._cosine_similarity(watch_embedding, centroid)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_expert_id = expert_id
                except Exception as e:
                    logger.warning(f"Similarity calculation failed for expert {expert_id}: {e}")
                    continue
        
        # Fallback to first expert if no good match
        if best_expert_id is None and self.experts:
            best_expert_id = list(self.experts.keys())[0]
            best_similarity = 0.0
        
        return best_expert_id, best_similarity
    
    def _should_create_new_expert(self, watch_id: int, reward: float) -> bool:
        """Determine if we should create a new expert for this watch."""
        # Only create new expert for liked watches
        if reward <= 0:
            return False
        
        # Check if we've hit the expert limit
        if len(self.experts) >= self.max_experts:
            return False
        
        # Find best similarity to existing experts
        _, best_similarity = self._find_best_expert_for_watch(watch_id)
        
        # Create new expert if similarity is below threshold
        return best_similarity < self.similarity_threshold
    
    def get_recommendations(self,
                          session_id: str,
                          context: np.ndarray,
                          exclude_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """Get recommendations using Clean Dynamic Clustering approach with similarity weighting."""
        exclude_ids = exclude_ids or set()
        
        # Convert exclude_ids to integers and ensure they're in a set
        if isinstance(exclude_ids, list):
            exclude_ids = set(int(x) for x in exclude_ids if x is not None)
        elif exclude_ids is None:
            exclude_ids = set()
        
        # Handle context vectors of different sizes
        if len(context) >= 2:
            clip_weight = context[0]
            text_weight = context[1]
        else:
            # Fallback for empty or too-small context
            clip_weight = text_weight = 0.5
        
        # Validate and normalize weights
        clip_weight = max(0.0, min(1.0, clip_weight))  # Bound between 0 and 1
        text_weight = max(0.0, min(1.0, text_weight))  # Bound between 0 and 1
        
        total_weight = clip_weight + text_weight
        if total_weight > 0:
            clip_weight = clip_weight / total_weight
            text_weight = text_weight / total_weight
        else:
            clip_weight = text_weight = 0.5
        
        # Get available watches (excluding already seen ones)
        available_watches = [
            watch_id for watch_id in self.available_watches 
            if watch_id not in exclude_ids
        ]
        
        logger.info(f"🔍 Available watches: {len(available_watches)} (excluding {len(exclude_ids)} seen watches)")
        
        if len(available_watches) < self.batch_size:
            logger.warning(f"⚠️  Only {len(available_watches)} available watches remaining!")
            # If we're running low on watches, we might need to expand the pool
            if len(available_watches) == 0:
                logger.error("❌ No available watches remaining!")
                return []
        
        session_experts = self.session_experts.get(session_id, [])
        session_num = self.session_interaction_counts.get(session_id, 0) + 1
        self.session_interaction_counts[session_id] = session_num  # FIXED: Actually update the counter
        
        if not session_experts:
            # Random exploration
            logger.info(f"🔍 RANDOM EXPLORATION (session {session_num}) - no experts yet")
            logger.info(f"🎯 Similarity weights: Visual={clip_weight:.2f}, Vibe={text_weight:.2f}")
            
            # Random selection from available watches
            selected_watches = np.random.choice(
                available_watches, 
                size=min(self.batch_size, len(available_watches)), 
                replace=False
            ).tolist()
            
            logger.info(f"🔍 Random exploration: {len(selected_watches)} watches")
            return [self._format_recommendation(watch_id, 0.5, "Random Exploration") for watch_id in selected_watches]
        
        # Expert-based recommendations
        logger.info(f"🎯 EXPERT-BASED RECOMMENDATIONS (session {session_num})")
        logger.info(f"   👥 Active experts: {len(session_experts)}")
        logger.info(f"   🎯 Similarity weights: Visual={clip_weight:.2f}, Vibe={text_weight:.2f}")
        
        recommendations = []
        expert_counts = {}
        
        # Get recommendations from each expert
        for expert_id in session_experts:
            if expert_id in self.experts:
                expert = self.experts[expert_id]
                expert_recs = self._get_expert_recommendations(
                    expert, available_watches, clip_weight, text_weight, exclude_ids
                )
                expert_counts[expert_id] = len(expert_recs)
                recommendations.extend(expert_recs)
        
        # Remove duplicates while preserving order
        seen_watch_ids = set()
        unique_recommendations = []
        for rec in recommendations:
            watch_id = rec['watch_id']
            if watch_id not in seen_watch_ids:
                seen_watch_ids.add(watch_id)
                unique_recommendations.append(rec)
        
        # Ensure we have enough recommendations
        if len(unique_recommendations) < self.batch_size and len(available_watches) > len(unique_recommendations):
            # Fill remaining slots with random selections
            remaining_watches = [w for w in available_watches if w not in seen_watch_ids]
            if remaining_watches:
                additional_count = min(
                    self.batch_size - len(unique_recommendations),
                    len(remaining_watches)
                )
                additional_watches = np.random.choice(
                    remaining_watches, 
                    size=additional_count, 
                    replace=False
                ).tolist()
                
                for watch_id in additional_watches:
                    unique_recommendations.append(
                        self._format_recommendation(watch_id, 0.4, "Random Fill")
                    )
        
        # Sort by confidence and take top batch_size
        unique_recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        final_recommendations = unique_recommendations[:self.batch_size]
        
        logger.info(f"🎯 Expert recommendations: {expert_counts}")
        logger.info(f"📋 Final recommendations: {len(final_recommendations)} unique watches")
        
        return final_recommendations
    
    def _get_expert_recommendations(self, expert, available_watches: List[int], clip_weight: float, text_weight: float, exclude_ids: Set[int]) -> List[Dict[str, Any]]:
        """Get recommendations from expert using weighted similarity between CLIP and text embeddings."""
        if expert.centroid is None:
            return []  # No preferences learned yet
            
        scores: List[Tuple[int, float]] = []
        
        for watch_id in available_watches:
            if watch_id in exclude_ids:
                continue
            
            # Get CLIP embedding if available
            clip_embedding = None
            if hasattr(self, 'watch_clip_embeddings') and watch_id in self.watch_clip_embeddings:
                clip_embedding = self.watch_clip_embeddings[watch_id]
            
            # Create weighted combined embedding
            combined_embedding = self._create_weighted_embedding(
                self.watch_embeddings[watch_id], clip_embedding, text_weight, clip_weight
            )
            
            # Get or create arm for this watch
            if watch_id not in expert.arms:
                arm = Arm(self.dim)
                arm.features = combined_embedding
                expert.arms[watch_id] = arm
            else:
                arm = expert.arms[watch_id]
            
            # Combine expert's centroid with weighted watch embedding as context
            combined_context = expert._combine_context(expert.centroid, combined_embedding)
            
            # Calculate UCB score
            ucb = arm.get_ucb(combined_context, expert.alpha)
            scores.append((watch_id, ucb))
        
        # Sort by UCB score and return top recommendations
        scores.sort(key=lambda x: x[1], reverse=True)
        return [self._format_recommendation(watch_id, ucb, f'expert_{expert.expert_id}') for watch_id, ucb in scores[:self.batch_size]]
    
    def _create_weighted_embedding(self, text_embedding: np.ndarray, clip_embedding: Optional[np.ndarray], 
                                 text_weight: float, clip_weight: float) -> np.ndarray:
        """Create a concatenated combination of text and CLIP embeddings."""
        # Ensure text embedding is reduced to target dimension
        if len(text_embedding) != self.dim // 2:  # Half dimension for text
            text_reduced = self._reduce_features(text_embedding, target_dim=self.dim // 2)
        else:
            text_reduced = text_embedding
        
        if clip_embedding is None:
            # If no CLIP embedding, pad with zeros for second half
            clip_reduced = np.zeros(self.dim // 2)
        else:
            # Reduce CLIP embedding to half dimension
            clip_reduced = self._reduce_clip_embedding(clip_embedding, target_dim=self.dim // 2)
        
        # Normalize both halves independently
        text_norm = text_reduced / (np.linalg.norm(text_reduced) + 1e-8)
        clip_norm = clip_reduced / (np.linalg.norm(clip_reduced) + 1e-8)
        
        # Apply user weights to each modality
        text_weighted = text_weight * text_norm
        clip_weighted = clip_weight * clip_norm
        
        # CONCATENATE the two modalities
        concatenated_embedding = np.concatenate([text_weighted, clip_weighted])
        
        # Final normalization of the full vector
        return concatenated_embedding / (np.linalg.norm(concatenated_embedding) + 1e-8)
    
    def _reduce_clip_embedding(self, clip_embedding: np.ndarray, target_dim: int) -> np.ndarray:
        """Reduce CLIP embedding to target dimension."""
        if len(clip_embedding) == target_dim:
            return clip_embedding
        elif len(clip_embedding) > target_dim:
            # Use every nth element for better representation
            indices = np.linspace(0, len(clip_embedding) - 1, target_dim).astype(int)
            return clip_embedding[indices]
        else:
            # Pad with zeros or wrap
            return np.pad(clip_embedding, (0, target_dim - len(clip_embedding)), mode='constant')
    
    def _apply_diversity_filter(self, candidate_recommendations: List[Tuple[int, float]], 
                               used_brands: Set[str], max_count: int) -> List[Tuple[int, float]]:
        """Apply diversity filtering to recommendations to avoid brand/price clustering."""
        if not candidate_recommendations:
            return []
        
        diverse_recs = []
        local_used_brands = used_brands.copy()
        price_ranges_used = set()
        
        # Sort candidates by score but apply diversity constraints
        for watch_id, score in candidate_recommendations:
            if len(diverse_recs) >= max_count:
                break
                
            if watch_id not in self.watch_data:
                continue
            
            watch_data = self.watch_data[watch_id]
            brand = watch_data.get('brand', 'Unknown')
            price = watch_data.get('price', 0)
            
            # Create price range buckets
            if isinstance(price, str):
                price = 0
            price_range = self._get_price_range(price)
            
            # Diversity constraints (relaxed for small expert pools)
            brand_penalty = brand in local_used_brands
            price_penalty = price_range in price_ranges_used
            
            # Allow some brand repetition but prefer diversity
            if len(diverse_recs) < max_count // 2:
                # First half: strict diversity
                if not brand_penalty:
                    diverse_recs.append((watch_id, score))
                    local_used_brands.add(brand)
                    price_ranges_used.add(price_range)
            else:
                # Second half: relaxed diversity (avoid identical brand+price combos)
                if not (brand_penalty and price_penalty):
                    diverse_recs.append((watch_id, score))
                    local_used_brands.add(brand)
                    price_ranges_used.add(price_range)
        
        # If we still need more and have candidates, take the best remaining
        if len(diverse_recs) < max_count:
            remaining_candidates = candidate_recommendations[len(diverse_recs):]
            for watch_id, score in remaining_candidates:
                if len(diverse_recs) >= max_count:
                    break
                if watch_id in self.watch_data:
                    diverse_recs.append((watch_id, score))
        
        return diverse_recs
    
    def _diverse_exploration_sampling(self, available_watches: List[int], 
                                    sample_size: int, used_brands: Set[str]) -> List[int]:
        """Sample exploration watches with diversity preferences."""
        if sample_size >= len(available_watches):
            return available_watches
        
        # Group watches by brand
        brand_groups = {}
        for watch_id in available_watches:
            if watch_id in self.watch_data:
                brand = self.watch_data[watch_id].get('brand', 'Unknown')
                if brand not in brand_groups:
                    brand_groups[brand] = []
                brand_groups[brand].append(watch_id)
        
        # Sample with brand diversity preference
        selected = []
        
        # Randomize brand order to see different brands across batches
        available_brands = [brand for brand in brand_groups.keys() if brand not in used_brands]
        if available_brands:
            # Shuffle to get different brand combinations each time
            random.shuffle(available_brands)
            
            # Select one watch from each available brand (up to sample_size)
            for brand in available_brands:
                if len(selected) >= sample_size:
                    break
                selected.append(random.choice(brand_groups[brand]))
        
        # If we need more watches and have exhausted unused brands, 
        # sample from used brands or any remaining watches
        if len(selected) < sample_size:
            remaining = [w for w in available_watches if w not in selected]
            additional_needed = sample_size - len(selected)
            if remaining:
                additional = random.sample(remaining, min(additional_needed, len(remaining)))
                selected.extend(additional)
        
        return selected
    
    def _get_price_range(self, price: float) -> str:
        """Categorize price into ranges for diversity."""
        if price < 500:
            return 'budget'
        elif price < 2000:
            return 'mid_range'
        elif price < 5000:
            return 'premium'
        else:
            return 'luxury'
    
    def update(self,
               session_id: str,
               watch_id: int,
               reward: float,
               context: np.ndarray) -> None:
        """Update system with feedback using Clean Dynamic Clustering approach."""
        
        # Initialize session tracking if needed
        if session_id not in self.session_liked_watches:
            self.session_liked_watches[session_id] = []
        if session_id not in self.session_experts:
            self.session_experts[session_id] = []
        
        # Get watch embedding
        if watch_id not in self.watch_embeddings:
            logger.warning(f"Watch {watch_id} has no embedding, skipping update")
            return
            
        watch_embedding = self.watch_embeddings[watch_id]
        
        # Extend context to full dimension if needed (for LinUCB algorithm)
        if len(context) < self.dim:
            full_context = np.zeros(self.dim)
            full_context[:len(context)] = context
            context = full_context
        
        # Track likes
        if reward > 0:  # User liked the watch
            if watch_id not in self.session_liked_watches[session_id]:
                self.session_liked_watches[session_id].append(watch_id)
                logger.info(f"💝 Session {session_id}: Watch {watch_id} liked! Total session likes: {len(self.session_liked_watches[session_id])}")
        
        session_experts = self.session_experts[session_id]
        
        # CASE 1: NO EXPERTS YET - first like creates first expert
        if len(session_experts) == 0:
            if reward > 0:  # First like!
                logger.info(f"🎯 FIRST LIKE! Creating Expert {self.next_expert_id} for session {session_id}")
                expert_id = self._create_new_expert()
                self.session_experts[session_id].append(expert_id)
                
                # Add the liked watch to define this expert's preferences
                expert = self.experts[expert_id]
                expert.add_liked_watch(watch_id, watch_embedding)
                
                # Update the expert with this feedback
                expert.update(watch_id, reward, watch_embedding)
                
                logger.info(f"✅ Expert {expert_id} created with centroid from watch {watch_id}")
            else:
                logger.info(f"👎 Session {session_id}: Watch {watch_id} disliked during random exploration")
            return
        
        # CASE 2: EXPERTS EXIST - assign to best expert or create new one
        if reward > 0:  # User liked the watch
            # Find the best existing expert for this watch
            best_expert_id = None
            best_similarity = -1.0
            
            for expert_id in session_experts:
                if expert_id in self.experts:
                    expert = self.experts[expert_id]
                    if expert.centroid is not None:
                        similarity = self._cosine_similarity(watch_embedding, expert.centroid)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_expert_id = expert_id
            
            # Decide: join existing expert or create new one
            if best_similarity >= self.similarity_threshold:
                # Join existing expert - add to its learned preferences
                logger.info(f"📌 Adding liked watch {watch_id} to existing Expert {best_expert_id} (similarity: {best_similarity:.3f})")
                expert = self.experts[best_expert_id]
                expert.add_liked_watch(watch_id, watch_embedding)
                expert.update(watch_id, reward, watch_embedding)
                
            else:
                # Create new expert (if we haven't hit the limit)
                if len(session_experts) < self.max_experts:
                    logger.info(f"🆕 Creating NEW Expert {self.next_expert_id} for session {session_id} (similarity: {best_similarity:.3f} < {self.similarity_threshold})")
                    expert_id = self._create_new_expert()
                    self.session_experts[session_id].append(expert_id)
                    
                    # Initialize new expert with this liked watch
                    expert = self.experts[expert_id]
                    expert.add_liked_watch(watch_id, watch_embedding)
                    expert.update(watch_id, reward, watch_embedding)
                    
                    logger.info(f"✅ Expert {expert_id} created with centroid from watch {watch_id}")
                else:
                    # Hit expert limit - assign to best existing expert anyway
                    logger.info(f"📌 Expert limit reached, adding watch {watch_id} to best Expert {best_expert_id}")
                    expert = self.experts[best_expert_id]
                    expert.add_liked_watch(watch_id, watch_embedding)
                    expert.update(watch_id, reward, watch_embedding)
        
        else:  # User disliked the watch
            # Update all experts with this negative feedback
            for expert_id in session_experts:
                if expert_id in self.experts:
                    expert = self.experts[expert_id]
                    expert.update(watch_id, reward, watch_embedding)
            logger.info(f"👎 Updated all experts with dislike for watch {watch_id}")
        
        # Log current session state
        logger.info(f"📊 Session {session_id} status: {len(self.session_experts[session_id])} experts, {len(self.session_liked_watches[session_id])} likes")
    
    def _reduce_features(self, embedding: np.ndarray, target_dim: int) -> np.ndarray:
        """Reduce embedding dimensionality using PCA - OPTIMIZED FOR INFORMATION RETENTION."""
        # Handle empty or invalid embeddings
        if embedding is None or len(embedding) == 0:
            logger.warning("Empty embedding provided, returning zero vector")
            return np.zeros(target_dim)
            
        if len(embedding) <= target_dim:
            result = np.zeros(target_dim)
            result[:len(embedding)] = embedding
            return result
        else:
            # Initialize PCA if not done yet
            if not hasattr(self, '_pca_reducer'):
                logger.info(f"Initializing PCA reducer for {target_dim}D target dimension")
                
                # FIXED: Fit PCA on RAW embeddings during data loading
                if hasattr(self, '_raw_embeddings') and len(self._raw_embeddings) > 0:
                    embeddings_matrix = np.array(self._raw_embeddings)
                    
                    # Standardize and fit PCA
                    self._scaler = StandardScaler()
                    embeddings_scaled = self._scaler.fit_transform(embeddings_matrix)
                    
                    self._pca_reducer = PCA(n_components=target_dim)
                    self._pca_reducer.fit(embeddings_scaled)
                    
                    logger.info(f"PCA reducer initialized: {target_dim}D retains "
                              f"{np.sum(self._pca_reducer.explained_variance_ratio_)*100:.1f}% variance")
                else:
                    logger.warning("No raw embeddings available for PCA, using truncation")
                    return embedding[:target_dim] if len(embedding) >= target_dim else np.pad(embedding, (0, target_dim - len(embedding)))
            
            try:
                # Apply standardization and PCA reduction
                embedding_scaled = self._scaler.transform(embedding.reshape(1, -1))
                reduced = self._pca_reducer.transform(embedding_scaled)
                return reduced.flatten()
            except Exception as e:
                logger.warning(f"PCA reduction failed: {e}, using truncation")
                return embedding[:target_dim] if len(embedding) >= target_dim else np.pad(embedding, (0, target_dim - len(embedding)))
    
    def _create_fallback_data(self) -> None:
        """Create minimal fallback data if loading fails."""
        logger.warning("Creating fallback data for testing")
        
        for i in range(20):  # Create more test watches
            watch_id = i
            self.watch_data[watch_id] = {
                'watch_id': watch_id,
                'index': watch_id,
                'brand': f'TestBrand{i%5}',  # 5 different brands
                'model': f'Model{i}',
                'price': 1000 + i * 200,
                'description': f'Test watch {i}',
                'image_url': f'https://example.com/watch{i}.jpg',
                'product_url': f'https://example.com/watch{i}',
                'source': 'test_data',
                'specs': {
                    'brand': f'TestBrand{i%5}',
                    'model': f'Model{i}',
                    'case_material': 'Stainless Steel',
                    'movement': 'Automatic'
                },
                'algorithm': 'dynamic_multi_expert_linucb',
                'confidence': 0.5,
                'score': 0.0
            }
            
            # Create diverse embeddings for different brands
            base_embedding = np.random.randn(1536) * 0.1
            brand_offset = (i % 5) * 2.0  # Different brands have different embedding regions
            base_embedding[:10] += brand_offset
            
            self.watch_embeddings[watch_id] = base_embedding
            self.unassigned_watches.add(watch_id)
    
    def get_watches(self, watch_ids: Set[int]) -> List[Dict[str, Any]]:
        """Get watch data for multiple watches."""
        return [
            self.watch_data[watch_id]
            for watch_id in watch_ids
            if watch_id in self.watch_data
        ]
    
    def get_watch_details(self, watch_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific watch."""
        return self.watch_data.get(watch_id)

    def _format_recommendation(self, watch_id: int, confidence: float, recommendation_type: str) -> Dict[str, Any]:
        """Format a recommendation as a dictionary."""
        if watch_id in self.watch_data:
            watch = self.watch_data[watch_id].copy()
            watch.update({
                'score': confidence,
                'confidence': confidence,
                'algorithm': recommendation_type,
                'watch_id': watch_id
            })
            return watch
        else:
            return {
                'watch_id': watch_id,
                'confidence': confidence,
                'score': confidence,
                'algorithm': recommendation_type,
                'error': 'Watch not found'
            }

# For backward compatibility, alias the dynamic engine as the main engine
MultiExpertLinUCBEngine = DynamicMultiExpertLinUCBEngine
LinUCBEngine = DynamicMultiExpertLinUCBEngine 