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

class SimplifiedExpert:
    """Simplified expert using contextual bandits - one arm per expert, not per watch."""
    def __init__(self, expert_id: int, dim: int, alpha: float):
        self.expert_id = expert_id
        self.dim = dim
        self.alpha = alpha
        
        # Single LinUCB arm for this expert (using contexts = watch embeddings)
        self.A = np.identity(dim)  # Feature covariance matrix
        self.b = np.zeros(dim)     # Feature-reward vector 
        self.theta = np.zeros(dim) # Learned weights
        
        # Expert's preference tracking
        self.liked_watches: List[int] = []
        self.centroid: Optional[np.ndarray] = None
    
    def add_liked_watch(self, watch_id: int, embedding: np.ndarray):
        """Add a liked watch to track expert preferences."""
        self.liked_watches.append(watch_id)
        
        # Update centroid (simple running average)
        if self.centroid is None:
            self.centroid = embedding.copy()
        else:
            # Running average
            alpha = 1.0 / len(self.liked_watches)
            self.centroid = (1 - alpha) * self.centroid + alpha * embedding
        
        # Normalize for cosine similarity
        norm = np.linalg.norm(self.centroid)
        if norm > 0:
            self.centroid = self.centroid / norm
    
    def update(self, context: np.ndarray, reward: float):
        """Update expert with feedback using context (watch embedding)."""
        # Standard LinUCB update
        self.A += np.outer(context, context)
        self.b += reward * context
        
        # Solve for theta (expert's learned weights)
        try:
            self.theta = np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:
            # Fallback for numerical issues
            A_reg = self.A + 1e-6 * np.identity(self.dim)
            self.theta = np.linalg.solve(A_reg, self.b)
    
    def get_ucb_score(self, context: np.ndarray) -> float:
        """Get UCB score using context (watch embedding)."""
        # Standard contextual UCB
        mean = np.dot(self.theta, context)
        
        try:
            A_inv = np.linalg.inv(self.A + 1e-6 * np.identity(self.dim))
            confidence = self.alpha * np.sqrt(np.dot(context, np.dot(A_inv, context)))
        except np.linalg.LinAlgError:
            # Simple fallback
            confidence = self.alpha * np.linalg.norm(context) / np.sqrt(len(self.liked_watches) + 1)
        
        return mean + confidence
    
    def batch_get_ucb_scores(self, contexts: np.ndarray) -> np.ndarray:
        """Vectorized UCB scoring for multiple contexts."""
        # Vectorized mean calculation
        means = contexts @ self.theta
        
        try:
            A_inv = np.linalg.inv(self.A + 1e-6 * np.identity(self.dim))
            # Vectorized confidence calculation
            confidences = self.alpha * np.sqrt(np.sum(contexts @ A_inv * contexts, axis=1))
        except np.linalg.LinAlgError:
            # Fallback
            confidences = self.alpha * np.linalg.norm(contexts, axis=1) / np.sqrt(len(self.liked_watches) + 1)
        
        return means + confidences

class OptimizedLinUCBEngine:
    """Optimized Multi-Expert LinUCB engine with performance improvements."""
    def __init__(self, 
                 text_dim: int = 100,  # Target text PCA dimensions
                 clip_dim: int = 100,  # Target CLIP PCA dimensions  
                 alpha: float = 0.1,  # Optimal: Lower exploration for better convergence
                 batch_size: int = 5,
                 max_experts: int = 4,  # Optimal: 4 experts for balanced specialization
                 similarity_threshold: float = 0.95,  # Very high threshold for testing dress watch similarity
                 data_dir: Optional[str] = None):
        """Initialize optimized engine."""
        self.text_dim = text_dim
        self.clip_dim = clip_dim
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_experts = max_experts
        self.similarity_threshold = similarity_threshold
        
        # Dimension will be set after loading data
        self.dim = None  # Auto-detected from actual embeddings
        
        # Expert management
        self.experts: Dict[int, SimplifiedExpert] = {}
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
        
        logger.info(f"✅ Simplified LinUCB engine initialized with {len(self.watch_data)} watches (embedding dim: {self.dim})")
    
    def _load_data(self) -> None:
        """Load and preprocess watch data with detailed logging."""
        import time
        import traceback
        
        total_start = time.time()
        logger.info("🔄 Starting data loading process...")
        
        try:
            # Load watch metadata
            logger.info("📖 Loading watch metadata...")
            load_start = time.time()
            metadata_path = os.path.join(self.data_dir, 'watch_text_metadata.pkl')
            with open(metadata_path, 'rb') as f:
                metadata_list = pickle.load(f)
            logger.info(f"✅ Loaded {len(metadata_list)} watch metadata entries in {time.time() - load_start:.2f}s")
            
            # Load text embeddings
            logger.info("📖 Loading text embeddings...")
            load_start = time.time()
            text_embeddings_path = os.path.join(self.data_dir, 'watch_text_embeddings.pkl')
            with open(text_embeddings_path, 'rb') as f:
                text_embeddings_array = pickle.load(f)
            text_shape = text_embeddings_array.shape if hasattr(text_embeddings_array, 'shape') else f"List of {len(text_embeddings_array)}"
            logger.info(f"✅ Loaded text embeddings {text_shape} in {time.time() - load_start:.2f}s")
            
            # Load CLIP embeddings
            logger.info("📖 Loading CLIP embeddings...")
            load_start = time.time()
            clip_embeddings_path = os.path.join(self.data_dir, 'watch_clip_embeddings.pkl')
            try:
                with open(clip_embeddings_path, 'rb') as f:
                    clip_embeddings_array = pickle.load(f)
                clip_shape = clip_embeddings_array.shape if hasattr(clip_embeddings_array, 'shape') else f"List of {len(clip_embeddings_array)}"
                logger.info(f"✅ Loaded CLIP embeddings {clip_shape} in {time.time() - load_start:.2f}s")
            except FileNotFoundError:
                clip_embeddings_array = np.zeros((len(metadata_list), 512))
                logger.warning("⚠️ CLIP embeddings not found, using zeros")
            
            # Log original embedding sizes
            if len(text_embeddings_array) > 0:
                sample_text_size = len(text_embeddings_array[0]) if hasattr(text_embeddings_array[0], '__len__') else "scalar"
                logger.info(f"📏 Sample text embedding size: {sample_text_size}")
            if len(clip_embeddings_array) > 0:
                sample_clip_size = len(clip_embeddings_array[0]) if hasattr(clip_embeddings_array[0], '__len__') else "scalar"
                logger.info(f"📏 Sample CLIP embedding size: {sample_clip_size}")
            
            # Initialize PCA for both text and CLIP embeddings (ONLY DONE ONCE)
            pca_start = time.time()
            if not hasattr(self, '_text_pca_reducer'):
                logger.info(f"🔬 Initializing PCA reducers (text: {self.text_dim}D, CLIP: {self.clip_dim}D)...")
                
                # Collect sample embeddings for PCA fitting
                sample_size = min(1000, len(metadata_list))
                logger.info(f"📊 Using {sample_size} samples for PCA fitting")
                
                # Fit PCA for text embeddings
                logger.info("🔬 Fitting PCA for text embeddings...")
                pca_text_start = time.time()
                text_samples = []
                for idx in range(sample_size):
                    if idx < len(text_embeddings_array):
                        text_samples.append(text_embeddings_array[idx])
                
                if text_samples:
                    logger.info(f"📏 Text samples shape: {np.array(text_samples).shape}")
                    # Fit text PCA
                    self._text_scaler = StandardScaler()
                    text_scaled = self._text_scaler.fit_transform(text_samples)
                    self._text_pca_reducer = PCA(n_components=self.text_dim)
                    self._text_pca_reducer.fit(text_scaled)
                    explained_var = sum(self._text_pca_reducer.explained_variance_ratio_)
                    logger.info(f"✅ Text PCA fitted in {time.time() - pca_text_start:.2f}s (explained variance: {explained_var:.3f})")
                
                # Fit PCA for CLIP embeddings
                logger.info("🔬 Fitting PCA for CLIP embeddings...")
                pca_clip_start = time.time()
                clip_samples = []
                for idx in range(sample_size):
                    if idx < len(clip_embeddings_array):
                        clip_samples.append(clip_embeddings_array[idx])
                
                if clip_samples:
                    logger.info(f"📏 CLIP samples shape: {np.array(clip_samples).shape}")
                    # Fit CLIP PCA
                    self._clip_scaler = StandardScaler()
                    clip_scaled = self._clip_scaler.fit_transform(clip_samples)
                    self._clip_pca_reducer = PCA(n_components=self.clip_dim)
                    self._clip_pca_reducer.fit(clip_scaled)
                    explained_var = sum(self._clip_pca_reducer.explained_variance_ratio_)
                    logger.info(f"✅ CLIP PCA fitted in {time.time() - pca_clip_start:.2f}s (explained variance: {explained_var:.3f})")
                
                logger.info(f"✅ PCA initialization complete in {time.time() - pca_start:.2f}s")
            else:
                logger.info("✅ PCA reducers already initialized (skipping)")
            
            # Auto-detect actual embedding dimensions from first watch
            logger.info("🔍 Auto-detecting embedding dimensions...")
            if len(metadata_list) > 0:
                # Process first watch to get actual dimensions
                text_emb = text_embeddings_array[0] if len(text_embeddings_array) > 0 else None
                clip_emb = clip_embeddings_array[0] if len(clip_embeddings_array) > 0 else None
                
                text_reduced = self._reduce_text_features(text_emb) if text_emb is not None else np.zeros(self.text_dim)
                clip_reduced = self._reduce_clip_features(clip_emb) if clip_emb is not None else np.zeros(self.clip_dim)
                
                # Set actual dimension from concatenated embedding
                self.dim = len(text_reduced) + len(clip_reduced)
                logger.info(f"📏 Auto-detected embedding dimension: {len(text_reduced)}D text + {len(clip_reduced)}D CLIP = {self.dim}D total")
            else:
                # Fallback if no data
                self.dim = self.text_dim + self.clip_dim
                logger.warning(f"⚠️ No data found, using fallback dimension: {self.dim}D")
            
            # Process all watches
            logger.info("🔄 Processing all watches with PCA reduction...")
            process_start = time.time()
            
            processed_count = 0
            text_reduction_times = []
            clip_reduction_times = []
            
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
                        text_start = time.time()
                        text_emb = text_embeddings_array[idx]
                        text_reduced = self._reduce_text_features(text_emb)
                        text_reduction_times.append(time.time() - text_start)
                        
                        self.watch_text_reduced[watch_id] = text_reduced
                        
                        # Log sizes for first few watches
                        if idx < 3:
                            orig_size = len(text_emb) if hasattr(text_emb, '__len__') else "scalar"
                            reduced_size = len(text_reduced) if hasattr(text_reduced, '__len__') else "scalar"
                            logger.info(f"📏 Watch {watch_id}: Text {orig_size} → {reduced_size}")
                    
                    # Get and reduce CLIP embedding with PCA
                    if idx < len(clip_embeddings_array):
                        clip_start = time.time()
                        clip_emb = clip_embeddings_array[idx]
                        clip_reduced = self._reduce_clip_features(clip_emb)
                        clip_reduction_times.append(time.time() - clip_start)
                        
                        self.watch_clip_reduced[watch_id] = clip_reduced
                        
                        # Log sizes for first few watches
                        if idx < 3:
                            orig_size = len(clip_emb) if hasattr(clip_emb, '__len__') else "scalar"
                            reduced_size = len(clip_reduced) if hasattr(clip_reduced, '__len__') else "scalar"
                            logger.info(f"📏 Watch {watch_id}: CLIP {orig_size} → {reduced_size}")
                    
                    self.available_watches.add(watch_id)
                    processed_count += 1
                    
                    # Log progress every 100 watches
                    if processed_count % 100 == 0:
                        elapsed = time.time() - process_start
                        avg_text_time = np.mean(text_reduction_times[-100:]) if text_reduction_times else 0
                        avg_clip_time = np.mean(clip_reduction_times[-100:]) if clip_reduction_times else 0
                        logger.info(f"🔄 Processed {processed_count}/{len(metadata_list)} watches ({elapsed:.1f}s, avg: {(avg_text_time + avg_clip_time)*1000:.1f}ms/watch)")
                    
                except Exception as e:
                    logger.error(f"Error processing watch {idx}: {e}")
                    continue
            
            # Final timing statistics
            total_time = time.time() - total_start
            process_time = time.time() - process_start
            avg_text_time = np.mean(text_reduction_times) if text_reduction_times else 0
            avg_clip_time = np.mean(clip_reduction_times) if clip_reduction_times else 0
            
            logger.info(f"📊 Processing complete:")
            logger.info(f"   • Total time: {total_time:.2f}s")
            logger.info(f"   • Processing time: {process_time:.2f}s")
            logger.info(f"   • Avg text reduction: {avg_text_time*1000:.1f}ms")
            logger.info(f"   • Avg CLIP reduction: {avg_clip_time*1000:.1f}ms")
            logger.info(f"✅ Loaded {len(self.watch_data)} watches with PCA-reduced embeddings")
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            logger.error(f"❌ Error details: {traceback.format_exc()}")
            self._create_fallback_data()
    
    def _reduce_text_features(self, embedding: np.ndarray) -> np.ndarray:
        """Reduce text embedding dimensionality with PCA."""
        if embedding is None or len(embedding) == 0:
            return np.zeros(self.text_dim)
            
        if len(embedding) <= self.text_dim:
            return np.pad(embedding, (0, self.text_dim - len(embedding)))
        
        try:
            if hasattr(self, '_text_pca_reducer'):
                embedding_scaled = self._text_scaler.transform(embedding.reshape(1, -1))
                return self._text_pca_reducer.transform(embedding_scaled).flatten()
            else:
                return embedding[:self.text_dim]
        except:
            return embedding[:self.text_dim]
    
    def _reduce_clip_features(self, embedding: np.ndarray) -> np.ndarray:
        """Reduce CLIP embedding dimensionality with PCA."""
        if embedding is None or len(embedding) == 0:
            return np.zeros(self.clip_dim)
            
        if len(embedding) <= self.clip_dim:
            return np.pad(embedding, (0, self.clip_dim - len(embedding)))
        
        try:
            if hasattr(self, '_clip_pca_reducer'):
                embedding_scaled = self._clip_scaler.transform(embedding.reshape(1, -1))
                return self._clip_pca_reducer.transform(embedding_scaled).flatten()
            else:
                return embedding[:self.clip_dim]
        except:
            return embedding[:self.clip_dim]
    
    def create_session(self, session_id: str) -> None:
        """Initialize a new session with pre-computed embeddings."""
        import time
        session_start = time.time()
        logger.info(f"🔄 Creating session {session_id}...")
        
        # Pre-compute concatenated embeddings for all watches (no user scaling)
        self.session_embeddings[session_id] = {}
        
        # Process in batches to manage memory
        batch_size = 1000
        watch_ids = list(self.available_watches)
        logger.info(f"📊 Processing {len(watch_ids)} watches for session embedding precomputation...")
        
        for i in range(0, len(watch_ids), batch_size):
            batch_ids = watch_ids[i:i + batch_size]
            for watch_id in batch_ids:
                text_emb = self.watch_text_reduced[watch_id]
                clip_emb = self.watch_clip_reduced.get(watch_id, np.zeros(self.clip_dim))
                
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
        
        session_time = time.time() - session_start
        logger.info(f"✅ Created session {session_id} with {len(self.session_embeddings[session_id])} pre-computed embeddings in {session_time:.2f}s")
    
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
        
        # Add timing for UCB calculations
        import time
        ucb_start = time.time()
        
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
                batch_scores = expert.batch_get_ucb_scores(batch_embeddings)
                
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
        
        # Log UCB timing
        ucb_time = time.time() - ucb_start
        logger.info(f"⏱️ UCB calculations completed in {ucb_time:.2f}s for {len(available_watches)} watches")
        
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
                expert.update(watch_embedding, reward)
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
                expert.update(watch_embedding, reward)
                logger.info(f"✅ Expert {best_expert_id}: Added watch {watch_id} (similarity: {best_similarity:.3f} ≥ {self.similarity_threshold}) | Liked watches: {len(expert.liked_watches)}")
            elif len(session_experts) < self.max_experts:
                # Create new expert
                expert_id = self._create_new_expert()
                self.session_experts[session_id].append(expert_id)
                expert = self.experts[expert_id]
                expert.add_liked_watch(watch_id, watch_embedding)
                expert.update(watch_embedding, reward)
                logger.info(f"🆕 Expert {expert_id}: New expert created for watch {watch_id} (similarity: {best_similarity:.3f} < {self.similarity_threshold})")
                logger.info(f"📊 DEBUG: All expert similarities for watch {watch_id}: {[(exp_id, np.dot(watch_embedding, self.experts[exp_id].centroid) if self.experts[exp_id].centroid is not None else 0.0) for exp_id in session_experts[:-1]]}")
            else:
                # Add to best expert if at limit
                expert = self.experts[best_expert_id]
                expert.add_liked_watch(watch_id, watch_embedding)
                expert.update(watch_embedding, reward)
                logger.info(f"🔄 Expert {best_expert_id}: Added watch {watch_id} (at max experts, best similarity: {best_similarity:.3f}) | Liked watches: {len(expert.liked_watches)}")
        else:
            # Update all experts with negative feedback
            negative_count = 0
            for expert_id in session_experts:
                if expert_id in self.experts:
                    expert = self.experts[expert_id]
                    expert.update(watch_embedding, reward)
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
        self.experts[expert_id] = SimplifiedExpert(expert_id, self.dim, self.alpha)
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
        
        total_positive_feedback = 0
        
        for expert_id, expert in self.experts.items():
            expert_stats = {
                'expert_id': expert_id,
                'num_liked_watches': len(expert.liked_watches),
                'theta_norm': float(np.linalg.norm(expert.theta)),
                'A_trace': float(np.trace(expert.A))  # Measure of learning activity
            }
            
            expert_stats['total_positive_feedback'] = len(expert.liked_watches)
            total_positive_feedback += len(expert.liked_watches)
            
            stats['experts'].append(expert_stats)
        
        # Add global stats
        stats['global'] = {
            'total_positive_feedback': total_positive_feedback,
            'avg_feedback_per_expert': total_positive_feedback / len(self.experts) if self.experts else 0
        }
        
        return stats

# For backward compatibility
MultiExpertLinUCBEngine = OptimizedLinUCBEngine
LinUCBEngine = OptimizedLinUCBEngine 