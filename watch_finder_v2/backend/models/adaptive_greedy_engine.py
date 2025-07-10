"""
AdaptiveGreedy-based Recommendation Engine
========================================

Implements AdaptiveGreedy contextual bandits for watch recommendations:
- Uses embeddings as contexts for each watch
- Applies efficient filtering at session start
- Maintains same API as existing engines
- Uses adaptive threshold-based exploration
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import logging
import random
import time

# sklearn imports for base classifier and preprocessing
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    print("⚠️  sklearn not installed. Run: pip install scikit-learn")
    LogisticRegression = None
    StandardScaler = None
    PCA = None
    RandomForestClassifier = None

# AdaptiveGreedy from contextual-bandits
try:
    from contextualbandits.online import AdaptiveGreedy
except ImportError:
    print("⚠️  contextual-bandits not installed. Run: pip install contextual-bandits")
    AdaptiveGreedy = None

logger = logging.getLogger(__name__)

class AdaptiveGreedyEngine:
    """AdaptiveGreedy-based recommendation engine with efficient filtering."""
    
    def __init__(self, 
                 text_dim: int = 100,
                 clip_dim: int = 100,
                 batch_size: int = 5,
                 window_size: int = 500,
                 percentile: int = 30,
                 decay: float = 0.9998,
                 data_dir: Optional[str] = None):
        """Initialize AdaptiveGreedy engine."""
        
        if AdaptiveGreedy is None:
            raise ImportError("contextual-bandits package is required. Install with: pip install contextual-bandits")
        
        self.text_dim = text_dim
        self.clip_dim = clip_dim
        self.batch_size = batch_size
        self.window_size = window_size
        self.percentile = percentile
        self.decay = decay
        
        # Will be set after loading data
        self.dim = None
        self.nchoices = None
        
        # Session management
        self.session_adaptive_greedy: Dict[str, AdaptiveGreedy] = {}
        self.session_liked_watches: Dict[str, List[int]] = {}
        self.session_embeddings: Dict[str, Dict[int, np.ndarray]] = {}
        self.session_shown_watches: Dict[str, Set[int]] = {}
        self.session_filtered_watches: Dict[str, List[int]] = {}  # Pre-filtered watches per session
        
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
        
        logger.info(f"✅ AdaptiveGreedy engine initialized with {len(self.watch_data)} watches (embedding dim: {self.dim})")
    
    def _load_data(self) -> None:
        """Load and preprocess watch data."""
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
            logger.info(f"✅ Loaded text embeddings in {time.time() - load_start:.2f}s")
            
            # Load CLIP embeddings
            logger.info("📖 Loading CLIP embeddings...")
            load_start = time.time()
            clip_embeddings_path = os.path.join(self.data_dir, 'watch_clip_embeddings.pkl')
            try:
                with open(clip_embeddings_path, 'rb') as f:
                    clip_embeddings_array = pickle.load(f)
                logger.info(f"✅ Loaded CLIP embeddings in {time.time() - load_start:.2f}s")
            except FileNotFoundError:
                clip_embeddings_array = np.zeros((len(metadata_list), 512))
                logger.warning("⚠️ CLIP embeddings not found, using zeros")
            
            # Initialize PCA reducers
            pca_start = time.time()
            if not hasattr(self, '_text_pca_reducer'):
                logger.info(f"🔬 Initializing PCA reducers (text: {self.text_dim}D, CLIP: {self.clip_dim}D)...")
                
                sample_size = min(1000, len(metadata_list))
                
                # Fit PCA for text embeddings
                text_samples = text_embeddings_array[:sample_size] if len(text_embeddings_array) >= sample_size else text_embeddings_array
                if len(text_samples) > 0:
                    self._text_scaler = StandardScaler()
                    text_scaled = self._text_scaler.fit_transform(text_samples)
                    self._text_pca_reducer = PCA(n_components=self.text_dim)
                    self._text_pca_reducer.fit(text_scaled)
                    logger.info(f"✅ Text PCA fitted (explained variance: {sum(self._text_pca_reducer.explained_variance_ratio_):.3f})")
                
                # Fit PCA for CLIP embeddings
                clip_samples = clip_embeddings_array[:sample_size] if len(clip_embeddings_array) >= sample_size else clip_embeddings_array
                if len(clip_samples) > 0:
                    self._clip_scaler = StandardScaler()
                    clip_scaled = self._clip_scaler.fit_transform(clip_samples)
                    self._clip_pca_reducer = PCA(n_components=self.clip_dim)
                    self._clip_pca_reducer.fit(clip_scaled)
                    logger.info(f"✅ CLIP PCA fitted (explained variance: {sum(self._clip_pca_reducer.explained_variance_ratio_):.3f})")
                
                logger.info(f"✅ PCA initialization complete in {time.time() - pca_start:.2f}s")
            
            # Auto-detect embedding dimensions
            if len(metadata_list) > 0:
                text_emb = text_embeddings_array[0] if len(text_embeddings_array) > 0 else None
                clip_emb = clip_embeddings_array[0] if len(clip_embeddings_array) > 0 else None
                
                text_reduced = self._reduce_text_features(text_emb) if text_emb is not None else np.zeros(self.text_dim)
                clip_reduced = self._reduce_clip_features(clip_emb) if clip_emb is not None else np.zeros(self.clip_dim)
                
                self.dim = len(text_reduced) + len(clip_reduced)
                logger.info(f"📏 Auto-detected embedding dimension: {self.dim}D total")
            else:
                self.dim = self.text_dim + self.clip_dim
            
            # Process all watches
            logger.info("🔄 Processing all watches with PCA reduction...")
            process_start = time.time()
            
            for idx, watch_dict in enumerate(metadata_list):
                try:
                    watch_id = watch_dict.get('index', idx)
                    
                    # Store watch data
                    self.watch_data[watch_id] = {
                        **watch_dict,
                        'watch_id': watch_id,
                        'index': watch_id
                    }
                    
                    # Process embeddings
                    if idx < len(text_embeddings_array):
                        text_emb = text_embeddings_array[idx]
                        text_reduced = self._reduce_text_features(text_emb)
                        self.watch_text_reduced[watch_id] = text_reduced
                    
                    if idx < len(clip_embeddings_array):
                        clip_emb = clip_embeddings_array[idx]
                        clip_reduced = self._reduce_clip_features(clip_emb)
                        self.watch_clip_reduced[watch_id] = clip_reduced
                    
                    self.available_watches.add(watch_id)
                    
                except Exception as e:
                    logger.error(f"Error processing watch {idx}: {e}")
                    continue
            
            # Set number of choices for AdaptiveGreedy
            self.nchoices = len(self.available_watches)
            logger.info(f"✅ Set nchoices = {self.nchoices} for AdaptiveGreedy")
            
            total_time = time.time() - total_start
            logger.info(f"✅ Data loading complete in {total_time:.2f}s with {len(self.watch_data)} watches")
            
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
    
    def _create_base_classifier(self):
        """Create base classifier for AdaptiveGreedy."""
        # Use LogisticRegression as base classifier
        # AdaptiveGreedy works best with probabilistic classifiers
        return LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='liblinear',  # Fast solver for small datasets
            C=1.0  # Regularization strength
        )
    
    def _initialize_adaptive_greedy(self, nchoices: int) -> AdaptiveGreedy:
        """Initialize AdaptiveGreedy instance."""
        base_algorithm = self._create_base_classifier()
        
        # Create AdaptiveGreedy with optimized parameters
        adaptive_greedy = AdaptiveGreedy(
            base_algorithm=base_algorithm,
            nchoices=nchoices,
            window_size=self.window_size,
            percentile=self.percentile,
            decay=self.decay,
            decay_type='percentile',
            initial_thr='auto',  # Automatically set threshold
            beta_prior='auto',   # Automatically set beta prior
            smoothing=None,      # Don't use smoothing with beta_prior
            active_choice=None,  # Random exploration when below threshold
            random_state=42,
            njobs=1  # Single-threaded for consistency
        )
        
        return adaptive_greedy
    
    def create_session(self, session_id: str, filter_preferences: Optional[Dict[str, Any]] = None) -> None:
        """Initialize session with efficient pre-filtering."""
        session_start = time.time()
        logger.info(f"🔄 Creating AdaptiveGreedy session {session_id}...")
        
        # Apply filters at session start for efficiency
        filtered_watches = self._get_filtered_watch_pool(filter_preferences or {})
        self.session_filtered_watches[session_id] = filtered_watches
        
        logger.info(f"🔍 Session {session_id}: Filtered to {len(filtered_watches)} watches from {len(self.available_watches)} total")
        
        # Pre-compute embeddings for filtered watches only
        self.session_embeddings[session_id] = {}
        for watch_id in filtered_watches:
            text_emb = self.watch_text_reduced[watch_id]
            clip_emb = self.watch_clip_reduced.get(watch_id, np.zeros(self.clip_dim))
            
            # Concatenate and normalize
            combined = np.concatenate([text_emb, clip_emb])
            combined_norm = np.linalg.norm(combined)
            if combined_norm > 0:
                combined = combined / combined_norm
            
            self.session_embeddings[session_id][watch_id] = combined
        
        # Initialize AdaptiveGreedy for this session
        # Each session gets its own AdaptiveGreedy instance
        if len(filtered_watches) > 0:
            self.session_adaptive_greedy[session_id] = self._initialize_adaptive_greedy(len(filtered_watches))
        
        # Initialize session tracking
        self.session_liked_watches[session_id] = []
        self.session_shown_watches[session_id] = set()
        
        session_time = time.time() - session_start
        logger.info(f"✅ Created AdaptiveGreedy session {session_id} with {len(filtered_watches)} watches in {session_time:.2f}s")
    
    def _get_filtered_watch_pool(self, filter_preferences: Dict[str, Any]) -> List[int]:
        """Apply all filters efficiently at session start."""
        available_watches = list(self.available_watches)
        
        # Apply price filters
        if 'price_range' in filter_preferences:
            available_watches = self._apply_price_filters(available_watches, filter_preferences['price_range'])
        
        # Apply brand filters
        if 'preferred_brands' in filter_preferences:
            available_watches = self._apply_brand_filters(available_watches, filter_preferences['preferred_brands'])
        
        # Apply style filters
        if 'watch_styles' in filter_preferences:
            available_watches = self._apply_style_filters(available_watches, filter_preferences['watch_styles'])
        
        # Apply size filters
        if 'case_size_range' in filter_preferences:
            available_watches = self._apply_size_filters(available_watches, filter_preferences['case_size_range'])
        
        # Deduplicate
        available_watches = self._get_unique_watches(set(), available_watches)
        
        return available_watches
    
    def _apply_price_filters(self, watches: List[int], price_range: Dict[str, float]) -> List[int]:
        """Filter watches by price range."""
        min_price = price_range.get('min', 0)
        max_price = price_range.get('max', float('inf'))
        
        filtered = []
        for watch_id in watches:
            watch = self.watch_data.get(watch_id, {})
            price = watch.get('price', 0)
            if isinstance(price, (int, float)) and min_price <= price <= max_price:
                filtered.append(watch_id)
        
        logger.info(f"🔍 Price filter ({min_price}-{max_price}): {len(watches)} → {len(filtered)} watches")
        return filtered
    
    def _apply_brand_filters(self, watches: List[int], preferred_brands: List[str]) -> List[int]:
        """Filter watches by preferred brands."""
        if not preferred_brands:
            return watches
        
        preferred_brands_lower = [brand.lower() for brand in preferred_brands]
        
        filtered = []
        for watch_id in watches:
            watch = self.watch_data.get(watch_id, {})
            brand = watch.get('brand', '').lower()
            if brand in preferred_brands_lower:
                filtered.append(watch_id)
        
        logger.info(f"🔍 Brand filter ({preferred_brands}): {len(watches)} → {len(filtered)} watches")
        return filtered
    
    def _apply_style_filters(self, watches: List[int], watch_styles: List[str]) -> List[int]:
        """Filter watches by styles/categories."""
        if not watch_styles:
            return watches
        
        style_keywords = [style.lower() for style in watch_styles]
        
        filtered = []
        for watch_id in watches:
            watch = self.watch_data.get(watch_id, {})
            
            # Check in multiple fields
            text_fields = [
                watch.get('description', '').lower(),
                watch.get('model', '').lower(),
                str(watch.get('specs', {})).lower()
            ]
            
            match_found = False
            for style in style_keywords:
                if any(style in field for field in text_fields):
                    match_found = True
                    break
            
            if match_found:
                filtered.append(watch_id)
        
        logger.info(f"🔍 Style filter ({watch_styles}): {len(watches)} → {len(filtered)} watches")
        return filtered
    
    def _apply_size_filters(self, watches: List[int], size_range: Dict[str, float]) -> List[int]:
        """Filter watches by case size range."""
        min_size = size_range.get('min', 0)
        max_size = size_range.get('max', 100)
        
        filtered = []
        for watch_id in watches:
            watch = self.watch_data.get(watch_id, {})
            specs = watch.get('specs', {})
            
            # Try to extract numeric case size
            case_size_str = specs.get('case_size', '0')
            try:
                # Extract numeric part (handle formats like "42mm", "42.5mm", etc.)
                import re
                size_match = re.search(r'(\d+\.?\d*)', str(case_size_str))
                if size_match:
                    case_size = float(size_match.group(1))
                    if min_size <= case_size <= max_size:
                        filtered.append(watch_id)
                else:
                    # If no size info, include in results
                    filtered.append(watch_id)
            except:
                # If parsing fails, include in results
                filtered.append(watch_id)
        
        logger.info(f"🔍 Size filter ({min_size}-{max_size}mm): {len(watches)} → {len(filtered)} watches")
        return filtered
    
    def get_recommendations(self,
                          session_id: str,
                          exclude_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """Get recommendations using AdaptiveGreedy."""
        exclude_ids = exclude_ids or set()
        
        # Ensure session exists
        if session_id not in self.session_adaptive_greedy:
            self.create_session(session_id)
        
        # Get pre-filtered watches for this session
        filtered_watches = self.session_filtered_watches.get(session_id, [])
        if not filtered_watches:
            logger.warning(f"No filtered watches for session {session_id}")
            return []
        
        # Exclude shown watches and provided excludes
        session_shown_watches = self.session_shown_watches.setdefault(session_id, set())
        all_excludes = exclude_ids | session_shown_watches
        
        # Get available watches from pre-filtered pool
        available_watches = [w for w in filtered_watches if w not in all_excludes]
        
        if not available_watches:
            logger.warning(f"No available watches left for session {session_id}")
            return []
        
        adaptive_greedy = self.session_adaptive_greedy[session_id]
        
        # Prepare contexts (embeddings) for available watches
        contexts = np.array([
            self.session_embeddings[session_id][watch_id]
            for watch_id in available_watches
        ])
        
        logger.info(f"🎯 Session {session_id}: AdaptiveGreedy predicting for {len(available_watches)} watches")
        
        # Get predictions from AdaptiveGreedy
        ag_start = time.time()
        try:
            # AdaptiveGreedy.predict() returns arm indices
            # We need to map these back to watch IDs
            predicted_arms = adaptive_greedy.predict(contexts)
            
            # Convert arm indices to watch IDs and scores
            recommendations = []
            for i, arm_idx in enumerate(predicted_arms[:self.batch_size]):
                if arm_idx < len(available_watches):
                    watch_id = available_watches[arm_idx]
                    
                    # Get confidence score from prediction probabilities if available
                    try:
                        if hasattr(adaptive_greedy.base_algorithm, 'predict_proba'):
                            proba = adaptive_greedy.base_algorithm.predict_proba(contexts[arm_idx:arm_idx+1])
                            confidence = float(proba[0][1]) if len(proba[0]) > 1 else 0.5
                        else:
                            confidence = 0.5  # Default confidence
                    except:
                        confidence = 0.5
                    
                    recommendations.append(
                        self._format_recommendation(watch_id, confidence, "AdaptiveGreedy")
                    )
            
            ag_time = time.time() - ag_start
            logger.info(f"⏱️ AdaptiveGreedy prediction completed in {ag_time:.3f}s")
            
            # Track shown watches
            shown_watch_ids = [rec.get('watch_id') for rec in recommendations if rec.get('watch_id') is not None]
            session_shown_watches.update(shown_watch_ids)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ AdaptiveGreedy prediction failed: {e}")
            # Fallback to random selection
            selected_watches = np.random.choice(
                available_watches,
                size=min(self.batch_size, len(available_watches)),
                replace=False
            ).tolist()
            
            session_shown_watches.update(selected_watches)
            return [self._format_recommendation(watch_id, 0.5, "Random_Fallback") for watch_id in selected_watches]
    
    def update(self, session_id: str, watch_id: int, reward: float) -> None:
        """Update AdaptiveGreedy with feedback."""
        if session_id not in self.session_adaptive_greedy:
            logger.warning(f"Session {session_id} not found for update")
            return
        
        if watch_id not in self.session_embeddings.get(session_id, {}):
            logger.warning(f"Watch {watch_id} not found in session {session_id} embeddings")
            return
        
        # Track likes
        if reward > 0:
            if watch_id not in self.session_liked_watches[session_id]:
                self.session_liked_watches[session_id].append(watch_id)
        
        # Get watch embedding and arm index
        watch_embedding = self.session_embeddings[session_id][watch_id]
        
        # Find arm index for this watch in the filtered set
        filtered_watches = self.session_filtered_watches.get(session_id, [])
        try:
            arm_idx = filtered_watches.index(watch_id)
        except ValueError:
            logger.warning(f"Watch {watch_id} not found in filtered watches for session {session_id}")
            return
        
        # Update AdaptiveGreedy
        adaptive_greedy = self.session_adaptive_greedy[session_id]
        
        try:
            # AdaptiveGreedy expects context and reward for the chosen arm
            # Reshape context for single prediction
            context = watch_embedding.reshape(1, -1)
            
            # Convert reward to binary (AdaptiveGreedy typically uses binary rewards)
            binary_reward = 1 if reward > 0 else 0
            
            # Update the model
            adaptive_greedy.partial_fit(context, [arm_idx], [binary_reward])
            
            logger.info(f"✅ AdaptiveGreedy updated: session {session_id}, watch {watch_id}, reward {reward} → {binary_reward}")
            
        except Exception as e:
            logger.error(f"❌ AdaptiveGreedy update failed: {e}")
    
    def _get_unique_watches(self, exclude_ids: Set[int], candidate_watches: Optional[List[int]] = None) -> List[int]:
        """Get unique watches by signature, from candidate list if provided."""
        if candidate_watches is None:
            candidate_watches = list(self.available_watches)
        
        seen_signatures = set()
        unique_watches = []
        
        for watch_id in candidate_watches:
            if watch_id in exclude_ids:
                continue
                
            watch = self.watch_data.get(watch_id, {})
            specs = watch.get('specs', {})
            
            # Create comprehensive signature
            signature_parts = []
            
            brand = watch.get('brand', '').strip().lower()
            model = watch.get('model', '').strip().lower()
            signature_parts.append(f"brand:{brand}")
            signature_parts.append(f"model:{model}")
            
            # Key differentiating specs
            for key in ['dial_color', 'case_material', 'movement', 'case_size', 'strap_material']:
                value = specs.get(key, '').strip().lower()
                if value and value != '-' and value != 'null':
                    signature_parts.append(f"{key}:{value}")
            
            product_url = watch.get('product_url', '').strip().lower()
            if product_url:
                signature_parts.append(f"url:{product_url}")
            
            full_signature = "|".join(signature_parts)
            
            if full_signature not in seen_signatures:
                seen_signatures.add(full_signature)
                unique_watches.append(watch_id)
        
        return unique_watches
    
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
            text_reduced = np.random.randn(self.text_dim)
            clip_reduced = np.random.randn(self.clip_dim)
            
            self.watch_text_reduced[watch_id] = text_reduced
            self.watch_clip_reduced[watch_id] = clip_reduced
            self.available_watches.add(watch_id)
        
        self.nchoices = 20

    def shutdown(self) -> None:
        """Clean up resources."""
        logger.info("🛑 Shutting down AdaptiveGreedyEngine...")
        self.session_adaptive_greedy.clear()
        self.watch_data.clear()
        self.available_watches.clear()
        self.session_embeddings.clear()
        logger.info("✅ AdaptiveGreedyEngine shutdown complete")

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get session statistics."""
        if session_id not in self.session_adaptive_greedy:
            return {'error': 'Session not found'}
        
        return {
            'session_id': session_id,
            'num_liked_watches': len(self.session_liked_watches.get(session_id, [])),
            'num_shown_watches': len(self.session_shown_watches.get(session_id, set())),
            'num_filtered_watches': len(self.session_filtered_watches.get(session_id, [])),
            'adaptive_greedy_fitted': hasattr(self.session_adaptive_greedy[session_id], 'base_algorithm')
        } 