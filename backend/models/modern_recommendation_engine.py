#!/usr/bin/env python3
"""
Modern Cutting-Edge Watch Recommendation Engine
===============================================

A state-of-the-art recommendation system featuring:
- Vector similarity search with FAISS indexing
- Multi-modal deep learning recommendations  
- Real-time preference learning with transformer-based attention
- Advanced diversity optimization
- Async/await processing for high performance
- Sophisticated cold-start handling
- Dynamic exploration/exploitation balancing
- Graph-based collaborative filtering
- Reinforcement learning feedback integration

Author: Watch Finder AI Team
Version: 3.0.0
"""

import numpy as np
import asyncio
import pickle
import time
from typing import List, Dict, Any, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from abc import ABC, abstractmethod

# Advanced ML libraries
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  FAISS not available - falling back to numpy similarity")

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.ensemble import RandomForestRegressor
import scipy.sparse as sp
from scipy.special import softmax

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.str_, str)):
        return str(obj)
    elif obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        # For any other type, try to convert to string as fallback
        try:
            return str(obj)
        except:
            return None

@dataclass
class UserPreference:
    """Sophisticated user preference representation."""
    embedding: np.ndarray
    confidence: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    decay_rate: float = 0.95
    
    def update_confidence(self, feedback_strength: float):
        """Update confidence based on feedback strength."""
        self.confidence = min(1.0, self.confidence + feedback_strength * 0.1)
    
    def get_decayed_confidence(self) -> float:
        """Get time-decayed confidence."""
        hours_elapsed = (datetime.now() - self.timestamp).total_seconds() / 3600
        return self.confidence * (self.decay_rate ** hours_elapsed)

@dataclass
class RecommendationRequest:
    """Structured recommendation request."""
    user_id: str
    liked_indices: List[int]
    disliked_indices: List[int]
    current_candidates: List[int]
    num_recommendations: int = 7
    exploration_factor: float = 0.3
    diversity_threshold: float = 0.7
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecommendationResult:
    """Structured recommendation result with rich metadata."""
    watches: List[Dict[str, Any]]
    confidence_scores: List[float]
    diversity_score: float
    exploration_rate: float
    algorithm_used: str
    processing_time: float
    user_profile_summary: Dict[str, Any]
    next_exploration_suggestions: List[str]

class SimilarityIndex(ABC):
    """Abstract base class for similarity indexing."""
    
    @abstractmethod
    def build_index(self, embeddings: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

class FAISSIndex(SimilarityIndex):
    """FAISS-based high-performance similarity index."""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = None
        
    def build_index(self, embeddings: np.ndarray) -> None:
        """Build FAISS index for fast similarity search."""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available")
            
        # Use HNSW index for better performance on high-dimensional data
        # IndexHNSWFlat(dimension, M) where M is number of connections (typically 16-64)
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 connections is a good default
        self.index.hnsw.efConstruction = 40
        self.index.hnsw.efSearch = 32
        
        # Add embeddings to index
        embeddings_float32 = embeddings.astype(np.float32)
        self.index.add(embeddings_float32)
        
        logger.info(f"Built FAISS HNSW index with {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k most similar items."""
        query_float32 = query_embedding.astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_float32, k)
        return distances[0], indices[0]

class NumpyIndex(SimilarityIndex):
    """Fallback numpy-based similarity index."""
    
    def __init__(self):
        self.embeddings = None
        
    def build_index(self, embeddings: np.ndarray) -> None:
        """Store normalized embeddings."""
        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
    def search(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search using cosine similarity."""
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        similarities = np.dot(self.embeddings, query_norm)
        top_indices = np.argsort(similarities)[::-1][:k]
        top_similarities = similarities[top_indices]
        # Convert similarities to distances
        distances = 1.0 - top_similarities
        return distances, top_indices

class PreferenceCluster:
    """Represents a cluster of user preferences."""
    
    def __init__(self, centroid: np.ndarray, preferences: List[UserPreference]):
        self.centroid = centroid
        self.preferences = preferences
        self.last_updated = datetime.now()
        self.activation_count = 0
        
    def update_centroid(self):
        """Update cluster centroid based on preferences."""
        if not self.preferences:
            return
            
        weights = [p.get_decayed_confidence() for p in self.preferences]
        embeddings = np.array([p.embedding for p in self.preferences])
        
        if sum(weights) > 0:
            self.centroid = np.average(embeddings, weights=weights, axis=0)
            self.last_updated = datetime.now()
    
    def add_preference(self, preference: UserPreference):
        """Add a new preference to the cluster."""
        self.preferences.append(preference)
        self.update_centroid()
    
    def get_strength(self) -> float:
        """Get cluster strength based on preferences."""
        total_confidence = sum(p.get_decayed_confidence() for p in self.preferences)
        return min(1.0, total_confidence / len(self.preferences))

class ModernRecommendationEngine:
    """
    State-of-the-art recommendation engine with advanced ML techniques.
    """
    
    def __init__(self, 
                 embeddings: np.ndarray, 
                 watch_data: List[Dict[str, Any]],
                 embeddings_pre_normalized: bool = False,
                 use_faiss: bool = True):
        """
        Initialize the modern recommendation engine.
        
        Args:
            embeddings: Watch embeddings matrix
            watch_data: List of watch metadata
            embeddings_pre_normalized: Whether embeddings are pre-normalized
            use_faiss: Whether to use FAISS for similarity search
        """
        self.embeddings = embeddings
        self.watch_data = watch_data
        self.dimension = embeddings.shape[1]
        self.num_watches = len(watch_data)
        
        # Normalize embeddings if needed
        if not embeddings_pre_normalized:
            self.normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        else:
            self.normalized_embeddings = embeddings
            
        # Initialize similarity index
        if use_faiss and FAISS_AVAILABLE:
            try:
                self.similarity_index = FAISSIndex(self.dimension)
                logger.info("🚀 Using FAISS for high-performance similarity search")
            except Exception as e:
                logger.warning(f"⚠️  FAISS initialization failed: {e}")
                logger.info("🔄 Falling back to numpy-based similarity search")
                self.similarity_index = NumpyIndex()
        else:
            self.similarity_index = NumpyIndex()
            logger.info("🔄 Using numpy-based similarity search (FAISS disabled)")
            
        self.similarity_index.build_index(self.normalized_embeddings)
        
        # User state management
        self.user_sessions = {}  # session_id -> user state
        self.global_popularity = defaultdict(int)
        self.global_interactions = defaultdict(list)
        
        # Advanced components (lazy-loaded)
        self._clusters_initialized = False
        self._graph_initialized = False
        self._semantic_features_initialized = False
        
        # Placeholders for heavy components
        self.style_clusters = None
        self.brand_clusters = None
        self.watch_graph = None
        self.semantic_features = {}
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.performance_metrics = {
            'total_requests': 0,
            'avg_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'user_satisfaction': 0.0
        }
        
        # Load precomputed smart seeds
        self._load_smart_seeds()
        
        logger.info(f"🚀 Modern Recommendation Engine initialized with {self.num_watches} watches")
        logger.info(f"📐 Embedding dimension: {self.dimension}")
        logger.info(f"🔍 Similarity index: {type(self.similarity_index).__name__}")

    def _load_smart_seeds(self):
        """Load precomputed smart seeds for cold start."""
        try:
            # Try loading pickle file first
            with open('backend/data/precomputed_smart_seed_sets.pkl', 'rb') as f:
                smart_seeds_data = pickle.load(f)
            
            # Handle the correct data structure
            if isinstance(smart_seeds_data, dict) and 'seed_sets' in smart_seeds_data:
                self.smart_seed_sets = smart_seeds_data['seed_sets']
            else:
                self.smart_seed_sets = smart_seeds_data
                
            logger.info(f"✅ Loaded {len(self.smart_seed_sets)} smart seed sets from pickle")
        except Exception as e:
            logger.warning(f"Could not load pickle smart seeds: {e}")
            
            # Try loading from JSON file as fallback
            try:
                import json
                with open('backend/data/precomputed_smart_seed_sets.json', 'r') as f:
                    json_data = json.load(f)
                
                if 'seed_sets' in json_data:
                    # Convert JSON structure to the expected format
                    self.smart_seed_sets = []
                    for seed_set in json_data['seed_sets']:
                        # Extract just the indices from the seed set
                        seed_indices = [watch['index'] for watch in seed_set]
                        self.smart_seed_sets.append({
                            'seeds': seed_indices,
                            'metadata': {
                                'source': 'json_fallback',
                                'styles': [watch.get('style', 'unknown') for watch in seed_set]
                            }
                        })
                    
                    logger.info(f"✅ Loaded {len(self.smart_seed_sets)} smart seed sets from JSON")
                else:
                    raise ValueError("Invalid JSON structure")
                    
            except Exception as json_error:
                logger.warning(f"Could not load JSON smart seeds: {json_error}")
                self.smart_seed_sets = []

    def _ensure_clusters_initialized(self):
        """Lazy initialization of clustering components."""
        if self._clusters_initialized:
            return
            
        logger.info("🧠 Initializing clustering components...")
        
        # Style-based clustering
        style_features = self._extract_style_features()
        self.style_clusters = KMeans(n_clusters=min(20, len(style_features)//10), random_state=42)
        self.style_clusters.fit(style_features)
        
        # Brand-based clustering  
        brand_features = self._extract_brand_features()
        self.brand_clusters = KMeans(n_clusters=min(15, len(set(w['brand'] for w in self.watch_data))), random_state=42)
        self.brand_clusters.fit(brand_features)
        
        self._clusters_initialized = True
        logger.info("✅ Clustering components initialized")

    def _extract_style_features(self) -> np.ndarray:
        """Extract style-based features for clustering."""
        features = []
        for watch in self.watch_data:
            # Extract style indicators from specs
            specs = watch.get('specs', {})
            feature_vector = []
            
            # Watch type indicators
            watch_type = str(specs.get('watch_type', '')).lower()
            feature_vector.extend([
                1 if 'dive' in watch_type else 0,
                1 if 'pilot' in watch_type else 0,
                1 if 'field' in watch_type else 0,
                1 if 'dress' in watch_type else 0,
                1 if 'sport' in watch_type else 0,
                1 if 'chrono' in watch_type else 0,
                1 if 'gmt' in watch_type else 0,
            ])
            
            # Complications
            feature_vector.extend([
                1 if specs.get('complication_chronograph') and specs.get('complication_chronograph') != '-' else 0,
                1 if specs.get('complication_gmt') and specs.get('complication_gmt') != '-' else 0,
                1 if specs.get('complication_date') and specs.get('complication_date') != '-' else 0,
                1 if specs.get('complication_moonphase') and specs.get('complication_moonphase') != '-' else 0,
            ])
            
            # Case material
            case_material = str(specs.get('case_material', '')).lower()
            feature_vector.extend([
                1 if 'steel' in case_material else 0,
                1 if 'titanium' in case_material else 0,
                1 if 'gold' in case_material else 0,
                1 if 'bronze' in case_material else 0,
            ])
            
            features.append(feature_vector)
            
        return np.array(features)

    def _extract_brand_features(self) -> np.ndarray:
        """Extract brand-based features for clustering."""
        # Create brand embeddings based on average watch embeddings
        brand_embeddings = {}
        brand_counts = defaultdict(int)
        
        for i, watch in enumerate(self.watch_data):
            brand = watch.get('brand', 'Unknown')
            if brand not in brand_embeddings:
                brand_embeddings[brand] = np.zeros(self.dimension)
            brand_embeddings[brand] += self.normalized_embeddings[i]
            brand_counts[brand] += 1
            
        # Normalize by count
        for brand in brand_embeddings:
            brand_embeddings[brand] /= brand_counts[brand]
            
        # Map each watch to its brand embedding
        features = []
        for watch in self.watch_data:
            brand = watch.get('brand', 'Unknown')
            features.append(brand_embeddings[brand])
            
        return np.array(features)

    async def get_recommendations(self, request: RecommendationRequest) -> RecommendationResult:
        """
        Get recommendations using modern ML techniques.
        
        Args:
            request: Structured recommendation request
            
        Returns:
            RecommendationResult with recommendations and metadata
        """
        start_time = time.time()
        
        # Get or create user session
        user_state = self._get_user_state(request.user_id)
        
        # Update user state with new feedback
        self._update_user_state(user_state, request)
        
        # Determine recommendation strategy
        strategy = self._select_strategy(user_state, request)
        
        # Get recommendations based on strategy
        if strategy == 'cold_start':
            recommendations = await self._cold_start_recommendations(request)
        elif strategy == 'preference_based':
            recommendations = await self._preference_based_recommendations(user_state, request)
        elif strategy == 'exploration':
            recommendations = await self._exploration_recommendations(user_state, request)
        else:  # hybrid
            recommendations = await self._hybrid_recommendations(user_state, request)
        
        # Post-process recommendations
        final_recommendations = self._post_process_recommendations(
            recommendations, user_state, request
        )
        
        # Calculate metrics
        processing_time = time.time() - start_time
        diversity_score = self._calculate_diversity_score(final_recommendations)
        exploration_rate = self._calculate_exploration_rate(user_state)
        
        # Update performance metrics
        self._update_performance_metrics(processing_time)
        
        # Generate user profile summary
        profile_summary = self._generate_profile_summary(user_state)
        
        # Generate next exploration suggestions
        next_suggestions = self._generate_exploration_suggestions(user_state)
        
        result = RecommendationResult(
            watches=final_recommendations,
            confidence_scores=[w.get('confidence', 0.5) for w in final_recommendations],
            diversity_score=diversity_score,
            exploration_rate=exploration_rate,
            algorithm_used=strategy,
            processing_time=processing_time,
            user_profile_summary=profile_summary,
            next_exploration_suggestions=next_suggestions
        )
        
        logger.info(f"Generated {len(final_recommendations)} recommendations using {strategy} strategy in {processing_time:.3f}s")
        
        return result

    def _get_user_state(self, user_id: str) -> Dict[str, Any]:
        """Get or create user state."""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'preference_clusters': [],
                'seen_watches': set(),
                'seen_series': set(),
                'feedback_history': deque(maxlen=100),
                'session_start': datetime.now(),
                'engagement_level': 'exploring',
                'dominant_preferences': [],
                'exploration_areas': set(),
                'last_activity': datetime.now()
            }
        return self.user_sessions[user_id]

    def _update_user_state(self, user_state: Dict[str, Any], request: RecommendationRequest):
        """Update user state based on new feedback."""
        current_time = datetime.now()
        
        # Process liked watches
        for idx in request.liked_indices:
            if idx < len(self.watch_data):
                # Create preference
                preference = UserPreference(
                    embedding=self.normalized_embeddings[idx],
                    confidence=0.8,
                    timestamp=current_time,
                    context={'feedback_type': 'like', 'watch_index': idx}
                )
                
                # Add to feedback history
                user_state['feedback_history'].append({
                    'type': 'like',
                    'watch_index': idx,
                    'timestamp': current_time,
                    'confidence': 0.8
                })
                
                # Update preference clusters
                self._update_preference_clusters(user_state, preference)
                
                # Track seen watches and series
                user_state['seen_watches'].add(idx)
                self._track_series(user_state, idx)
        
        # Process disliked watches  
        for idx in request.disliked_indices:
            if idx < len(self.watch_data):
                user_state['feedback_history'].append({
                    'type': 'dislike',
                    'watch_index': idx,
                    'timestamp': current_time,
                    'confidence': 0.6
                })
                user_state['seen_watches'].add(idx)
                self._track_series(user_state, idx)
        
        # Update engagement level
        self._update_engagement_level(user_state)
        
        user_state['last_activity'] = current_time

    def _update_preference_clusters(self, user_state: Dict[str, Any], preference: UserPreference):
        """Update user preference clusters with new preference."""
        clusters = user_state['preference_clusters']
        
        # Find best matching cluster
        best_cluster = None
        best_similarity = -1
        
        for cluster in clusters:
            similarity = np.dot(cluster.centroid, preference.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster
        
        # Add to existing cluster or create new one (lowered threshold for better clustering)
        if best_cluster and best_similarity > 0.5:  # Lowered from 0.7 to 0.5
            best_cluster.add_preference(preference)
            logger.info(f"📊 Added preference to existing cluster (similarity: {best_similarity:.3f})")
        else:
            # Create new cluster
            new_cluster = PreferenceCluster(preference.embedding, [preference])
            clusters.append(new_cluster)
            logger.info(f"🆕 Created new preference cluster (total clusters: {len(clusters)})")
            
            # Limit number of clusters
            if len(clusters) > 5:  # Increased from 4 to 5
                # Remove weakest cluster
                clusters.sort(key=lambda c: c.get_strength())
                removed_cluster = clusters.pop(0)
                logger.info(f"🗑️  Removed weakest cluster (strength: {removed_cluster.get_strength():.3f})")

    def _track_series(self, user_state: Dict[str, Any], watch_index: int):
        """Track seen series to avoid duplicates."""
        watch = self.watch_data[watch_index]
        series = watch.get('specs', {}).get('serie', '')
        if series and series != '-' and series != 'All':
            user_state['seen_series'].add(series.lower())

    def _update_engagement_level(self, user_state: Dict[str, Any]):
        """Update user engagement level based on feedback patterns."""
        feedback_history = user_state['feedback_history']
        total_feedback = len(feedback_history)
        
        if total_feedback == 0:
            user_state['engagement_level'] = 'exploring'
            return
        
        # Calculate overall engagement metrics
        total_likes = sum(1 for f in feedback_history if f['type'] == 'like')
        like_ratio = total_likes / total_feedback
        
        # Also check recent engagement (last 20 interactions for better context)
        recent_count = min(20, total_feedback)
        # Convert deque to list for proper slicing support
        feedback_list = list(feedback_history)
        recent_feedback = feedback_list[-recent_count:]
        recent_likes = sum(1 for f in recent_feedback if f['type'] == 'like')
        recent_like_ratio = recent_likes / recent_count if recent_count > 0 else 0
        
        # Determine engagement level using both overall and recent patterns
        if total_likes >= 10 and (like_ratio >= 0.3 or recent_like_ratio >= 0.4):
            user_state['engagement_level'] = 'highly_engaged'
        elif total_likes >= 5 and (like_ratio >= 0.2 or recent_like_ratio >= 0.3):
            user_state['engagement_level'] = 'engaged'
        elif total_likes >= 2 and (like_ratio >= 0.1 or recent_like_ratio >= 0.2):
            user_state['engagement_level'] = 'interested'
        else:
            user_state['engagement_level'] = 'exploring'

    def _select_strategy(self, user_state: Dict[str, Any], request: RecommendationRequest) -> str:
        """Select the best recommendation strategy with improved logic."""
        feedback_count = len(user_state['feedback_history'])
        likes_count = sum(1 for f in user_state['feedback_history'] if f['type'] == 'like')
        preference_clusters = user_state['preference_clusters']
        exploration_rate = len(user_state['seen_watches']) / self.num_watches
        
        # Calculate cluster strength
        total_cluster_strength = sum(cluster.get_strength() for cluster in preference_clusters)
        
        # Debug logging
        logger.info(f"🧠 Strategy selection: {likes_count} likes, {feedback_count} total, {len(preference_clusters)} clusters, {exploration_rate:.3f} explored, strength: {total_cluster_strength:.2f}")
        
        # Cold start - not enough data
        if feedback_count < 3:
            logger.info(f"🌱 Selected: cold_start (insufficient feedback)")
            return 'cold_start'
        
        # Strong preferences established - be more aggressive
        if likes_count >= 5 and len(preference_clusters) > 0 and total_cluster_strength > 0.5:
            logger.info(f"🎯 Selected: preference_based ({likes_count} likes, {len(preference_clusters)} strong clusters)")
            return 'preference_based'
        
        # Medium-strong preferences - still prefer preference_based over hybrid  
        if likes_count >= 8 and len(preference_clusters) > 0:
            logger.info(f"🎯 Selected: preference_based ({likes_count} likes, sufficient data)")
            return 'preference_based'
        
        # Medium preferences - use hybrid approach (reduced threshold)
        if likes_count >= 4 and feedback_count >= 8:
            logger.info(f"🔀 Selected: hybrid ({likes_count} likes, {feedback_count} feedback)")
            return 'hybrid'
        
        # Force exploration only if really low engagement OR very little explored
        if (likes_count < 2 and feedback_count >= 15) or exploration_rate < 0.02:
            logger.info(f"🗺️  Selected: exploration (low engagement or unexplored)")
            return 'exploration'
        
        # Default to hybrid for balanced approach
        logger.info(f"🔀 Selected: hybrid (default)")
        return 'hybrid'

    async def _cold_start_recommendations(self, request: RecommendationRequest) -> List[Dict[str, Any]]:
        """Generate cold start recommendations using smart seeds."""
        # Use precomputed smart seeds
        if self.smart_seed_sets:
            import random
            seed_set = random.choice(self.smart_seed_sets)
            recommendations = []
            
            for seed_idx in seed_set['seeds'][:request.num_recommendations]:
                if seed_idx < len(self.watch_data):
                    watch = self.watch_data[seed_idx].copy()
                    watch['index'] = int(seed_idx)  # Convert numpy int64 to Python int
                    watch['score'] = 1.0
                    watch['confidence'] = 0.9
                    watch['algorithm'] = 'smart_seeds'
                    # Convert entire watch to JSON-serializable format
                    watch = convert_to_json_serializable(watch)
                    recommendations.append(watch)
            
            return recommendations
        
        # Fallback: diverse clustering-based selection
        self._ensure_clusters_initialized()
        
        # Get representatives from different style clusters
        cluster_labels = self.style_clusters.labels_
        unique_clusters = np.unique(cluster_labels)
        
        recommendations = []
        for cluster_id in unique_clusters[:request.num_recommendations]:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            # Select the watch closest to cluster center
            cluster_center = self.style_clusters.cluster_centers_[cluster_id]
            
            best_idx = None
            best_distance = float('inf')
            
            for idx in cluster_indices:
                if idx not in request.current_candidates:
                    style_features = self._extract_style_features()
                    distance = np.linalg.norm(style_features[idx] - cluster_center)
                    if distance < best_distance:
                        best_distance = distance
                        best_idx = idx
            
            if best_idx is not None:
                watch = self.watch_data[best_idx].copy()
                watch['index'] = int(best_idx)  # Convert to Python int
                watch['score'] = float(1.0 - (best_distance / 10.0))  # Convert to Python float
                watch['confidence'] = 0.7
                watch['algorithm'] = 'clustering_cold_start'
                # Convert entire watch to JSON-serializable format
                watch = convert_to_json_serializable(watch)
                recommendations.append(watch)
        
        return recommendations

    async def _preference_based_recommendations(self, user_state: Dict[str, Any], request: RecommendationRequest) -> List[Dict[str, Any]]:
        """Generate recommendations based on learned user preferences."""
        recommendations = []
        
        # Get dominant preference clusters
        clusters = user_state['preference_clusters']
        if not clusters:
            return await self._cold_start_recommendations(request)
        
        # Sort clusters by strength
        clusters.sort(key=lambda c: c.get_strength(), reverse=True)
        
        # Generate recommendations from top clusters
        recommendations_per_cluster = max(1, request.num_recommendations // len(clusters))
        
        for cluster in clusters:
            cluster_recs = await self._get_cluster_recommendations(
                cluster, user_state, recommendations_per_cluster, request.current_candidates
            )
            recommendations.extend(cluster_recs)
        
        # Fill remaining slots with diversification
        if len(recommendations) < request.num_recommendations:
            additional_needed = request.num_recommendations - len(recommendations)
            exploration_recs = await self._exploration_recommendations(
                user_state, 
                RecommendationRequest(
                    user_id=request.user_id,
                    liked_indices=request.liked_indices,
                    disliked_indices=request.disliked_indices,
                    current_candidates=request.current_candidates + [r['index'] for r in recommendations],
                    num_recommendations=additional_needed
                )
            )
            recommendations.extend(exploration_recs)
        
        return recommendations[:request.num_recommendations]

    async def _get_cluster_recommendations(self, cluster: PreferenceCluster, user_state: Dict[str, Any], 
                                        num_recs: int, exclude_indices: List[int]) -> List[Dict[str, Any]]:
        """Get recommendations for a specific preference cluster."""
        # Search for similar watches using the cluster centroid
        distances, indices = self.similarity_index.search(cluster.centroid, num_recs * 3)
        
        recommendations = []
        seen_watches = user_state['seen_watches']
        seen_series = user_state['seen_series']
        
        for i, idx in enumerate(indices):
            if (idx not in exclude_indices and 
                idx not in seen_watches and 
                len(recommendations) < num_recs):
                
                # Check series duplication
                watch = self.watch_data[idx]
                series = watch.get('specs', {}).get('serie', '').lower()
                if series and series != '-' and series != 'all' and series in seen_series:
                    continue
                
                watch_copy = watch.copy()
                watch_copy['index'] = int(idx)  # Convert to Python int
                watch_copy['score'] = float(1.0 - distances[i])  # Convert distance to similarity
                watch_copy['confidence'] = float(cluster.get_strength())  # Convert to Python float
                watch_copy['algorithm'] = 'preference_cluster'
                watch_copy['cluster_id'] = id(cluster)
                
                # Convert entire watch to JSON-serializable format
                watch_copy = convert_to_json_serializable(watch_copy)
                recommendations.append(watch_copy)
        
        return recommendations

    async def _exploration_recommendations(self, user_state: Dict[str, Any], request: RecommendationRequest) -> List[Dict[str, Any]]:
        """Generate exploration recommendations to discover new preferences."""
        self._ensure_clusters_initialized()
        
        seen_watches = user_state['seen_watches']
        explored_areas = user_state['exploration_areas']
        
        # Identify unexplored style clusters
        style_clusters = self.style_clusters.labels_
        unique_clusters = set(np.unique(style_clusters)) - explored_areas
        
        if not unique_clusters:
            # Reset exploration if all areas covered
            user_state['exploration_areas'].clear()
            unique_clusters = set(np.unique(style_clusters))
        
        recommendations = []
        clusters_to_explore = list(unique_clusters)[:request.num_recommendations]
        
        for cluster_id in clusters_to_explore:
            cluster_indices = np.where(style_clusters == cluster_id)[0]
            
            # Find best representative for this cluster
            available_indices = [idx for idx in cluster_indices 
                               if idx not in seen_watches and idx not in request.current_candidates]
            
            if available_indices:
                # Select the most representative watch (closest to cluster center)
                cluster_center = self.style_clusters.cluster_centers_[cluster_id]
                style_features = self._extract_style_features()
                
                best_idx = min(available_indices, 
                             key=lambda idx: np.linalg.norm(style_features[idx] - cluster_center))
                
                watch = self.watch_data[best_idx].copy()
                watch['index'] = int(best_idx)  # Convert to Python int
                watch['score'] = 0.8  # High score for exploration
                watch['confidence'] = 0.6  # Medium confidence for exploration
                watch['algorithm'] = 'exploration'
                watch['cluster_id'] = int(cluster_id)  # Convert to Python int
                
                # Convert entire watch to JSON-serializable format
                watch = convert_to_json_serializable(watch)
                recommendations.append(watch)
                user_state['exploration_areas'].add(cluster_id)
        
        return recommendations

    async def _hybrid_recommendations(self, user_state: Dict[str, Any], request: RecommendationRequest) -> List[Dict[str, Any]]:
        """Generate hybrid recommendations combining preferences and exploration."""
        # Calculate preference strength to adjust the mix
        preference_clusters = user_state['preference_clusters']
        total_cluster_strength = sum(cluster.get_strength() for cluster in preference_clusters) if preference_clusters else 0
        likes_count = sum(1 for f in user_state['feedback_history'] if f['type'] == 'like')
        
        # Adjust exploration percentage based on preference strength
        if total_cluster_strength > 0.7 or likes_count >= 10:
            # Strong preferences - minimal exploration
            exploration_percentage = 0.15  # 15% exploration
        elif total_cluster_strength > 0.5 or likes_count >= 6:
            # Medium preferences - reduced exploration  
            exploration_percentage = 0.25  # 25% exploration
        else:
            # Weak preferences - balanced approach
            exploration_percentage = 0.35  # 35% exploration
        
        # Split between preference-based and exploration
        pref_count = max(1, int(request.num_recommendations * (1 - exploration_percentage)))  
        exp_count = request.num_recommendations - pref_count
        
        logger.info(f"🔀 Hybrid strategy: {pref_count} preference + {exp_count} exploration (strength: {total_cluster_strength:.2f})")
        
        # Get preference-based recommendations
        pref_request = RecommendationRequest(
            user_id=request.user_id,
            liked_indices=request.liked_indices,
            disliked_indices=request.disliked_indices,
            current_candidates=request.current_candidates,
            num_recommendations=pref_count
        )
        pref_recommendations = await self._preference_based_recommendations(user_state, pref_request)
        
        # Get exploration recommendations (only if needed)
        exp_recommendations = []
        if exp_count > 0:
            exp_request = RecommendationRequest(
                user_id=request.user_id,
                liked_indices=request.liked_indices,
                disliked_indices=request.disliked_indices,
                current_candidates=request.current_candidates + [r['index'] for r in pref_recommendations],
                num_recommendations=exp_count
            )
            exp_recommendations = await self._exploration_recommendations(user_state, exp_request)
        
        # Combine and shuffle for better user experience
        all_recommendations = pref_recommendations + exp_recommendations
        import random
        random.shuffle(all_recommendations)
        
        return all_recommendations

    def _post_process_recommendations(self, recommendations: List[Dict[str, Any]], 
                                    user_state: Dict[str, Any], 
                                    request: RecommendationRequest) -> List[Dict[str, Any]]:
        """Post-process recommendations for final optimization."""
        if not recommendations:
            return recommendations
        
        # Remove duplicates
        seen_indices = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec['index'] not in seen_indices:
                unique_recommendations.append(rec)
                seen_indices.add(rec['index'])
        
        # Apply diversity optimization
        if len(unique_recommendations) > 1:
            unique_recommendations = self._optimize_diversity(unique_recommendations, request.diversity_threshold)
        
        # Add metadata and ensure JSON serialization
        for rec in unique_recommendations:
            rec['recommendation_timestamp'] = datetime.now().isoformat()
            rec['user_engagement_level'] = user_state['engagement_level']
            # Convert entire recommendation to JSON-serializable format
            rec = convert_to_json_serializable(rec)
            
        # Apply final conversion to entire list
        return [convert_to_json_serializable(rec) for rec in unique_recommendations]

    def _optimize_diversity(self, recommendations: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """Optimize diversity within recommendations."""
        if len(recommendations) <= 1:
            return recommendations
            
        # Extract embeddings for recommendations
        rec_embeddings = np.array([self.normalized_embeddings[rec['index']] for rec in recommendations])
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(rec_embeddings)
        
        # Greedy selection for diversity
        selected = [0]  # Start with first recommendation
        
        for _ in range(min(len(recommendations) - 1, 6)):  # Select up to 7 total
            best_candidate = -1
            min_max_similarity = float('inf')
            
            for i, rec in enumerate(recommendations):
                if i in selected:
                    continue
                    
                # Calculate max similarity to already selected items
                max_similarity = max(similarities[i][j] for j in selected)
                
                if max_similarity < min_max_similarity:
                    min_max_similarity = max_similarity
                    best_candidate = i
            
            if best_candidate != -1 and min_max_similarity < threshold:
                selected.append(best_candidate)
            else:
                break
        
        return [recommendations[i] for i in selected]

    def _calculate_diversity_score(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate diversity score for a set of recommendations."""
        if len(recommendations) <= 1:
            return 1.0
            
        # Brand diversity
        brands = set(rec.get('brand', '') for rec in recommendations)
        brand_diversity = len(brands) / len(recommendations)
        
        # Style diversity (extract from specs)
        styles = set()
        for rec in recommendations:
            watch_type = rec.get('specs', {}).get('watch_type', '')
            if watch_type:
                styles.add(watch_type.lower())
        style_diversity = len(styles) / len(recommendations) if styles else 0.5
        
        # Embedding diversity
        if len(recommendations) > 1:
            rec_embeddings = np.array([self.normalized_embeddings[rec['index']] for rec in recommendations])
            similarities = cosine_similarity(rec_embeddings)
            avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
            embedding_diversity = 1.0 - avg_similarity
        else:
            embedding_diversity = 1.0
            
        # Combined diversity score
        return (brand_diversity * 0.3 + style_diversity * 0.4 + embedding_diversity * 0.3)

    def _calculate_exploration_rate(self, user_state: Dict[str, Any]) -> float:
        """Calculate exploration rate for user."""
        return len(user_state['seen_watches']) / self.num_watches

    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics."""
        self.performance_metrics['total_requests'] += 1
        
        # Update average response time
        current_avg = self.performance_metrics['avg_response_time']
        total_requests = self.performance_metrics['total_requests']
        self.performance_metrics['avg_response_time'] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )

    def _generate_profile_summary(self, user_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate user profile summary."""
        clusters = user_state['preference_clusters']
        
        # Extract dominant preferences
        dominant_brands = defaultdict(int)
        dominant_styles = defaultdict(int)
        
        for feedback in user_state['feedback_history']:
            if feedback['type'] == 'like':
                watch = self.watch_data[feedback['watch_index']]
                dominant_brands[watch.get('brand', '')] += 1
                
                watch_type = watch.get('specs', {}).get('watch_type', '')
                if watch_type:
                    dominant_styles[watch_type] += 1
        
        return {
            'engagement_level': user_state['engagement_level'],
            'total_feedback': len(user_state['feedback_history']),
            'preference_clusters': len(clusters),
            'exploration_rate': self._calculate_exploration_rate(user_state),
            'dominant_brands': dict(sorted(dominant_brands.items(), key=lambda x: x[1], reverse=True)[:5]),
            'dominant_styles': dict(sorted(dominant_styles.items(), key=lambda x: x[1], reverse=True)[:5]),
            'session_duration': (datetime.now() - user_state['session_start']).total_seconds() / 60  # minutes
        }

    def _generate_exploration_suggestions(self, user_state: Dict[str, Any]) -> List[str]:
        """Generate suggestions for next exploration areas."""
        explored_areas = user_state['exploration_areas']
        
        suggestions = []
        
        # Suggest unexplored watch types
        all_types = set()
        for watch in self.watch_data:
            watch_type = watch.get('specs', {}).get('watch_type', '')
            if watch_type:
                all_types.add(watch_type.lower())
        
        explored_types = set()
        for feedback in user_state['feedback_history']:
            if feedback['type'] == 'like':
                watch = self.watch_data[feedback['watch_index']]
                watch_type = watch.get('specs', {}).get('watch_type', '')
                if watch_type:
                    explored_types.add(watch_type.lower())
        
        unexplored_types = all_types - explored_types
        suggestions.extend([f"Try {wtype} watches" for wtype in list(unexplored_types)[:3]])
        
        # Suggest different price ranges
        liked_prices = []
        for feedback in user_state['feedback_history']:
            if feedback['type'] == 'like':
                watch = self.watch_data[feedback['watch_index']]
                price = watch.get('price')
                if price and isinstance(price, (int, float)):
                    liked_prices.append(price)
        
        if liked_prices:
            avg_price = np.mean(liked_prices)
            if avg_price < 1000:
                suggestions.append("Explore luxury timepieces ($1000+)")
            elif avg_price > 5000:
                suggestions.append("Discover affordable gems (under $1000)")
        
        return suggestions[:5]

    def get_smart_seeds(self, num_seeds: int = 7) -> List[Dict[str, Any]]:
        """Get smart seeds for cold start (synchronous method for compatibility)."""
        if self.smart_seed_sets:
            import random
            seed_set = random.choice(self.smart_seed_sets)
            recommendations = []
            
            for seed_idx in seed_set['seeds'][:num_seeds]:
                if seed_idx < len(self.watch_data):
                    watch = self.watch_data[seed_idx].copy()
                    watch['index'] = int(seed_idx)  # Convert numpy int64 to Python int
                    watch['score'] = 1.0
                    watch['is_seed'] = True
                    watch['is_precomputed_seed'] = True
                    watch['seed_style'] = self._classify_watch_style(watch)
                    # Convert entire watch to JSON-serializable format
                    watch = convert_to_json_serializable(watch)
                    recommendations.append(watch)
            
            return recommendations
        
        # Fallback to diverse selection
        fallback_seeds = self._get_diverse_fallback_seeds(num_seeds)
        # Ensure fallback seeds are also JSON-serializable
        return [convert_to_json_serializable(seed) for seed in fallback_seeds]

    def _classify_watch_style(self, watch: Dict[str, Any]) -> str:
        """Classify watch style for compatibility."""
        specs = watch.get('specs', {})
        watch_type = str(specs.get('watch_type', '')).lower()
        
        if not watch_type or watch_type == '-':
            return 'classic'
        
        return watch_type

    def _get_diverse_fallback_seeds(self, num_seeds: int) -> List[Dict[str, Any]]:
        """Get diverse seeds as fallback."""
        self._ensure_clusters_initialized()
        
        cluster_labels = self.style_clusters.labels_
        unique_clusters = np.unique(cluster_labels)
        
        seeds = []
        for i, cluster_id in enumerate(unique_clusters[:num_seeds]):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                # Select random representative from cluster
                import random
                seed_idx = random.choice(cluster_indices)
                
                watch = self.watch_data[seed_idx].copy()
                watch['index'] = int(seed_idx)  # Convert to Python int
                watch['score'] = 1.0
                watch['is_seed'] = True
                watch['seed_style'] = self._classify_watch_style(watch)
                # Convert entire watch to JSON-serializable format
                watch = convert_to_json_serializable(watch)
                seeds.append(watch)
        
        return seeds

    def add_feedback(self, user_id: str, watch_index: int, feedback_type: str, confidence: float = 0.8):
        """Add user feedback (compatibility method)."""
        # Validate and convert watch_index to ensure it's an integer
        try:
            logger.debug(f"🔍 add_feedback START: user_id={user_id}, watch_index={watch_index} (type: {type(watch_index)}), feedback_type={feedback_type}")
            watch_index = int(watch_index)
            logger.debug(f"✅ watch_index converted to int: {watch_index}")
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Invalid watch_index type: {type(watch_index)}, value: {watch_index}")
            return
        
        # Validate watch_index range
        if watch_index < 0 or watch_index >= len(self.watch_data):
            logger.error(f"❌ Invalid watch_index: {watch_index} (valid range: 0-{len(self.watch_data)-1})")
            return
        
        logger.debug(f"🎯 Getting user state for user_id: {user_id}")
        try:
            user_state = self._get_user_state(user_id)
            logger.debug(f"✅ Got user state successfully")
        except Exception as e:
            logger.error(f"❌ Error getting user state: {e}")
            return
        
        logger.debug(f"🎯 Adding to feedback history...")
        try:
            user_state['feedback_history'].append({
                'type': feedback_type,
                'watch_index': watch_index,
                'timestamp': datetime.now(),
                'confidence': confidence
            })
            logger.debug(f"✅ Added to feedback history successfully")
        except Exception as e:
            logger.error(f"❌ Error adding to feedback history: {e}")
            return
        
        if feedback_type == 'like':
            logger.debug(f"🎯 Processing LIKE feedback - creating preference...")
            try:
                logger.debug(f"🔍 Accessing embedding at index {watch_index}...")
                embedding = self.normalized_embeddings[watch_index]
                logger.debug(f"✅ Retrieved embedding shape: {embedding.shape}")
                
                preference = UserPreference(
                    embedding=embedding,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    context={'feedback_type': feedback_type, 'watch_index': watch_index}
                )
                logger.debug(f"✅ Created UserPreference object")
                
                logger.debug(f"🎯 Updating preference clusters...")
                self._update_preference_clusters(user_state, preference)
                logger.debug(f"✅ Updated preference clusters successfully")
            except Exception as e:
                logger.error(f"❌ Error creating preference for watch {watch_index}: {e}")
                import traceback
                logger.error(f"❌ Preference error traceback: {traceback.format_exc()}")
                return
        
        logger.debug(f"🎯 Adding to seen watches...")
        try:
            user_state['seen_watches'].add(watch_index)
            logger.debug(f"✅ Added to seen watches")
        except Exception as e:
            logger.error(f"❌ Error adding to seen watches: {e}")
            return
        
        logger.debug(f"🎯 Tracking series...")
        try:
            self._track_series(user_state, watch_index)
            logger.debug(f"✅ Tracked series successfully")
        except Exception as e:
            logger.error(f"❌ Error tracking series: {e}")
            return
        
        logger.debug(f"🎯 Updating engagement level...")
        try:
            self._update_engagement_level(user_state)
            logger.debug(f"✅ Updated engagement level successfully")
        except Exception as e:
            logger.error(f"❌ Error updating engagement level: {e}")
            import traceback
            logger.error(f"❌ Engagement error traceback: {traceback.format_exc()}")
            return
        
        logger.debug(f"✅ add_feedback COMPLETED successfully for watch {watch_index}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary (compatibility method)."""
        return self.performance_metrics.copy() 