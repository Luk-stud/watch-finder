"""
Enhanced Beam Search v2 for Watch Recommendations
Improvements over v1:
- Multi-objective optimization (similarity + diversity + novelty + style coherence)
- Dynamic beam width based on user engagement
- Hierarchical search (brand/style/complications levels)
- Better text embedding utilization
- Contextual bandits for exploration/exploitation
- Active learning feedback incorporation
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
import time
from datetime import datetime
import sys
import os
from collections import defaultdict, deque
from sklearn.metrics import silhouette_score
import pickle
import re

# Add current directory to path for variant_detector import
sys.path.append(os.path.dirname(__file__))
from variant_detector import WatchVariantDetector

class EnhancedWatchBeamSearch:
    def __init__(self, embeddings: np.ndarray, watch_data: List[Dict[str, Any]], 
                 initial_beam_width: int = 15, max_beam_width: int = 30,
                 embeddings_pre_normalized: bool = False):
        """
        Enhanced beam search with multi-objective optimization and multi-modal preference handling.
        Optimized initialization - expensive operations are lazy-loaded when needed.
        
        Args:
            embeddings: Watch text embeddings (1536D from OpenAI)
            watch_data: Watch metadata with AI descriptions
            initial_beam_width: Starting beam width
            max_beam_width: Maximum adaptive beam width
            embeddings_pre_normalized: If True, embeddings are already normalized (skip normalization)
        """
        self.embeddings = embeddings
        self.watch_data = watch_data
        self.initial_beam_width = initial_beam_width
        self.max_beam_width = max_beam_width
        self.current_beam_width = initial_beam_width
        
        # Enhanced tracking
        self.seen_watches = set()
        self.seen_series = set()
        self.seen_brands = set()
        self.seen_styles = set()
        
        # 🚀 OPTIMIZED EMBEDDING NORMALIZATION
        self.dimension = embeddings.shape[1]
        if embeddings_pre_normalized:
            # Embeddings already normalized at server startup - just use them!
            self.normalized_embeddings = embeddings
            print("⚡ Using pre-normalized embeddings (server startup optimization)")
        else:
            # Fallback: normalize embeddings (for backward compatibility)
            self.normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            print("🔧 Normalized embeddings during session creation (consider pre-normalizing at startup)")
        
        # Multi-objective weights (dynamically adjusted based on user behavior)
        self.objective_weights = {
            'similarity': 0.4,      # Similarity to liked watches
            'diversity': 0.25,      # Diversity within recommendation batch
            'novelty': 0.2,         # Unexplored brands/styles
            'style_coherence': 0.15 # Coherent style progression
        }
        
        # Enhanced feedback system
        self.feedback_history = deque(maxlen=100)  # Rolling window
        self.user_profile = self._initialize_user_profile()
        
        # 🆕 MULTI-MODAL PREFERENCE SYSTEM
        self.preference_modes = []  # List of distinct preference vectors
        self.mode_weights = []      # Confidence/strength of each mode
        self.mode_contexts = []     # Context/labels for each mode
        self.modal_threshold = 0.3  # Similarity threshold for creating new modes
        self.max_modes = 4          # Maximum number of preference modes
        self.mode_update_frequency = 3  # Re-cluster every N feedback items
        
        # Contextual bandit for exploration/exploitation
        self.exploration_factor = 0.3  # Start exploratory
        self.exploitation_threshold = 5  # Switch after 5 positive feedbacks
        
        # Hierarchical search levels
        self.search_levels = ['brand', 'style', 'complications', 'semantic']
        self.current_search_level = 'semantic'  # Start with semantic similarity
        
        # 🆕 LAZY INITIALIZATION FLAGS
        self._semantic_features_initialized = False
        self._clusters_initialized = False
        self._similarity_matrix_initialized = False
        self._variant_detector_initialized = False
        
        # Placeholders for lazy-loaded components
        self.semantic_features = {}
        self.cluster_mappings = {}
        self.similarity_matrix = None
        self.variant_detector = None
        
        # 🆕 Load precomputed smart seeds (fast operation)
        self._load_precomputed_smart_seeds()
        
        # 🆕 VARIANT FILTERING SETUP (Similar to ModernRecommendationEngine)
        self._initialize_variant_filter()
        
        # Performance tracking
        self.performance_metrics = {
            'recommendation_quality': deque(maxlen=20),
            'user_engagement': deque(maxlen=20),
            'diversity_scores': deque(maxlen=20),
            'novelty_scores': deque(maxlen=20),
            'modal_diversity': deque(maxlen=20)  # Track how well we serve different modes
        }
        
        print("🚀 Fast initialization complete - expensive operations will be lazy-loaded when needed")
        
    def _initialize_user_profile(self) -> Dict:
        """Initialize comprehensive user profile for personalization."""
        return {
            'brand_preferences': defaultdict(float),
            'style_preferences': defaultdict(float),
            'price_range': {'min': 0, 'max': float('inf')},
            'complication_preferences': defaultdict(float),
            'aesthetic_preferences': defaultdict(float),
            'engagement_level': 'exploring',  # exploring, focused, decided
            'expertise_level': 'novice',      # novice, intermediate, expert
            'decision_speed': 'moderate'      # fast, moderate, slow
        }
    
    def _ensure_semantic_features(self):
        """Lazy initialization of semantic features."""
        if not self._semantic_features_initialized:
            print("⏳ Initializing semantic features (first time)...")
            self._extract_semantic_features()
            self._semantic_features_initialized = True
            print("✅ Semantic features initialized")
    
    def _ensure_clusters(self):
        """Lazy initialization of clustering."""
        if not self._clusters_initialized:
            print("⏳ Initializing enhanced clusters (first time)...")
            self._initialize_enhanced_clusters()
            self._clusters_initialized = True
            print("✅ Enhanced clusters initialized")
    
    def _ensure_similarity_matrix(self):
        """Lazy initialization of similarity matrix."""
        if not self._similarity_matrix_initialized:
            print("⏳ Computing similarity matrix (first time)...")
            self.similarity_matrix = np.dot(self.normalized_embeddings, self.normalized_embeddings.T)
            self._similarity_matrix_initialized = True
            print("✅ Similarity matrix computed")
    
    def _ensure_variant_detector(self):
        """Lazy initialization of variant detector."""
        if not self._variant_detector_initialized:
            print("⏳ Initializing variant detector (first time)...")
            self.variant_detector = WatchVariantDetector(self.watch_data, self.normalized_embeddings)
            self._variant_detector_initialized = True
            print("✅ Variant detector initialized")
    
    def _extract_semantic_features(self):
        """Extract semantic features from AI descriptions for enhanced matching."""
        self.semantic_features = {}
        
        for i, watch in enumerate(self.watch_data):
            description = watch.get('ai_description', '')
            
            # Extract aesthetic keywords from AI descriptions
            aesthetic_keywords = self._extract_aesthetic_keywords(description)
            brand = watch.get('brand', '').lower()
            style = self._classify_watch_style_enhanced(watch)
            
            self.semantic_features[i] = {
                'aesthetic_keywords': aesthetic_keywords,
                'brand_category': self._categorize_brand(brand),
                'style_family': style,
                'luxury_level': self._assess_luxury_level(watch),
                'target_demographic': self._identify_target_demographic(description)
            }
    
    def _extract_aesthetic_keywords(self, description: str) -> List[str]:
        """Extract aesthetic keywords from AI-generated descriptions."""
        aesthetic_terms = [
            'vintage', 'modern', 'elegant', 'sporty', 'rugged', 'sophisticated',
            'minimalist', 'bold', 'classic', 'contemporary', 'luxurious', 'casual',
            'refined', 'adventurous', 'timeless', 'distinctive', 'harmonious'
        ]
        
        description_lower = description.lower()
        found_keywords = [term for term in aesthetic_terms if term in description_lower]
        return found_keywords
    
    def _classify_watch_style_enhanced(self, watch: Dict) -> str:
        """Enhanced style classification using AI description and specs."""
        specs = watch.get('specs', {})
        description = watch.get('ai_description', '').lower()
        
        # Primary classification from watch type
        watch_type = specs.get('watch_type', '').lower()
        if watch_type:
            return watch_type
        
        # Secondary classification from description analysis
        if any(term in description for term in ['dive', 'water', 'ocean', 'underwater']):
            return 'diver'
        elif any(term in description for term in ['dress', 'formal', 'elegant']):
            return 'dress'
        elif any(term in description for term in ['sport', 'athletic', 'performance']):
            return 'sport'
        elif any(term in description for term in ['pilot', 'aviation', 'field']):
            return 'field'
        else:
            return 'casual'
    
    def _categorize_brand(self, brand: str) -> str:
        """Categorize brand into luxury tiers."""
        luxury_brands = ['rolex', 'patek philippe', 'audemars piguet', 'vacheron constantin']
        premium_brands = ['omega', 'breitling', 'tag heuer', 'tudor']
        accessible_brands = ['seiko', 'citizen', 'casio', 'timex']
        
        brand_lower = brand.lower()
        if any(b in brand_lower for b in luxury_brands):
            return 'luxury'
        elif any(b in brand_lower for b in premium_brands):
            return 'premium'
        elif any(b in brand_lower for b in accessible_brands):
            return 'accessible'
        else:
            return 'independent'  # Independent/microbrand
    
    def _assess_luxury_level(self, watch: Dict) -> float:
        """Assess luxury level from 0-1 based on price and brand."""
        price = watch.get('price', 0)
        if isinstance(price, str):
            return 0.5  # Unknown price
        
        # Normalize price to luxury scale (rough heuristic)
        if price < 200:
            return 0.1
        elif price < 500:
            return 0.3
        elif price < 1000:
            return 0.5
        elif price < 2000:
            return 0.7
        else:
            return 0.9
    
    def _identify_target_demographic(self, description: str) -> str:
        """Identify target demographic from AI description."""
        description_lower = description.lower()
        
        if any(term in description_lower for term in ['adventure', 'outdoor', 'active']):
            return 'adventurer'
        elif any(term in description_lower for term in ['professional', 'business', 'executive']):
            return 'professional'
        elif any(term in description_lower for term in ['collector', 'enthusiast', 'heritage']):
            return 'collector'
        elif any(term in description_lower for term in ['casual', 'everyday', 'lifestyle']):
            return 'casual_wearer'
        else:
            return 'general'
    
    def _initialize_enhanced_clusters(self):
        """Initialize multi-level clustering for hierarchical search."""
        try:
            # Style-based clustering (20 clusters)
            n_style = min(20, max(2, len(self.normalized_embeddings)//10))
            self.style_clusters = KMeans(n_clusters=n_style, random_state=42, n_init=10).fit_predict(self.normalized_embeddings)
            
            # Brand-based clustering (15 clusters)
            brand_features = self._create_brand_feature_matrix()
            n_brand = min(15, max(2, len(set(w['brand'] for w in self.watch_data))))
            self.brand_clusters = KMeans(n_clusters=n_brand, random_state=43, n_init=10).fit_predict(brand_features)
            
            # Aesthetic clustering using description embeddings (10 clusters)
            aesthetic_features = self._create_aesthetic_feature_matrix()
            n_aesthetic = min(10, max(2, len(aesthetic_features)//20))
            self.aesthetic_clusters = KMeans(n_clusters=n_aesthetic, random_state=44, n_init=10).fit_predict(aesthetic_features)
            
            # Create cluster mappings
            self.cluster_mappings = {
                'style': defaultdict(list),
                'brand': defaultdict(list),
                'aesthetic': defaultdict(list)
            }
            
            for i in range(len(self.watch_data)):
                self.cluster_mappings['style'][self.style_clusters[i]].append(i)
                self.cluster_mappings['brand'][self.brand_clusters[i]].append(i)
                self.cluster_mappings['aesthetic'][self.aesthetic_clusters[i]].append(i)
            
            print(f"Enhanced clustering: {n_style} style + {n_brand} brand + {n_aesthetic} aesthetic clusters")
            
        except Exception as e:
            print(f"Error in enhanced clustering: {e}")
            # Fallback to simple clustering
            self.style_clusters = np.zeros(len(self.watch_data), dtype=int)
            self.brand_clusters = np.zeros(len(self.watch_data), dtype=int)
            self.aesthetic_clusters = np.zeros(len(self.watch_data), dtype=int)
            self.cluster_mappings = {
                'style': {0: list(range(len(self.watch_data)))},
                'brand': {0: list(range(len(self.watch_data)))},
                'aesthetic': {0: list(range(len(self.watch_data)))}
            }
    
    def _create_brand_feature_matrix(self) -> np.ndarray:
        """Create feature matrix based on brand characteristics."""
        features = []
        for watch in self.watch_data:
            brand = watch.get('brand', '').lower()
            brand_category = self._categorize_brand(brand)
            luxury_level = self._assess_luxury_level(watch)
            
            # One-hot encode brand category + luxury level
            feature_vector = [0, 0, 0, 0, luxury_level]  # 4 categories + luxury
            if brand_category == 'luxury':
                feature_vector[0] = 1
            elif brand_category == 'premium':
                feature_vector[1] = 1
            elif brand_category == 'accessible':
                feature_vector[2] = 1
            elif brand_category == 'independent':
                feature_vector[3] = 1
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _create_aesthetic_feature_matrix(self) -> np.ndarray:
        """Create feature matrix based on aesthetic keywords."""
        aesthetic_terms = [
            'vintage', 'modern', 'elegant', 'sporty', 'rugged', 'sophisticated',
            'minimalist', 'bold', 'classic', 'contemporary', 'luxurious', 'casual'
        ]
        
        features = []
        for i, watch in enumerate(self.watch_data):
            description = watch.get('ai_description', '').lower()
            feature_vector = [1 if term in description else 0 for term in aesthetic_terms]
            features.append(feature_vector)
        
        return np.array(features)
    
    def add_feedback(self, watch_index: int, feedback_type: str, confidence: float = 1.0):
        """Enhanced feedback with user profile updates, adaptive beam width, and multi-modal preference learning."""
        feedback = {
            'watch_index': watch_index,
            'type': feedback_type,
            'confidence': confidence,
            'timestamp': time.time(),
            'context': self._capture_feedback_context(watch_index)
        }
        
        self.feedback_history.append(feedback)
        self._update_user_profile(watch_index, feedback_type, confidence)
        
        # 🆕 MULTI-MODAL PREFERENCE UPDATE
        if feedback_type == 'like':
            self._update_preference_modes(watch_index, confidence)
        
        self._adapt_beam_width()
        self._update_objective_weights()
        self._adjust_exploration_factor()
    
    def _capture_feedback_context(self, watch_index: int) -> Dict:
        """Capture context around feedback for better learning."""
        watch = self.watch_data[watch_index]
        return {
            'brand': watch.get('brand'),
            'style': self._classify_watch_style_enhanced(watch),
            'price': watch.get('price'),
            'aesthetic_keywords': self.semantic_features[watch_index]['aesthetic_keywords'],
            'session_position': len(self.feedback_history)
        }
    
    def _update_user_profile(self, watch_index: int, feedback_type: str, confidence: float):
        """Update user profile based on feedback."""
        watch = self.watch_data[watch_index]
        weight = confidence * (1.0 if feedback_type == 'like' else -0.5)
        
        # Update brand preferences
        brand = watch.get('brand', '').lower()
        self.user_profile['brand_preferences'][brand] += weight
        
        # Update style preferences
        style = self._classify_watch_style_enhanced(watch)
        self.user_profile['style_preferences'][style] += weight
        
        # Update aesthetic preferences (lazy load semantic features if needed)
        self._ensure_semantic_features()
        aesthetic_keywords = self.semantic_features[watch_index]['aesthetic_keywords']
        for keyword in aesthetic_keywords:
            self.user_profile['aesthetic_preferences'][keyword] += weight
        
        # Update engagement level
        recent_likes = sum(1 for f in list(self.feedback_history)[-5:] if f['type'] == 'like')
        if recent_likes >= 3:
            self.user_profile['engagement_level'] = 'focused'
        elif recent_likes >= 2:
            self.user_profile['engagement_level'] = 'interested'
        else:
            self.user_profile['engagement_level'] = 'exploring'
    
    def _adapt_beam_width(self):
        """Dynamically adjust beam width based on user engagement."""
        engagement = self.user_profile['engagement_level']
        
        if engagement == 'focused':
            # User is focused, use narrower beam for precision
            self.current_beam_width = max(self.initial_beam_width - 3, 8)
        elif engagement == 'interested':
            # User showing interest, moderate beam width
            self.current_beam_width = self.initial_beam_width
        else:
            # User exploring, use wider beam for discovery
            self.current_beam_width = min(self.initial_beam_width + 5, self.max_beam_width)
    
    def _update_objective_weights(self):
        """Adapt objective weights based on user behavior patterns."""
        recent_feedback = list(self.feedback_history)[-10:]
        
        # If user likes diverse styles, increase diversity weight
        liked_styles = set()
        for f in recent_feedback:
            if f['type'] == 'like':
                style = f['context']['style']
                liked_styles.add(style)
        
        style_diversity = len(liked_styles) / max(len([f for f in recent_feedback if f['type'] == 'like']), 1)
        
        if style_diversity > 0.7:  # User likes diverse styles
            self.objective_weights['diversity'] = min(0.35, self.objective_weights['diversity'] + 0.05)
            self.objective_weights['similarity'] = max(0.25, self.objective_weights['similarity'] - 0.05)
        else:  # User has focused preferences
            self.objective_weights['similarity'] = min(0.5, self.objective_weights['similarity'] + 0.05)
            self.objective_weights['diversity'] = max(0.15, self.objective_weights['diversity'] - 0.05)
    
    def _adjust_exploration_factor(self):
        """Adjust exploration vs exploitation based on feedback patterns."""
        recent_likes = sum(1 for f in list(self.feedback_history)[-5:] if f['type'] == 'like')
        
        if recent_likes >= self.exploitation_threshold:
            # User found preferences, focus on exploitation
            self.exploration_factor = max(0.1, self.exploration_factor - 0.05)
        elif recent_likes == 0:
            # No recent likes, increase exploration
            self.exploration_factor = min(0.5, self.exploration_factor + 0.05)
    
    def multi_objective_score(self, watch_idx: int, user_preferences: np.ndarray, 
                            current_batch: List[int]) -> float:
        """
        Calculate multi-objective score combining:
        - Semantic similarity to user preferences
        - Diversity within current batch
        - Novelty (unexplored brands/styles)
        - Style coherence with user profile
        """
        
        # Ensure semantic features are available
        self._ensure_semantic_features()
        
        # 1. Similarity Score
        similarity = np.dot(self.normalized_embeddings[watch_idx], user_preferences)
        
        # 2. Diversity Score (how different from current batch)
        diversity = 0.0
        if current_batch:
            batch_embeddings = self.normalized_embeddings[current_batch]
            watch_embedding = self.normalized_embeddings[watch_idx].reshape(1, -1)
            similarities_to_batch = cosine_similarity(watch_embedding, batch_embeddings)[0]
            diversity = 1.0 - np.mean(similarities_to_batch)
        else:
            diversity = 1.0
        
        # 3. Novelty Score (brand/style exploration bonus)
        watch = self.watch_data[watch_idx]
        brand = watch.get('brand', '').lower()
        style = self._classify_watch_style_enhanced(watch)
        
        brand_novelty = 0.5 if brand not in self.seen_brands else 0.0
        style_novelty = 0.5 if style not in self.seen_styles else 0.0
        novelty = (brand_novelty + style_novelty) / 2
        
        # 4. Style Coherence (alignment with user profile)
        brand_pref = self.user_profile['brand_preferences'].get(brand, 0)
        style_pref = self.user_profile['style_preferences'].get(style, 0)
        
        aesthetic_keywords = self.semantic_features[watch_idx]['aesthetic_keywords']
        aesthetic_score = sum(self.user_profile['aesthetic_preferences'].get(kw, 0) 
                            for kw in aesthetic_keywords)
        
        style_coherence = (brand_pref + style_pref + aesthetic_score) / 3
        style_coherence = max(0, min(1, (style_coherence + 1) / 2))  # Normalize to [0,1]
        
        # Combine objectives with current weights
        final_score = (
            self.objective_weights['similarity'] * similarity +
            self.objective_weights['diversity'] * diversity +
            self.objective_weights['novelty'] * novelty +
            self.objective_weights['style_coherence'] * style_coherence
        )
        
        return final_score
    
    def enhanced_beam_search_step(self, current_candidates: List[int], 
                                user_preferences: np.ndarray, step: int = 0) -> List[Dict[str, Any]]:
        """
        Enhanced beam search step with multi-objective optimization and hierarchical search.
        """
        step_start_time = time.time()
        
        # 🆕 ROBUST DUPLICATE PREVENTION - Start with all unseen watches
        all_indices = set(range(len(self.watch_data)))
        unseen_candidates = list(all_indices - self.seen_watches)
        
        print(f"Step {step}: Total watches: {len(self.watch_data)}, Seen: {len(self.seen_watches)}, Unseen: {len(unseen_candidates)}")
        
        # If we're running out of unseen watches, expand the pool
        if len(unseen_candidates) < self.current_beam_width * 2:
            print(f"⚠️  Low unseen watch count ({len(unseen_candidates)}), expanding search pool...")
            # Allow some previously seen watches but with lower priority
            recently_seen = set(list(self.seen_watches)[-50:])  # Last 50 seen watches
            expanded_candidates = list(all_indices - recently_seen)
            if len(expanded_candidates) > len(unseen_candidates):
                unseen_candidates = expanded_candidates
                print(f"📈 Expanded pool to {len(unseen_candidates)} watches")
        
        # Score all candidates using multi-objective function
        candidate_scores = []
        for idx in unseen_candidates:
            if 0 <= idx < len(self.watch_data):
                # 🆕 ADDITIONAL DUPLICATE CHECK - Skip if already seen
                if idx in self.seen_watches:
                    continue
                    
                score = self.multi_objective_score(idx, user_preferences, current_candidates)
                candidate_scores.append((idx, score))
        
        print(f"Step {step}: Scored {len(candidate_scores)} valid candidates")
        
        # Sort by score and select top candidates
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply exploration bonus randomly to some candidates
        if random.random() < self.exploration_factor:
            # Occasionally promote a lower-scored but novel candidate
            exploration_boost = int(len(candidate_scores) * 0.3)
            if exploration_boost > 0:
                boosted_idx = random.randint(self.current_beam_width, 
                                           min(self.current_beam_width + exploration_boost, 
                                               len(candidate_scores) - 1))
                # Swap with a top candidate
                if boosted_idx < len(candidate_scores):
                    candidate_scores[self.current_beam_width//2] = candidate_scores[boosted_idx]
        
        # Select top candidates
        selected_candidates = candidate_scores[:self.current_beam_width]
        
        # Apply variant filtering to reduce redundancy
        try:
            self._ensure_variant_detector()
            filtered_candidates = self.variant_detector.filter_diverse_watches(
                [idx for idx, _ in selected_candidates])
        except:
            filtered_candidates = [idx for idx, _ in selected_candidates]
        
        # 🆕 APPLY ADDITIONAL VARIANT FILTERING
        # Convert indices to watch objects for filtering
        candidate_watches_for_filtering = []
        for idx in filtered_candidates:
            if 0 <= idx < len(self.watch_data):
                watch = self.watch_data[idx].copy()
                watch['index'] = idx
                candidate_watches_for_filtering.append(watch)
        
        # Apply our efficient variant filtering
        candidate_watches_for_filtering = self._filter_variant_duplicates_beam(candidate_watches_for_filtering)
        
        # Extract final candidate indices
        final_candidates = [w['index'] for w in candidate_watches_for_filtering]
        
        # 🆕 FINAL DUPLICATE VALIDATION - Ensure no duplicates in results
        final_unique_candidates = []
        for idx in final_candidates:
            if idx not in self.seen_watches and idx not in final_unique_candidates:
                final_unique_candidates.append(idx)
        
        # Build result watches with enhanced duplicate prevention
        result_watches = []
        added_indices = set()
        
        for idx in final_unique_candidates[:min(10, len(final_unique_candidates))]:
            if 0 <= idx < len(self.watch_data) and idx not in added_indices:
                # 🆕 TRIPLE CHECK - Final validation before adding
                if idx in self.seen_watches:
                    print(f"⚠️  Skipping duplicate watch {idx} that was already seen")
                    continue
                    
                watch = self.watch_data[idx].copy()
                watch['index'] = idx
                watch['score'] = next((score for i, score in selected_candidates if i == idx), 0.0)
                
                # 🆕 ADD UNIQUENESS METADATA
                watch['recommendation_step'] = step
                watch['seen_count'] = len(self.seen_watches)
                watch['variant_filtered'] = True  # Indicate this went through variant filtering
                
                result_watches.append(watch)
                added_indices.add(idx)
                
                # 🆕 IMMEDIATELY ADD TO SEEN WATCHES to prevent duplicates in same batch
                self.seen_watches.add(idx)
                
                # Update tracking
                brand = watch.get('brand', '').lower()
                style = self._classify_watch_style_enhanced(watch)
                self.seen_brands.add(brand)
                self.seen_styles.add(style)
        
        print(f"Step {step}: Selected {len(result_watches)} unique watches, total seen now: {len(self.seen_watches)}")
        
        # Update performance metrics
        step_time = time.time() - step_start_time
        self.performance_metrics['recommendation_quality'].append(
            np.mean([w.get('score', 0) for w in result_watches]))
        
        return result_watches
    
    def reset_seen_watches(self):
        """Reset the seen watches tracking - useful when user wants fresh recommendations."""
        print(f"🔄 Resetting seen watches: was {len(self.seen_watches)}, now 0")
        self.seen_watches.clear()
        self.seen_brands.clear()
        self.seen_styles.clear()
        self.seen_series.clear()
        
        # Reset performance metrics
        for metric in self.performance_metrics.values():
            metric.clear()
        
        print("✅ Seen watches reset complete")
    
    def get_exploration_stats(self) -> Dict[str, Any]:
        """Get exploration statistics for monitoring recommendation diversity."""
        total_watches = len(self.watch_data)
        seen_count = len(self.seen_watches)
        exploration_percentage = (seen_count / total_watches * 100) if total_watches > 0 else 0
        
        return {
            'total_watches': total_watches,
            'seen_watches': seen_count,
            'exploration_percentage': exploration_percentage,
            'brands_explored': len(self.seen_brands),
            'styles_explored': len(self.seen_styles),
            'can_reset': exploration_percentage > 70  # Suggest reset when >70% explored
        }

    def _load_precomputed_smart_seeds(self):
        """Load precomputed smart seed sets from file."""
        try:
            # Try new format first (multiple sets)
            seeds_path = os.path.join(os.path.dirname(__file__), '../data/precomputed_smart_seed_sets.pkl')
            
            if os.path.exists(seeds_path):
                with open(seeds_path, 'rb') as f:
                    seeds_data = pickle.load(f)
                
                if 'seed_sets' in seeds_data:
                    # New format: multiple sets
                    self.precomputed_seed_sets = seeds_data.get('seed_sets', [])
                    self.seeds_metadata = seeds_data.get('metadata', {})
                    
                    total_seeds = sum(len(s) for s in self.precomputed_seed_sets)
                    print(f"✅ Loaded {len(self.precomputed_seed_sets)} seed sets ({total_seeds} total seeds)")
                    print(f"📊 Sets version: {seeds_data.get('version', 'unknown')}")
                    print(f"🎯 Generation method: {self.seeds_metadata.get('generation_method', 'unknown')}")
                    
                    return True
            
            # Fallback to old format (single set)
            old_seeds_path = os.path.join(os.path.dirname(__file__), '../data/precomputed_smart_seeds.pkl')
            
            if os.path.exists(old_seeds_path):
                with open(old_seeds_path, 'rb') as f:
                    seeds_data = pickle.load(f)
                
                # Convert old format to new format (single set)
                if 'seeds' in seeds_data:
                    old_seeds = seeds_data.get('seeds', [])
                    self.precomputed_seed_sets = [old_seeds]  # Wrap in list to make it a single set
                    self.seeds_metadata = seeds_data.get('metadata', {})
                    
                    print(f"✅ Loaded legacy format: 1 seed set ({len(old_seeds)} seeds)")
                    print(f"📊 Legacy version: {seeds_data.get('version', 'unknown')}")
                    
                    return True
            
            print(f"⚠️  No precomputed seed sets found at {seeds_path} or {old_seeds_path}")
            self.precomputed_seed_sets = []
            self.seeds_metadata = {}
            return False
                
        except Exception as e:
            print(f"❌ Error loading precomputed seed sets: {e}")
            self.precomputed_seed_sets = []
            self.seeds_metadata = {}
            return False

    def get_smart_seeds(self, num_seeds: int = 7) -> List[Dict[str, Any]]:
        """Get smart seeds - randomly selects one complete set from precomputed sets."""
        
        # 🆕 USE PRECOMPUTED SEED SETS IF AVAILABLE
        if hasattr(self, 'precomputed_seed_sets') and self.precomputed_seed_sets:
            return self._get_precomputed_smart_seed_set(num_seeds)
        else:
            # Fallback to dynamic generation
            print("⚠️  Using dynamic seed generation (precomputed seed sets not available)")
            return self._get_dynamic_smart_seeds(num_seeds)
    
    def _get_precomputed_smart_seed_set(self, num_seeds: int = 7) -> List[Dict[str, Any]]:
        """Get smart seeds by randomly selecting one complete set from precomputed sets."""
        print(f"🌱 Getting seeds from {len(self.precomputed_seed_sets)} precomputed seed sets...")
        
        # Randomly select one complete seed set
        selected_set_idx = random.randint(0, len(self.precomputed_seed_sets) - 1)
        selected_set = self.precomputed_seed_sets[selected_set_idx]
        
        print(f"🎲 Randomly selected seed set {selected_set_idx + 1} with {len(selected_set)} seeds")
        
        # Filter out any seeds that are already seen (shouldn't happen but safety check)
        available_seeds = [seed for seed in selected_set 
                          if seed['index'] not in self.seen_watches]
        
        if len(available_seeds) < num_seeds:
            print(f"⚠️  Only {len(available_seeds)} unseen seeds in selected set, taking all available")
            selected_seeds = available_seeds
        else:
            # Take the requested number or the full set if requesting more than available
            selected_seeds = available_seeds[:num_seeds]
        
        # Process selected seeds
        seed_watches = []
        for seed in selected_seeds:
            # Make a copy and add session-specific metadata
            watch = seed.copy()
            
            # Ensure it has the required index
            watch_idx = watch['index']
            
            # Add session metadata
            watch['is_seed'] = True
            watch['is_precomputed'] = True
            watch['selected_from_set'] = selected_set_idx + 1
            watch['seed_order'] = len(seed_watches)
            
            seed_watches.append(watch)
            
            # 🆕 MARK AS SEEN IMMEDIATELY
            self.seen_watches.add(watch_idx)
            
            # Update tracking
            brand = watch.get('brand', '').lower()
            style = watch.get('style', self._classify_watch_style_enhanced(watch))
            self.seen_brands.add(brand)
            self.seen_styles.add(style)
        
        print(f"✅ Selected complete seed set {selected_set_idx + 1} with {len(seed_watches)} diverse seeds")
        
        # Show style diversity
        styles_in_set = [s.get('style') for s in seed_watches]
        unique_styles = len(set(styles_in_set))
        print(f"🎨 Style diversity: {unique_styles} unique styles: {', '.join(set(styles_in_set))}")
        print(f"🎯 Total seen now: {len(self.seen_watches)}")
        
        return seed_watches
    
    def _get_dynamic_smart_seeds(self, num_seeds: int = 3) -> List[Dict[str, Any]]:
        """Dynamic smart seed generation (fallback method)."""
        print("⚠️  Using dynamic seed generation - ensuring clusters are available...")
        self._ensure_clusters()
        
        seeds = []
        
        # 🆕 ENSURE SMART SEEDS ARE UNIQUE
        selected_indices = set()
        
        # Select from different aesthetic clusters for maximum diversity
        available_clusters = list(self.cluster_mappings['aesthetic'].keys())
        selected_clusters = random.sample(available_clusters, min(num_seeds, len(available_clusters)))
        
        for cluster_id in selected_clusters:
            cluster_watches = self.cluster_mappings['aesthetic'][cluster_id]
            
            if cluster_watches:
                # 🆕 FILTER OUT ALREADY SEEN AND SELECTED WATCHES
                available_watches = [idx for idx in cluster_watches 
                                   if idx not in self.seen_watches and idx not in selected_indices]
                
                if not available_watches:
                    # If no unseen watches in this cluster, try next cluster
                    continue
                
                # Select representative with highest multi-objective score
                best_watch = None
                best_score = -1
                
                # Use neutral user preferences for seed selection
                neutral_preferences = np.mean(self.normalized_embeddings, axis=0)
                
                for watch_idx in available_watches:
                    if 0 <= watch_idx < len(self.watch_data):
                        score = self.multi_objective_score(watch_idx, neutral_preferences, [])
                        if score > best_score:
                            best_score = score
                            best_watch = watch_idx
                
                if best_watch is not None:
                    seeds.append(best_watch)
                    selected_indices.add(best_watch)
        
        # If we need more seeds, fill from remaining unseen watches
        while len(seeds) < num_seeds:
            remaining_unseen = [idx for idx in range(len(self.watch_data)) 
                              if idx not in self.seen_watches and idx not in selected_indices]
            
            if not remaining_unseen:
                print(f"⚠️  All watches have been seen! Only got {len(seeds)} seeds instead of {num_seeds}")
                break
                
            # Pick a random unseen watch
            additional_seed = random.choice(remaining_unseen)
            seeds.append(additional_seed)
            selected_indices.add(additional_seed)
        
        # Convert to watch objects
        seed_watches = []
        for idx in seeds:
            if 0 <= idx < len(self.watch_data):
                watch = self.watch_data[idx].copy()
                watch['index'] = idx
                
                # 🆕 ADD SEED METADATA
                watch['is_seed'] = True
                watch['is_precomputed'] = False
                watch['seed_order'] = len(seed_watches)
                
                seed_watches.append(watch)
                
                # 🆕 MARK AS SEEN IMMEDIATELY
                self.seen_watches.add(idx)
                
                # Update tracking
                brand = watch.get('brand', '').lower()
                style = self._classify_watch_style_enhanced(watch)
                self.seen_brands.add(brand)
                self.seen_styles.add(style)
        
        print(f"🌱 Selected {len(seed_watches)} dynamic smart seeds, total seen now: {len(self.seen_watches)}")
        
        return seed_watches
    
    def calculate_weighted_user_preference_vector(self, liked_indices: List[int], 
                                                disliked_indices: List[int]) -> np.ndarray:
        """Enhanced preference calculation with temporal weighting and context."""
        if not liked_indices and not disliked_indices:
            return np.mean(self.normalized_embeddings, axis=0)
        
        preference_vector = np.zeros(self.dimension)
        total_weight = 0.0
        
        # Process likes with temporal decay and confidence weighting
        for feedback in self.feedback_history:
            if feedback['type'] == 'like' and feedback['watch_index'] in liked_indices:
                age = time.time() - feedback['timestamp']
                temporal_weight = np.exp(-age / 3600)  # Decay over 1 hour
                confidence_weight = feedback['confidence']
                
                weight = temporal_weight * confidence_weight
                preference_vector += weight * self.normalized_embeddings[feedback['watch_index']]
                total_weight += weight
        
        # Process dislikes with negative weighting
        for feedback in self.feedback_history:
            if feedback['type'] == 'dislike' and feedback['watch_index'] in disliked_indices:
                age = time.time() - feedback['timestamp']
                temporal_weight = np.exp(-age / 3600)
                confidence_weight = feedback['confidence']
                
                weight = temporal_weight * confidence_weight * 0.5  # Dislikes have less impact
                preference_vector -= weight * self.normalized_embeddings[feedback['watch_index']]
                total_weight += weight
        
        if total_weight > 0:
            preference_vector /= total_weight
            # Normalize the result
            preference_vector = preference_vector / np.linalg.norm(preference_vector)
        else:
            preference_vector = np.mean(self.normalized_embeddings, axis=0)
        
        return preference_vector
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for the enhanced system."""
        return {
            'current_beam_width': self.current_beam_width,
            'exploration_factor': self.exploration_factor,
            'objective_weights': self.objective_weights.copy(),
            'user_engagement_level': self.user_profile['engagement_level'],
            'brands_explored': len(self.seen_brands),
            'styles_explored': len(self.seen_styles),
            'avg_recommendation_quality': np.mean(list(self.performance_metrics['recommendation_quality'])) if self.performance_metrics['recommendation_quality'] else 0,
            'total_feedback_count': len(self.feedback_history),
            'recent_like_rate': sum(1 for f in list(self.feedback_history)[-10:] if f['type'] == 'like') / max(len(list(self.feedback_history)[-10:]), 1)
        }
    
    def _update_preference_modes(self, watch_index: int, confidence: float):
        """
        Update multi-modal preference system when user likes a watch.
        Creates distinct preference modes for different types of watches user likes.
        """
        watch_embedding = self.normalized_embeddings[watch_index]
        watch = self.watch_data[watch_index]
        
        # If no modes exist, create the first one
        if not self.preference_modes:
            self.preference_modes.append(watch_embedding.copy())
            self.mode_weights.append(confidence)
            self.mode_contexts.append(self._generate_mode_context(watch))
            print(f"🎯 Created first preference mode: {self.mode_contexts[0]['label']}")
            return
        
        # Find the most similar existing mode
        best_mode_idx = -1
        best_similarity = -1
        
        for i, mode_vector in enumerate(self.preference_modes):
            similarity = np.dot(watch_embedding, mode_vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_mode_idx = i
        
        # If similar enough to existing mode, update it
        if best_similarity > self.modal_threshold:
            # Weighted update of existing mode
            current_weight = self.mode_weights[best_mode_idx]
            total_weight = current_weight + confidence
            
            # Update mode vector (weighted average)
            self.preference_modes[best_mode_idx] = (
                (self.preference_modes[best_mode_idx] * current_weight + 
                 watch_embedding * confidence) / total_weight
            )
            self.mode_weights[best_mode_idx] = total_weight
            
            # Update context
            self._update_mode_context(best_mode_idx, watch)
            print(f"🔄 Updated existing mode {best_mode_idx}: {self.mode_contexts[best_mode_idx]['label']}")
            
        # Otherwise, create a new mode if we haven't reached the limit
        elif len(self.preference_modes) < self.max_modes:
            self.preference_modes.append(watch_embedding.copy())
            self.mode_weights.append(confidence)
            self.mode_contexts.append(self._generate_mode_context(watch))
            print(f"🆕 Created new preference mode {len(self.preference_modes)-1}: {self.mode_contexts[-1]['label']}")
            
        # If at max modes, merge the weakest modes or replace the weakest
        else:
            weakest_idx = np.argmin(self.mode_weights)
            if self.mode_weights[weakest_idx] < confidence:
                # Replace weakest mode
                self.preference_modes[weakest_idx] = watch_embedding.copy()
                self.mode_weights[weakest_idx] = confidence
                self.mode_contexts[weakest_idx] = self._generate_mode_context(watch)
                print(f"🔄 Replaced weak mode {weakest_idx}: {self.mode_contexts[weakest_idx]['label']}")
        
        # Normalize mode weights
        total_weight = sum(self.mode_weights)
        if total_weight > 0:
            self.mode_weights = [w / total_weight for w in self.mode_weights]
        
        # Re-cluster modes periodically to optimize
        if len(self.feedback_history) % self.mode_update_frequency == 0:
            self._optimize_preference_modes()
    
    def _generate_mode_context(self, watch: Dict) -> Dict:
        """Generate descriptive context for a preference mode."""
        specs = watch.get('specs', {})
        description = watch.get('ai_description', '')
        
        # Extract key characteristics
        style = self._classify_watch_style_enhanced(watch)
        brand = watch.get('brand', '')
        price = watch.get('price', 0)
        aesthetic_keywords = self.semantic_features.get(watch.get('index', 0), {}).get('aesthetic_keywords', [])
        
        # Generate a descriptive label
        primary_aesthetic = aesthetic_keywords[0] if aesthetic_keywords else 'distinctive'
        label = f"{primary_aesthetic.title()} {style.title()}"
        
        return {
            'label': label,
            'style': style,
            'brand_category': self._categorize_brand(brand.lower()),
            'price_range': self._categorize_price(price),
            'aesthetic_keywords': aesthetic_keywords[:3],  # Top 3 keywords
            'representative_brand': brand,
            'watch_count': 1
        }
    
    def _update_mode_context(self, mode_idx: int, watch: Dict):
        """Update context information for an existing mode."""
        context = self.mode_contexts[mode_idx]
        context['watch_count'] += 1
        
        # Update aesthetic keywords (merge and keep most common)
        new_keywords = self.semantic_features.get(watch.get('index', 0), {}).get('aesthetic_keywords', [])
        current_keywords = context['aesthetic_keywords']
        
        # Simple frequency-based merging
        combined_keywords = list(set(current_keywords + new_keywords))
        context['aesthetic_keywords'] = combined_keywords[:3]
        
        # Update representative brand if this one is more common
        new_brand = watch.get('brand', '')
        if new_brand and (context['watch_count'] % 3 == 0):  # Update every 3rd watch
            context['representative_brand'] = new_brand
    
    def _categorize_price(self, price) -> str:
        """Categorize price into ranges."""
        if isinstance(price, str) or price == 0:
            return 'unknown'
        elif price < 300:
            return 'affordable'
        elif price < 1000:
            return 'mid-range'
        elif price < 3000:
            return 'premium'
        else:
            return 'luxury'
    
    def _optimize_preference_modes(self):
        """
        Periodically optimize preference modes using clustering.
        This helps merge similar modes and split overly broad ones.
        """
        if len(self.preference_modes) < 2:
            return
        
        try:
            # Get all liked watch embeddings from recent history
            liked_watches = []
            for feedback in list(self.feedback_history)[-20:]:  # Last 20 feedback items
                if feedback['type'] == 'like':
                    liked_watches.append(feedback['watch_index'])
            
            if len(liked_watches) < 3:
                return
            
            # Extract embeddings for liked watches
            liked_embeddings = self.normalized_embeddings[liked_watches]
            
            # Use DBSCAN to find natural clusters
            clustering = DBSCAN(eps=0.4, min_samples=2).fit(liked_embeddings)
            
            # If we found good clusters, update our modes
            unique_labels = set(clustering.labels_)
            if len(unique_labels) > 1 and -1 not in unique_labels:  # -1 means noise in DBSCAN
                new_modes = []
                new_weights = []
                new_contexts = []
                
                for label in unique_labels:
                    if label == -1:  # Skip noise points
                        continue
                    
                    cluster_indices = [i for i, l in enumerate(clustering.labels_) if l == label]
                    cluster_embeddings = liked_embeddings[cluster_indices]
                    
                    # Create mode from cluster centroid
                    cluster_centroid = np.mean(cluster_embeddings, axis=0)
                    cluster_centroid = cluster_centroid / np.linalg.norm(cluster_centroid)
                    
                    # Generate context from representative watch in cluster
                    representative_idx = liked_watches[cluster_indices[0]]
                    representative_watch = self.watch_data[representative_idx]
                    
                    new_modes.append(cluster_centroid)
                    new_weights.append(len(cluster_indices) / len(liked_watches))
                    new_contexts.append(self._generate_mode_context(representative_watch))
                    new_contexts[-1]['watch_count'] = len(cluster_indices)
                
                # Update our modes if clustering found something reasonable
                if 2 <= len(new_modes) <= self.max_modes:
                    self.preference_modes = new_modes
                    self.mode_weights = new_weights
                    self.mode_contexts = new_contexts
                    print(f"🔧 Optimized preference modes: found {len(new_modes)} distinct clusters")
                    for i, context in enumerate(new_contexts):
                        print(f"   Mode {i}: {context['label']} ({context['watch_count']} watches)")
        
        except Exception as e:
            print(f"Warning: Mode optimization failed: {e}")
    
    def calculate_multi_modal_preference_vector(self, liked_indices: List[int], 
                                              disliked_indices: List[int]) -> np.ndarray:
        """
        Calculate preference vector that handles multi-modal preferences intelligently.
        Instead of averaging all likes, maintain separate modes and combine them strategically.
        """
        if not self.preference_modes:
            # Fallback to standard method if no modes established
            return self.calculate_weighted_user_preference_vector(liked_indices, disliked_indices)
        
        # Use the strongest preference mode as primary, but blend with others
        if self.mode_weights:
            primary_mode_idx = np.argmax(self.mode_weights)
            primary_vector = self.preference_modes[primary_mode_idx]
            primary_weight = self.mode_weights[primary_mode_idx]
            
            # Blend with other modes based on their weights
            blended_vector = primary_vector * primary_weight
            total_weight = primary_weight
            
            for i, (mode_vector, weight) in enumerate(zip(self.preference_modes, self.mode_weights)):
                if i != primary_mode_idx and weight > 0.1:  # Only include meaningful modes
                    blended_vector += mode_vector * weight * 0.3  # Reduced influence of secondary modes
                    total_weight += weight * 0.3
            
            if total_weight > 0:
                blended_vector = blended_vector / total_weight
                blended_vector = blended_vector / np.linalg.norm(blended_vector)
            
            return blended_vector
        
        return np.mean(self.normalized_embeddings, axis=0)
    
    def multi_modal_beam_search_step(self, current_candidates: List[int], 
                                   liked_indices: List[int], disliked_indices: List[int],
                                   step: int = 0) -> List[Dict[str, Any]]:
        """
        Enhanced beam search that handles multi-modal preferences by running parallel searches
        for each preference mode and intelligently combining results.
        """
        step_start_time = time.time()
        
        if not self.preference_modes:
            # Fallback to standard method if no modes
            user_preferences = self.calculate_weighted_user_preference_vector(liked_indices, disliked_indices)
            return self.enhanced_beam_search_step(current_candidates, user_preferences, step)
        
        # 🆕 ROBUST DUPLICATE PREVENTION for multi-modal search
        all_indices = set(range(len(self.watch_data)))
        unseen_candidates = list(all_indices - self.seen_watches)
        
        print(f"Multi-modal Step {step}: Total watches: {len(self.watch_data)}, Seen: {len(self.seen_watches)}, Unseen: {len(unseen_candidates)}")
        
        # If we're running out of unseen watches, expand the pool
        if len(unseen_candidates) < self.current_beam_width * 2:
            print(f"⚠️  Multi-modal: Low unseen watch count ({len(unseen_candidates)}), expanding search pool...")
            recently_seen = set(list(self.seen_watches)[-50:])  # Last 50 seen watches
            expanded_candidates = list(all_indices - recently_seen)
            if len(expanded_candidates) > len(unseen_candidates):
                unseen_candidates = expanded_candidates
                print(f"📈 Multi-modal: Expanded pool to {len(unseen_candidates)} watches")
        
        # Run parallel beam search for each preference mode
        mode_results = []
        total_candidates_needed = min(self.current_beam_width, 10)
        candidates_per_mode = max(1, total_candidates_needed // len(self.preference_modes))
        
        for mode_idx, (mode_vector, mode_weight) in enumerate(zip(self.preference_modes, self.mode_weights)):
            if mode_weight < 0.05:  # Skip very weak modes
                continue
            
            mode_candidates = []
            
            # Score candidates for this specific mode
            for idx in unseen_candidates:
                if 0 <= idx < len(self.watch_data):
                    # 🆕 DUPLICATE CHECK in multi-modal search
                    if idx in self.seen_watches:
                        continue
                        
                    # Use mode-specific scoring
                    similarity = np.dot(self.normalized_embeddings[idx], mode_vector)
                    
                    # Calculate diversity within this mode's candidates
                    diversity = 0.0
                    if mode_candidates:
                        mode_batch_embeddings = self.normalized_embeddings[[c[0] for c in mode_candidates]]
                        watch_embedding = self.normalized_embeddings[idx].reshape(1, -1)
                        similarities_to_batch = cosine_similarity(watch_embedding, mode_batch_embeddings)[0]
                        diversity = 1.0 - np.mean(similarities_to_batch)
                    else:
                        diversity = 1.0
                    
                    # Novelty and style coherence (same as before)
                    watch = self.watch_data[idx]
                    brand = watch.get('brand', '').lower()
                    style = self._classify_watch_style_enhanced(watch)
                    
                    brand_novelty = 0.5 if brand not in self.seen_brands else 0.0
                    style_novelty = 0.5 if style not in self.seen_styles else 0.0
                    novelty = (brand_novelty + style_novelty) / 2
                    
                    # Mode-specific style coherence
                    mode_context = self.mode_contexts[mode_idx]
                    style_match = 1.0 if style == mode_context['style'] else 0.3
                    brand_match = 1.0 if self._categorize_brand(brand) == mode_context['brand_category'] else 0.5
                    
                    style_coherence = (style_match + brand_match) / 2
                    
                    # Combine scores with mode-specific weighting
                    mode_score = (
                        0.5 * similarity +           # Higher weight on similarity for mode-specific search
                        0.2 * diversity +
                        0.15 * novelty +
                        0.15 * style_coherence
                    ) * mode_weight  # Weight by mode strength
                    
                    mode_candidates.append((idx, mode_score, mode_idx))
            
            # Sort and select top candidates for this mode
            mode_candidates.sort(key=lambda x: x[1], reverse=True)
            selected_for_mode = mode_candidates[:candidates_per_mode]
            mode_results.extend(selected_for_mode)
        
        print(f"Multi-modal Step {step}: Collected {len(mode_results)} candidates from {len(self.preference_modes)} modes")
        
        # Combine results from all modes
        mode_results.sort(key=lambda x: x[1], reverse=True)  # Sort by score
        
        # Apply variant filtering and final selection
        selected_indices = [idx for idx, score, mode_idx in mode_results[:total_candidates_needed]]
        
        try:
            self._ensure_variant_detector()
            filtered_candidates = self.variant_detector.filter_diverse_watches(selected_indices)
        except:
            filtered_candidates = selected_indices
        
        # 🆕 APPLY ADDITIONAL VARIANT FILTERING FOR MULTI-MODAL
        # Convert indices to watch objects for filtering
        candidate_watches_for_filtering = []
        for idx in filtered_candidates:
            if 0 <= idx < len(self.watch_data):
                watch = self.watch_data[idx].copy()
                watch['index'] = idx
                candidate_watches_for_filtering.append(watch)
        
        # Apply our efficient variant filtering
        candidate_watches_for_filtering = self._filter_variant_duplicates_beam(candidate_watches_for_filtering)
        
        # Extract final candidate indices
        final_candidates = [w['index'] for w in candidate_watches_for_filtering]
        
        # 🆕 ENHANCED DUPLICATE VALIDATION for multi-modal
        final_unique_candidates = []
        for idx in final_candidates:
            if idx not in self.seen_watches and idx not in final_unique_candidates:
                final_unique_candidates.append(idx)
        
        # Build result watches with mode information and enhanced duplicate prevention
        result_watches = []
        added_indices = set()
        
        for idx in final_unique_candidates[:total_candidates_needed]:
            if 0 <= idx < len(self.watch_data) and idx not in added_indices:
                # 🆕 TRIPLE CHECK for multi-modal 
                if idx in self.seen_watches:
                    print(f"⚠️  Multi-modal: Skipping duplicate watch {idx} that was already seen")
                    continue
                    
                watch = self.watch_data[idx].copy()
                watch['index'] = idx
                
                # Find which mode this came from
                mode_info = next((m for i, s, m in mode_results if i == idx), 0)
                watch['preference_mode'] = mode_info
                watch['mode_context'] = self.mode_contexts[mode_info]['label'] if mode_info < len(self.mode_contexts) else 'General'
                watch['score'] = next((s for i, s, m in mode_results if i == idx), 0.0)
                
                # 🆕 ADD MULTI-MODAL METADATA
                watch['recommendation_step'] = step
                watch['seen_count'] = len(self.seen_watches)
                watch['is_multi_modal'] = True
                watch['variant_filtered'] = True  # Indicate this went through variant filtering
                
                result_watches.append(watch)
                added_indices.add(idx)
                
                # 🆕 IMMEDIATELY ADD TO SEEN WATCHES
                self.seen_watches.add(idx)
                
                # Update tracking
                brand = watch.get('brand', '').lower()
                style = self._classify_watch_style_enhanced(watch)
                self.seen_brands.add(brand)
                self.seen_styles.add(style)
        
        print(f"Multi-modal Step {step}: Selected {len(result_watches)} unique watches, total seen now: {len(self.seen_watches)}")
        
        # Update performance metrics
        step_time = time.time() - step_start_time
        self.performance_metrics['recommendation_quality'].append(
            np.mean([w.get('score', 0) for w in result_watches]))
        
        # Track modal diversity
        modes_represented = len(set(w.get('preference_mode', 0) for w in result_watches))
        self.performance_metrics['modal_diversity'].append(modes_represented / max(len(self.preference_modes), 1))
        
        return result_watches
    
    def get_preference_modes_summary(self) -> Dict[str, Any]:
        """Get summary of current preference modes for debugging/display."""
        if not self.preference_modes:
            return {'modes': [], 'status': 'no_modes_established'}
        
        modes_info = []
        for i, (weight, context) in enumerate(zip(self.mode_weights, self.mode_contexts)):
            modes_info.append({
                'mode_id': i,
                'label': context['label'],
                'weight': weight,
                'style': context['style'],
                'brand_category': context['brand_category'],
                'watch_count': context['watch_count'],
                'aesthetic_keywords': context['aesthetic_keywords'],
                'representative_brand': context['representative_brand']
            })
        
        return {
            'modes': modes_info,
            'total_modes': len(self.preference_modes),
            'status': 'multi_modal' if len(self.preference_modes) > 1 else 'single_mode',
            'modal_diversity_score': np.mean(list(self.performance_metrics['modal_diversity'])) if self.performance_metrics['modal_diversity'] else 0
        }
    
    def get_watch_by_index(self, index: int) -> Dict[str, Any]:
        """Get a specific watch by its index with enhanced metadata."""
        if 0 <= index < len(self.watch_data):
            watch = self.watch_data[index].copy()
            watch['index'] = index
            
            # Add enhanced metadata if available (lazy load semantic features)
            self._ensure_semantic_features()
            if index in self.semantic_features:
                watch['semantic_features'] = self.semantic_features[index]
            
            # Add cluster information (lazy load clusters)
            if hasattr(self, 'style_clusters') or not self._clusters_initialized:
                self._ensure_clusters()
                if hasattr(self, 'style_clusters'):
                    watch['style_cluster'] = self.style_clusters[index]
                if hasattr(self, 'brand_clusters'):
                    watch['brand_cluster'] = self.brand_clusters[index]
                if hasattr(self, 'aesthetic_clusters'):
                    watch['aesthetic_cluster'] = self.aesthetic_clusters[index]
            
            return watch
        else:
            raise IndexError(f"Watch index {index} out of range")
    
    def get_series_watches(self, watch_index: int) -> List[Dict[str, Any]]:
        """Get all watches from the same series as the specified watch."""
        if not (0 <= watch_index < len(self.watch_data)):
            raise IndexError(f"Watch index {watch_index} out of range")
        
        target_watch = self.watch_data[watch_index]
        target_specs = target_watch.get('specs', {})
        target_series = target_specs.get('serie', '')
        target_brand = target_watch.get('brand', '')
        
        if not target_series or target_series in ['-', '', 'N/A']:
            # If no series info, return just the target watch
            return [self.get_watch_by_index(watch_index)]
        
        # Find all watches with same brand and series
        series_watches = []
        for i, watch in enumerate(self.watch_data):
            watch_specs = watch.get('specs', {})
            watch_series = watch_specs.get('serie', '')
            watch_brand = watch.get('brand', '')
            
            if (watch_brand == target_brand and 
                watch_series == target_series and 
                watch_series not in ['-', '', 'N/A']):
                series_watches.append(self.get_watch_by_index(i))
        
        # Sort by model name for consistency
        series_watches.sort(key=lambda w: w.get('model', ''))
        
        return series_watches

    def _initialize_variant_filter(self):
        """Initialize efficient variant filtering using brand/model analysis."""
        print("🔧 Initializing variant filtering system for beam search...")
        
        # Build brand-model mapping for fast lookups
        self.brand_model_map = defaultdict(set)
        self.watch_signatures = {}  # watch_index -> (brand, base_model) tuple
        
        # Process all watches to create variant signatures
        for i, watch in enumerate(self.watch_data):
            brand = self._normalize_brand_name(watch.get('brand', ''))
            model = self._extract_base_model_name(watch.get('model', watch.get('model_name', '')))
            
            signature = (brand, model)
            self.watch_signatures[i] = signature
            self.brand_model_map[brand].add(model)
        
        # Calculate variant statistics
        total_signatures = len(set(self.watch_signatures.values()))
        total_watches = len(self.watch_data)
        unique_brands = len(self.brand_model_map)
        
        print(f"✅ Beam search variant filtering initialized:")
        print(f"   📊 {total_watches} watches → {total_signatures} unique signatures")
        print(f"   🏷️  {unique_brands} unique brands")
        print(f"   🔄 Variant reduction: {((total_watches - total_signatures) / total_watches * 100):.1f}%")

    def _normalize_brand_name(self, brand: str) -> str:
        """Normalize brand name for consistent matching."""
        if not brand:
            return 'unknown'
        
        normalized = brand.lower().strip()
        
        # Handle common brand name variations
        brand_normalizations = {
            'tag heuer': 'tagheuer',
            'a. lange & söhne': 'alangesoehne',
            'a. lange & sohne': 'alangesoehne',
            'audemars piguet': 'audemarspiguet',
            'vacheron constantin': 'vacheroncastantin',
            'patek philippe': 'patekphilippe',
            'franck muller': 'franckmuller',
            'frederique constant': 'frederiqueconstant',
        }
        
        return brand_normalizations.get(normalized, normalized)

    def _extract_base_model_name(self, model: str) -> str:
        """Extract base model name, removing variant indicators."""
        if not model or model.lower() in ['-', 'n/a', 'unknown', '']:
            return 'generic'
        
        base_name = model.lower().strip()
        
        # Remove common variant indicators
        variant_patterns = [
            r'\s*\([^)]*\).*$',     # Remove parenthetical info and everything after
            r'\s*-\s*\d+mm.*$',     # Remove size specifications like "- 42mm"
            r'\s*\d+mm.*$',         # Remove size specifications like "42mm"
            r'\s*-\s*steel.*$',     # Remove material specifications
            r'\s*-\s*gold.*$',      # Remove material specifications
            r'\s*-\s*titanium.*$',  # Remove material specifications
            r'\s*-\s*black.*$',     # Remove color specifications
            r'\s*-\s*white.*$',     # Remove color specifications
            r'\s*-\s*blue.*$',      # Remove color specifications
            r'\s*-\s*silver.*$',    # Remove color specifications
            r'\s*\|.*$',            # Remove pipe and everything after
            r'\s*with.*$',          # Remove "with..." and after
            r'\s*featuring.*$',     # Remove "featuring..." and after
        ]
        
        for pattern in variant_patterns:
            base_name = re.sub(pattern, '', base_name, flags=re.IGNORECASE)
        
        return base_name.strip() or 'generic'

    def _filter_variant_duplicates_beam(self, candidate_watches: List[Dict[str, Any]], 
                                       lookback_count: int = 8) -> List[Dict[str, Any]]:
        """Filter out variant duplicates from candidate watches for beam search."""
        if not candidate_watches:
            return candidate_watches
        
        # Track what we've already added in this batch
        batch_signatures = set()
        filtered_watches = []
        
        for watch in candidate_watches:
            watch_index = watch.get('index')
            if watch_index is None or watch_index not in self.watch_signatures:
                filtered_watches.append(watch)
                continue
            
            signature = self.watch_signatures[watch_index]
            
            # Check against recently seen watches (using beam search's seen_watches)
            is_variant_of_seen = False
            for seen_index in list(self.seen_watches)[-lookback_count:]:
                if seen_index in self.watch_signatures:
                    seen_signature = self.watch_signatures[seen_index]
                    if signature == seen_signature:
                        is_variant_of_seen = True
                        break
            
            if is_variant_of_seen:
                print(f"🚫 Beam search filtered recent variant: {signature}")
                continue
            
            # Check against current batch
            if signature in batch_signatures:
                print(f"🚫 Beam search filtered batch variant: {signature}")
                continue
            
            # Add to filtered list and track signature
            filtered_watches.append(watch)
            batch_signatures.add(signature)
        
        if len(filtered_watches) < len(candidate_watches):
            print(f"🎯 Beam search variant filtering: {len(candidate_watches)} → {len(filtered_watches)} watches")
        
        return filtered_watches 