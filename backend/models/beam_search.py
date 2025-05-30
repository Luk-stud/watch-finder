import numpy as np
from typing import List, Dict, Any, Tuple
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import time
from datetime import datetime
import sys
import os

# Add current directory to path for variant_detector import
sys.path.append(os.path.dirname(__file__))
from variant_detector import WatchVariantDetector

class WatchBeamSearch:
    def __init__(self, embeddings: np.ndarray, watch_data: List[Dict[str, Any]], beam_width: int = 10):
        """
        Initialize the beam search for watch recommendation.
        
        Args:
            embeddings: Watch embeddings array
            watch_data: List of watch dictionaries
            beam_width: Number of watches to consider at each step (increased for more exploration)
        """
        self.embeddings = embeddings
        self.watch_data = watch_data
        self.beam_width = beam_width
        self.seen_watches = set()  # Track all watches shown to avoid repetition
        
        # Store embedding dimensions and normalize embeddings for cosine similarity
        self.dimension = embeddings.shape[1]
        self.normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Enhanced feedback tracking
        self.feedback_history = []  # Store all feedback with timestamps
        self.session_start_time = time.time()
        
        # Metrics tracking
        self.metrics = {
            'total_steps': 0,
            'total_feedback': 0,
            'likes': 0,
            'dislikes': 0,
            'step_times': [],
            'diversity_scores': [],
            'exploration_rates': [],
            'brand_coverage': set(),
            'style_coverage': set()
        }
        
        # Initialize variant detection system
        print("Initializing variant detection system...")
        self.variant_detector = WatchVariantDetector(watch_data, normalized_embeddings)
        variant_stats = self.variant_detector.get_variant_stats()
        print(f"Variant detection: {variant_stats['variant_groups']} groups, {variant_stats['diversity_reduction']:.1f}% diversity improvement")
        
        # Precompute style clusters for smart seeds and diversity
        self._initialize_style_clusters()
        
    def _initialize_style_clusters(self):
        """Initialize style clusters for better recommendation diversity."""
        try:
            # Use K-means to find natural clusters in the embedding space
            n_clusters = min(20, len(self.watch_data) // 10)  # Adaptive cluster count
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.style_clusters = self.kmeans.fit_predict(self.normalized_embeddings)
            
            # Group watches by cluster for smart seed selection
            self.cluster_to_watches = {}
            for idx, cluster_id in enumerate(self.style_clusters):
                if cluster_id not in self.cluster_to_watches:
                    self.cluster_to_watches[cluster_id] = []
                self.cluster_to_watches[cluster_id].append(idx)
            
            print(f"Initialized {n_clusters} style clusters for smart recommendations")
        except Exception as e:
            print(f"Error initializing clusters: {e}")
            # Fallback: no clustering
            self.style_clusters = np.zeros(len(self.watch_data))
            self.cluster_to_watches = {0: list(range(len(self.watch_data)))}

    def get_smart_seeds(self, num_seeds: int = 3) -> List[Dict[str, Any]]:
        """
        Get smart seed watches from different style clusters, ensuring diversity and avoiding variants.
        This ensures initial diversity across different watch styles.
        """
        print(f"Getting {num_seeds} smart seeds from style clusters...")
        
        # Select diverse clusters
        available_clusters = list(self.cluster_to_watches.keys())
        selected_clusters = random.sample(available_clusters, min(num_seeds, len(available_clusters)))
        
        seeds = []
        for cluster_id in selected_clusters:
            cluster_watches = self.cluster_to_watches[cluster_id]
            
            # Filter to get diverse representatives (no variants)
            diverse_watches = self.variant_detector.filter_diverse_watches(cluster_watches)
            
            if diverse_watches:
                # Pick a representative watch from this cluster (closest to cluster center)
                cluster_center = self.kmeans.cluster_centers_[cluster_id]
                
                best_watch = None
                best_distance = float('inf')
                
                for watch_idx in diverse_watches:
                    distance = np.linalg.norm(self.normalized_embeddings[watch_idx] - cluster_center)
                    if distance < best_distance:
                        best_distance = distance
                        best_watch = watch_idx
                
                if best_watch is not None:
                    seeds.append(best_watch)
                    print(f"Selected seed from cluster {cluster_id}: {self.watch_data[best_watch]['brand']} {self.watch_data[best_watch]['model_name']}")
        
        # If we need more seeds, add diverse random ones
        while len(seeds) < num_seeds:
            available_indices = [i for i in range(len(self.watch_data)) if i not in seeds]
            if available_indices:
                # Filter for diversity
                diverse_available = self.variant_detector.filter_diverse_watches(available_indices)
                if diverse_available:
                    seeds.append(random.choice(diverse_available))
        
        # Convert to watch objects
        seed_watches = []
        for idx in seeds:
            watch = self.watch_data[idx].copy()
            watch['index'] = idx
            seed_watches.append(watch)
            self.seen_watches.add(idx)
        
        print(f"Smart seeds selected: {len(seed_watches)} diverse watches (variant-aware)")
        return seed_watches

    def get_random_seeds(self, num_seeds: int = 1) -> List[Dict[str, Any]]:
        """Fallback to random seeds if smart seeds fail, but now calls smart seeds by default."""
        return self.get_smart_seeds(num_seeds)

    def add_feedback(self, watch_index: int, feedback_type: str, confidence: float = 1.0):
        """
        Add user feedback with timestamp for weighted preference calculation.
        Now includes automatic feedback propagation to watch variants.
        
        Args:
            watch_index: Index of the watch
            feedback_type: 'like' or 'dislike'
            confidence: User confidence level (1.0 = very sure, 0.5 = somewhat sure)
        """
        feedback = {
            'watch_index': watch_index,
            'type': feedback_type,
            'confidence': confidence,
            'timestamp': time.time(),
            'step': self.metrics['total_steps']
        }
        
        self.feedback_history.append(feedback)
        
        # Update metrics
        self.metrics['total_feedback'] += 1
        if feedback_type == 'like':
            self.metrics['likes'] += 1
        elif feedback_type == 'dislike':
            self.metrics['dislikes'] += 1
        
        # Track brand and style coverage
        watch = self.watch_data[watch_index]
        self.metrics['brand_coverage'].add(watch.get('brand', 'unknown'))
        self.metrics['style_coverage'].add(self._classify_watch_style(watch))
        
        # Propagate feedback to variant watches with reduced weight
        variant_feedback = self.variant_detector.propagate_feedback_to_variants(
            watch_index, feedback_type, confidence, variant_weight=0.6
        )
        
        if variant_feedback:
            print(f"Propagating {feedback_type} feedback to {len(variant_feedback)} variants")
            for variant_idx, variant_type, variant_confidence in variant_feedback:
                variant_feedback_obj = {
                    'watch_index': variant_idx,
                    'type': variant_type,
                    'confidence': variant_confidence,
                    'timestamp': time.time(),
                    'step': self.metrics['total_steps'],
                    'is_variant_propagation': True  # Mark as propagated feedback
                }
                self.feedback_history.append(variant_feedback_obj)

    def _classify_watch_style(self, watch: Dict[str, Any]) -> str:
        """Classify watch into style categories."""
        description = watch.get('description', '').lower()
        model = watch.get('model_name', '').lower()
        combined_text = description + ' ' + model
        
        if any(word in combined_text for word in ['dive', 'diver', 'water', 'sea', 'ocean', 'submarine']):
            return 'dive'
        elif any(word in combined_text for word in ['chrono', 'chronograph', 'racing', 'speed', 'sport']):
            return 'sport'
        elif any(word in combined_text for word in ['dress', 'formal', 'elegant', 'classic']):
            return 'dress'
        elif any(word in combined_text for word in ['field', 'military', 'tactical', 'pilot', 'aviation']):
            return 'field'
        elif any(word in combined_text for word in ['gmt', 'world', 'travel', 'dual']):
            return 'travel'
        else:
            return 'casual'

    def calculate_weighted_user_preference_vector(self, liked_indices: List[int], disliked_indices: List[int]) -> np.ndarray:
        """
        Calculate user preference vector with time-weighted recent feedback.
        Recent feedback is weighted more heavily than older feedback.
        """
        preference_vector = np.zeros(self.dimension)
        current_time = time.time()
        
        # Use feedback history for more sophisticated weighting
        total_weight = 0
        
        for feedback in self.feedback_history:
            watch_idx = feedback['watch_index']
            feedback_type = feedback['type']
            confidence = feedback['confidence']
            feedback_time = feedback['timestamp']
            
            # Calculate time weight (exponential decay: recent = higher weight)
            time_elapsed = current_time - feedback_time
            time_weight = np.exp(-time_elapsed / 300.0)  # 5-minute half-life
            
            # Calculate step recency weight (more recent steps = higher weight)
            steps_ago = self.metrics['total_steps'] - feedback['step']
            step_weight = np.exp(-steps_ago * 0.1)  # Exponential decay based on steps
            
            # Combined weight
            final_weight = time_weight * step_weight * confidence
            
            if feedback_type == 'like':
                preference_vector += final_weight * self.normalized_embeddings[watch_idx]
                total_weight += final_weight
            elif feedback_type == 'dislike':
                preference_vector -= final_weight * self.normalized_embeddings[watch_idx]
                total_weight += final_weight
        
        # Fallback to simple method if no feedback history
        if total_weight == 0:
            if liked_indices:
                liked_embeddings = self.normalized_embeddings[liked_indices]
                preference_vector += np.mean(liked_embeddings, axis=0)
                total_weight += len(liked_indices)
            
            if disliked_indices:
                disliked_embeddings = self.normalized_embeddings[disliked_indices]
                preference_vector -= np.mean(disliked_embeddings, axis=0)
                total_weight += len(disliked_indices)
        
        # Normalize the preference vector
        norm = np.linalg.norm(preference_vector)
        if norm > 0:
            preference_vector = preference_vector / norm
            
        print(f"Calculated weighted preference vector from {len(self.feedback_history)} feedback items (total weight: {total_weight:.2f})")
        return preference_vector

    def calculate_user_preference_vector(self, liked_indices: List[int], disliked_indices: List[int]) -> np.ndarray:
        """Enhanced wrapper that uses weighted calculation."""
        return self.calculate_weighted_user_preference_vector(liked_indices, disliked_indices)
    
    def find_similar_watches(self, query_indices: List[int], k: int = 10) -> List[Tuple[int, float]]:
        """
        Find watches similar to the given query indices using numpy cosine similarity.
        
        Args:
            query_indices: List of watch indices to find neighbors for
            k: Number of similar watches to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if not query_indices:
            return []
        
        # Get average embedding of query watches
        query_embeddings = self.normalized_embeddings[query_indices]
        avg_embedding = np.mean(query_embeddings, axis=0)
        
        # Calculate cosine similarities with all watches
        similarities = np.dot(self.normalized_embeddings, avg_embedding)
        
        # Get indices sorted by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Filter out the query watches themselves and return top k
        results = []
        for idx in sorted_indices:
            if idx not in query_indices and len(results) < k:
                results.append((int(idx), float(similarities[idx])))
        
        return results
    
    def beam_search_step(self, current_candidates: List[int], user_preferences: np.ndarray, step: int = 0) -> List[Dict[str, Any]]:
        """
        Perform one step of beam search based on current candidates and user preferences.
        Enhanced with better exploration, diversity, and metrics tracking.
        
        Args:
            current_candidates: Current candidate watch indices
            user_preferences: User preference vector
            step: Current step number
            
        Returns:
            List of recommended watches with scores
        """
        step_start_time = time.time()
        print(f"Step {step}: Starting with {len(self.seen_watches)} seen watches")
        print(f"Current candidates to exclude: {current_candidates}")
        
        # Update metrics
        self.metrics['total_steps'] = step
        
        # Filter current_candidates to exclude already seen watches
        valid_current_candidates = [idx for idx in current_candidates if idx not in self.seen_watches]
        print(f"Valid current candidates after filtering seen: {len(valid_current_candidates)}")
        
        # Calculate exploration rate
        exploration_percentage = len(self.seen_watches) / len(self.watch_data) * 100
        self.metrics['exploration_rates'].append(exploration_percentage)
        
        if len(user_preferences) == 0:
            # No preferences yet, return diverse smart watches
            print("No user preferences yet, using smart diverse selection")
            if valid_current_candidates:
                similar_watches = self.find_similar_watches(valid_current_candidates, self.beam_width * 3)
                # Double-check to filter out seen watches
                similar_watches = [(idx, score) for idx, score in similar_watches if idx not in self.seen_watches]
            else:
                # Use smart clustering-based selection instead of random
                similar_watches = []
                available_indices = [i for i in range(len(self.watch_data)) if i not in self.seen_watches]
                print(f"Available unseen watches: {len(available_indices)}")
                
                if len(available_indices) == 0:
                    print("No unseen watches available - resetting exploration")
                    self.reset_exploration()
                    available_indices = list(range(len(self.watch_data)))
                
                # Smart selection from different clusters
                cluster_representatives = self._get_diverse_cluster_representatives(available_indices, self.beam_width * 3)
                for idx in cluster_representatives:
                    similar_watches.append((idx, 1.0))
        else:
            # Use preference vector to find watches, with enhanced weighting
            print(f"Using preference vector with {len(self.feedback_history)} feedback items")
            similarities = np.dot(self.normalized_embeddings, user_preferences)
            
            # Get top similar watches that aren't in current candidates or seen before
            sorted_indices = np.argsort(similarities)[::-1]
            similar_watches = []
            
            print(f"Filtering from {len(sorted_indices)} potential watches...")
            
            # Enhanced scoring with user engagement consideration
            for idx in sorted_indices:
                if (idx not in current_candidates and 
                    idx not in self.seen_watches and 
                    len(similar_watches) < self.beam_width * 3):
                    
                    # Base similarity score
                    base_score = float(similarities[idx])
                    
                    # Add exploration bonus for completely unseen watches
                    exploration_bonus = 0.1 if idx not in self.seen_watches else 0.0
                    
                    # Add style diversity bonus
                    style_bonus = self._calculate_style_diversity_bonus(idx, [w[0] for w in similar_watches])
                    
                    # Add cluster diversity bonus
                    cluster_bonus = self._calculate_cluster_diversity_bonus(idx, [w[0] for w in similar_watches])
                    
                    final_score = base_score + exploration_bonus + style_bonus + cluster_bonus
                    similar_watches.append((int(idx), final_score))
            
            print(f"Found {len(similar_watches)} relevant unseen watches")
            
            # If we don't have enough candidates, add some smart exploration
            if len(similar_watches) < self.beam_width:
                available_indices = [i for i in range(len(self.watch_data)) 
                                   if (i not in current_candidates and 
                                       i not in self.seen_watches and 
                                       i not in [w[0] for w in similar_watches])]
                
                print(f"Adding smart exploration from {len(available_indices)} available watches")
                
                if len(available_indices) == 0:
                    print("No more unseen watches - resetting exploration")
                    self.reset_exploration()
                    available_indices = [i for i in range(len(self.watch_data)) 
                                       if (i not in current_candidates and 
                                           i not in [w[0] for w in similar_watches])]
                
                # Smart exploration based on underrepresented clusters
                cluster_representatives = self._get_diverse_cluster_representatives(
                    available_indices, self.beam_width - len(similar_watches)
                )
                for idx in cluster_representatives:
                    similar_watches.append((idx, 0.6))  # Medium exploration score
        
        print(f"Total candidates before diversity selection: {len(similar_watches)}")
        
        # Enhanced diversity selection with variant awareness
        selected_watches = self._select_diverse_watches(similar_watches, self.beam_width)
        
        # Apply variant filtering to ensure model diversity (handles score preservation properly)
        final_selected = self._apply_variant_filtering_with_scores(selected_watches)
        
        # Final check - ensure no duplicates with seen watches
        verified_selected = []
        for idx, score in final_selected:
            if idx not in self.seen_watches:
                verified_selected.append((idx, score))
            else:
                print(f"WARNING: Filtered out duplicate watch {idx} at final stage")
        
        print(f"Final selected watches after variant filtering: {len(verified_selected)}")
        
        # Calculate diversity score for this batch
        if len(verified_selected) > 1:
            batch_diversity = self._calculate_batch_diversity([idx for idx, _ in verified_selected])
            self.metrics['diversity_scores'].append(batch_diversity)
        
        # Convert to watch objects with metadata and track seen watches
        recommendations = []
        for idx, score in verified_selected:
            watch = self.watch_data[idx].copy()
            watch['index'] = idx
            watch['score'] = score
            watch['step'] = step
            watch['style'] = self._classify_watch_style(watch)
            
            # Add variant information
            variants = self.variant_detector.get_variant_group(idx)
            watch['has_variants'] = len(variants) > 1
            watch['variant_count'] = len(variants)
            if len(variants) > 1:
                watch['is_representative'] = self.variant_detector.get_representative(idx) == idx
            
            recommendations.append(watch)
            
            # Add to seen watches to avoid repetition
            self.seen_watches.add(idx)
            print(f"Added watch {idx} ({watch['brand']} {watch['model_name']}) to seen list")
            
            # Update brand coverage metrics
            self.metrics['brand_coverage'].add(watch['brand'])
            self.metrics['style_coverage'].add(watch['style'])
        
        # Record step timing
        step_time = time.time() - step_start_time
        self.metrics['step_times'].append(step_time)
        
        print(f"Step {step}: Selected {len(recommendations)} watches, total seen: {len(self.seen_watches)}")
        print(f"Step completed in {step_time:.2f}s")
        return recommendations

    def _get_diverse_cluster_representatives(self, available_indices: List[int], num_select: int) -> List[int]:
        """Get representative watches from different clusters for diversity."""
        if not available_indices:
            return []
        
        # Group available watches by cluster
        cluster_groups = {}
        for idx in available_indices:
            cluster_id = self.style_clusters[idx]
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(idx)
        
        representatives = []
        cluster_ids = list(cluster_groups.keys())
        random.shuffle(cluster_ids)
        
        # Pick one representative from each cluster cyclically
        for i in range(num_select):
            if not cluster_ids:
                break
            cluster_id = cluster_ids[i % len(cluster_ids)]
            if cluster_groups[cluster_id]:
                representatives.append(cluster_groups[cluster_id].pop(0))
                if not cluster_groups[cluster_id]:
                    cluster_ids.remove(cluster_id)
        
        return representatives

    def _calculate_style_diversity_bonus(self, watch_idx: int, current_batch: List[int]) -> float:
        """Calculate bonus for style diversity within current batch."""
        if not current_batch:
            return 0.1
        
        watch_style = self._classify_watch_style(self.watch_data[watch_idx])
        batch_styles = [self._classify_watch_style(self.watch_data[idx]) for idx in current_batch]
        
        # Bonus if this style is not represented in current batch
        if watch_style not in batch_styles:
            return 0.15
        else:
            return 0.0

    def _calculate_cluster_diversity_bonus(self, watch_idx: int, current_batch: List[int]) -> float:
        """Calculate bonus for cluster diversity within current batch."""
        if not current_batch:
            return 0.05
        
        watch_cluster = self.style_clusters[watch_idx]
        batch_clusters = [self.style_clusters[idx] for idx in current_batch]
        
        # Bonus if this cluster is not represented in current batch
        if watch_cluster not in batch_clusters:
            return 0.1
        else:
            return 0.0

    def _calculate_batch_diversity(self, watch_indices: List[int]) -> float:
        """Calculate diversity score for a batch of watches."""
        if len(watch_indices) <= 1:
            return 0.0
        
        # Calculate average pairwise distance
        embeddings = self.normalized_embeddings[watch_indices]
        similarities = np.dot(embeddings, embeddings.T)
        
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(similarities, dtype=bool), k=1)
        pairwise_similarities = similarities[mask]
        
        # Diversity is 1 - average similarity
        average_similarity = np.mean(pairwise_similarities)
        diversity = 1.0 - average_similarity
        
        return float(diversity)

    def _select_diverse_watches(self, similar_watches: List[Tuple[int, float]], num_select: int) -> List[Tuple[int, float]]:
        """
        Select diverse watches from similar candidates to avoid showing too similar watches.
        Enhanced to prioritize exploration and diversity.
        """
        # First filter out any watches we've already seen
        unseen_watches = [(idx, score) for idx, score in similar_watches if idx not in self.seen_watches]
        print(f"Diversity selection: {len(similar_watches)} candidates -> {len(unseen_watches)} unseen")
        
        if len(unseen_watches) <= num_select:
            return unseen_watches
        
        selected = []
        remaining = unseen_watches.copy()
        
        # Always take the top candidate first (if it's unseen)
        if remaining:
            selected.append(remaining.pop(0))
        
        # Select remaining based on enhanced diversity
        while len(selected) < num_select and remaining:
            best_candidate = None
            best_score = -1
            best_idx = -1
            
            for i, (candidate_idx, candidate_score) in enumerate(remaining):
                # Double-check this watch hasn't been seen (extra safety)
                if candidate_idx in self.seen_watches:
                    print(f"WARNING: Found seen watch {candidate_idx} in diversity selection - skipping")
                    continue
                    
                # Calculate diversity score (average distance to already selected)
                diversity_score = 0
                exploration_bonus = 0
                
                if selected:
                    candidate_emb = self.normalized_embeddings[candidate_idx:candidate_idx+1]
                    selected_embs = self.normalized_embeddings[[s[0] for s in selected]]
                    similarities_to_selected = np.dot(candidate_emb, selected_embs.T)[0]
                    diversity_score = 1 - np.mean(similarities_to_selected)  # Lower similarity = higher diversity
                
                # Add exploration bonus for watches not seen before (should always be true now)
                exploration_bonus = 0.2
                
                # Combine scores: 50% relevance, 40% diversity, 10% exploration
                combined_score = (0.5 * candidate_score + 
                                0.4 * diversity_score + 
                                0.1 * exploration_bonus)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = (candidate_idx, candidate_score)
                    best_idx = i
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.pop(best_idx)
            else:
                # No more valid candidates
                break
        
        return selected
    
    def reset_exploration(self):
        """
        Reset the seen watches tracker to allow re-exploration.
        Called when we've seen too many watches or need fresh exploration.
        """
        print(f"Resetting exploration - had seen {len(self.seen_watches)} watches")
        self.seen_watches.clear()
    
    def get_exploration_stats(self):
        """
        Get statistics about exploration progress.
        """
        total_watches = len(self.watch_data)
        seen_watches = len(self.seen_watches)
        exploration_percentage = (seen_watches / total_watches) * 100
        
        return {
            'total_watches': total_watches,
            'seen_watches': seen_watches,
            'exploration_percentage': exploration_percentage
        }
    
    def get_watch_by_index(self, index: int) -> Dict[str, Any]:
        """
        Get watch data by index.
        """
        watch = self.watch_data[index].copy()
        watch['index'] = index
        return watch

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive recommendation metrics."""
        session_duration = time.time() - self.session_start_time
        
        # Calculate engagement metrics
        if self.metrics['total_steps'] > 0:
            feedback_rate = self.metrics['total_feedback'] / self.metrics['total_steps']
            like_rate = self.metrics['likes'] / max(1, self.metrics['total_feedback'])
            dislike_rate = self.metrics['dislikes'] / max(1, self.metrics['total_feedback'])
        else:
            feedback_rate = like_rate = dislike_rate = 0.0
        
        # Calculate diversity metrics
        avg_diversity = np.mean(self.metrics['diversity_scores']) if self.metrics['diversity_scores'] else 0.0
        
        # Calculate coverage metrics
        brand_coverage = len(self.metrics['brand_coverage']) / len(set(w['brand'] for w in self.watch_data))
        style_coverage = len(self.metrics['style_coverage']) / 6  # We have 6 style categories
        
        # Calculate exploration metrics
        exploration_percentage = len(self.seen_watches) / len(self.watch_data) * 100
        
        return {
            'session_duration_minutes': session_duration / 60,
            'total_steps': self.metrics['total_steps'],
            'total_feedback': self.metrics['total_feedback'],
            'feedback_rate': feedback_rate,
            'like_rate': like_rate,
            'dislike_rate': dislike_rate,
            'exploration_percentage': exploration_percentage,
            'watches_seen': len(self.seen_watches),
            'total_watches': len(self.watch_data),
            'average_diversity_score': avg_diversity,
            'brand_coverage_percentage': brand_coverage * 100,
            'style_coverage_percentage': style_coverage * 100,
            'brands_discovered': len(self.metrics['brand_coverage']),
            'styles_discovered': len(self.metrics['style_coverage']),
            'average_step_time': np.mean(self.metrics['step_times']) if self.metrics['step_times'] else 0.0,
            'feedback_distribution': {
                'likes': self.metrics['likes'],
                'dislikes': self.metrics['dislikes']
            },
            'recent_engagement': self._calculate_recent_engagement(),
            'recommendation_quality_score': self._calculate_quality_score()
        }

    def _calculate_recent_engagement(self) -> float:
        """Calculate engagement in recent steps."""
        if len(self.feedback_history) < 2:
            return 0.0
        
        recent_feedback = [f for f in self.feedback_history if f['step'] >= max(0, self.metrics['total_steps'] - 5)]
        return len(recent_feedback) / min(5, self.metrics['total_steps'] + 1)

    def _calculate_quality_score(self) -> float:
        """Calculate overall recommendation quality score."""
        if self.metrics['total_feedback'] == 0:
            return 0.5  # Neutral score with no feedback
        
        # Weighted score based on multiple factors
        like_ratio = self.metrics['likes'] / self.metrics['total_feedback']
        diversity_score = np.mean(self.metrics['diversity_scores']) if self.metrics['diversity_scores'] else 0.5
        exploration_score = min(1.0, len(self.seen_watches) / len(self.watch_data) * 2)  # Cap at 50% exploration
        
        # Combine scores
        quality_score = (0.5 * like_ratio + 0.3 * diversity_score + 0.2 * exploration_score)
        return float(np.clip(quality_score, 0.0, 1.0))

    def _apply_variant_filtering_with_scores(self, watch_score_pairs: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Apply variant filtering while properly preserving scores."""
        if not watch_score_pairs:
            return []
        
        # Create a map of original indices to scores
        score_map = {idx: score for idx, score in watch_score_pairs}
        original_indices = [idx for idx, _ in watch_score_pairs]
        
        # Apply variant filtering
        filtered_indices = self.variant_detector.filter_diverse_watches(original_indices, max_variants_per_group=1)
        
        # For each filtered index, find the best score
        result = []
        for filtered_idx in filtered_indices:
            if filtered_idx in score_map:
                # Direct match - use original score
                result.append((filtered_idx, score_map[filtered_idx]))
            else:
                # Filtered index is a representative - find the best score from its variant group
                variant_group = self.variant_detector.get_variant_group(filtered_idx)
                best_score = 0.0
                
                # Find the highest score among the variants that were in the original selection
                for variant_idx in variant_group:
                    if variant_idx in score_map:
                        best_score = max(best_score, score_map[variant_idx])
                
                # If no variants had scores, assign a default score
                if best_score == 0.0:
                    best_score = 0.5
                
                result.append((filtered_idx, best_score))
        
        if len(result) < len(watch_score_pairs):
            print(f"Variant filtering: {len(watch_score_pairs)} -> {len(result)} watches (removed {len(watch_score_pairs) - len(result)} variants)")
        
        return result

    def _apply_variant_filtering(self, watch_indices: List[int]) -> List[int]:
        """Apply variant filtering to ensure diverse model representation."""
        # Use variant detector to filter for diversity
        filtered_indices = self.variant_detector.filter_diverse_watches(watch_indices, max_variants_per_group=1)
        
        if len(filtered_indices) < len(watch_indices):
            print(f"Variant filtering: {len(watch_indices)} -> {len(filtered_indices)} watches (removed {len(watch_indices) - len(filtered_indices)} variants)")
        
        return filtered_indices 