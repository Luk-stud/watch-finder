#!/usr/bin/env python3
"""
MABWiser Thompson Sampling Engine with Precomputed Embeddings

This version uses MABWiser library (by Fidelity) for Thompson Sampling 
instead of custom LinUCB. Maintains the same interface as FastLinUCBEngine 
for easy drop-in replacement.

Expected startup time: <30 seconds vs 45+ minutes
"""

import os
import pickle
import time
import logging
import traceback
import numpy as np
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict

# MABWiser imports
try:
    from mabwiser.mab import MAB, LearningPolicy
    MABWISER_AVAILABLE = True
except ImportError:
    MABWISER_AVAILABLE = False
    print("MABWiser not available. Install with: pip install mabwiser")

logger = logging.getLogger(__name__)

class MABWiserExpert:
    """Expert using MABWiser Contextual Thompson Sampling for multi-armed bandits."""
    
    def __init__(self, expert_id: int, available_watch_ids: List[int]):
        self.expert_id = expert_id
        self.available_watch_ids = available_watch_ids  # Don't mix with engine.available_watches!
        
        # Initialize MABWiser with CONTEXTUAL Thompson Sampling
        from mabwiser.mab import NeighborhoodPolicy
        self.mab = MAB(
            arms=available_watch_ids,
            learning_policy=LearningPolicy.ThompsonSampling(),
            neighborhood_policy=NeighborhoodPolicy.KNearest(k=3)  # Use k=3 to avoid bounds issues with limited data
        )
        
        # Track interactions and liked watches
        self.liked_watches = []
        self.total_interactions = 0
        self.is_initialized = False
        self.recent_recommendations = []  # Track full recommendation sets for better targeting
        
    def add_liked_watch(self, watch_id: int, embedding: np.ndarray):
        """Add a liked watch to this expert's profile."""
        if watch_id not in self.liked_watches:
            self.liked_watches.append(watch_id)
        
        # Update with positive reward using context
        self.update(watch_id, 1.0, embedding)
        
    def update(self, watch_id: int, reward: float, context: np.ndarray):
        """Update expert with new feedback using CONTEXTUAL partial_fit with correct plural parameters."""
        if watch_id not in self.available_watch_ids:
            logger.warning(f"Watch {watch_id} not in expert {self.expert_id} available_watch_ids")
            return
            
        try:
            self.total_interactions += 1
            
            # Use ONLINE LEARNING with correct plural parameter names
            if not self.is_initialized:
                # Initialize with first context - use list format for initial fit
                self.mab.fit(
                    decisions=[watch_id],
                    rewards=[reward],
                    contexts=[context]
                )
                self.is_initialized = True
                logger.debug(f"Expert {self.expert_id}: Initialized with watch {watch_id}, reward {reward}")
            else:
                # Online update with correct plural parameter names (no singular keywords)
                self.mab.partial_fit(
                    decisions=[watch_id],      # Always use plural 'decisions' parameter
                    rewards=[reward],          # Always use plural 'rewards' parameter
                    contexts=[context]         # Always use plural 'contexts' parameter
                )
                logger.debug(f"Expert {self.expert_id}: Updated with watch {watch_id}, reward {reward}")
            
        except Exception as e:
            logger.error(f"Error updating contextual MABWiser expert {self.expert_id}: {e}")
    
    def get_recommendations(self, contexts_dict: Dict[int, np.ndarray], n_samples: int = 5) -> List[tuple]:
        """Get recommendations using CONTEXTUAL Thompson Sampling with multiple predict calls for diversity."""
        try:
            if not self.is_initialized:
                # Cold start: random recommendations
                random_watches = np.random.choice(
                    self.available_watch_ids, 
                    size=min(n_samples, len(self.available_watch_ids)), 
                    replace=False
                )
                recommendations = [(int(w), 0.5, self.expert_id) for w in random_watches]
                # Track recommendations for negative feedback targeting
                self.recent_recommendations = [int(w) for w in random_watches]
                return recommendations
            
            # Filter available contexts to only include our available_watch_ids
            available_contexts = {
                watch_id: context 
                for watch_id, context in contexts_dict.items() 
                if watch_id in self.available_watch_ids
            }
            
            if not available_contexts:
                logger.warning(f"Expert {self.expert_id}: No available contexts")
                return []
            
            # CRITICAL: Prepare ORDERED contexts for consistent predict calls
            watch_ids = list(available_contexts.keys())
            contexts = list(available_contexts.values())
            
            # THOMPSON SAMPLING DIVERSITY: Use multiple predict calls for stochastic exploration
            predictions = []
            used_watches = set()
            
            # Thompson Sampling is naturally stochastic - multiple calls give different results
            max_attempts = min(n_samples * 3, 20)  # Reasonable limit on attempts
            
            for attempt in range(max_attempts):
                if len(predictions) >= n_samples:
                    break
                
                try:
                    # Single predict call (no n= parameter exists in MABWiser)
                    pred = self.mab.predict(contexts=contexts)
                    
                    # Handle both single prediction and list of predictions
                    if isinstance(pred, list):
                        for p in pred:
                            if p not in used_watches and len(predictions) < n_samples:
                                used_watches.add(p)
                                predictions.append(p)
                    else:
                        if pred not in used_watches:
                            used_watches.add(pred)
                            predictions.append(pred)
                
                except Exception as e:
                    logger.debug(f"Expert {self.expert_id} prediction attempt {attempt} failed: {e}")
                    continue
            
            # Fill remaining slots if needed with random selection
            if len(predictions) < n_samples:
                remaining_watches = [w for w in watch_ids if w not in used_watches]
                needed = n_samples - len(predictions)
                if remaining_watches:
                    additional = np.random.choice(
                        remaining_watches,
                        size=min(needed, len(remaining_watches)),
                        replace=False
                    )
                    predictions.extend(additional)
            
            # Get confidence scores using predict_expectations
            try:
                # Get expectations for all contexts (returns dict format)
                expectations = self.mab.predict_expectations(contexts=contexts)
                
                # Extract confidences for our predictions
                confidences = []
                for watch_id in predictions:
                    if isinstance(expectations, dict):
                        # Single context case: {watch_id: confidence}
                        confidence = expectations.get(watch_id, 0.5)
                    elif isinstance(expectations, list) and len(expectations) > 0:
                        # Multiple contexts case: list of dicts
                        # Find confidence across all context predictions
                        confidence = 0.5
                        for exp_dict in expectations:
                            if isinstance(exp_dict, dict) and watch_id in exp_dict:
                                confidence = exp_dict[watch_id]
                                break
                    else:
                        confidence = 0.5
                    
                    # Handle nested dict format
                    if isinstance(confidence, dict):
                        confidence = list(confidence.values())[0] if confidence else 0.5
                    
                    confidences.append(confidence)
                    
            except Exception as e:
                logger.debug(f"Expert {self.expert_id} predict_expectations failed: {e}")
                confidences = [0.5] * len(predictions)
            
            # TRACK FULL RECOMMENDATION SET for better negative feedback targeting
            self.recent_recommendations = predictions.copy()
            
            # Format recommendations with robust confidence handling
            recommendations = []
            for watch_id, confidence in zip(predictions, confidences):
                try:
                    # Ensure confidence is a valid number
                    if not isinstance(confidence, (int, float)):
                        confidence = 0.5
                    
                    recommendations.append((int(watch_id), float(confidence), self.expert_id))
                except (ValueError, TypeError):
                    # Fallback to default confidence
                    recommendations.append((int(watch_id), 0.5, self.expert_id))
            
            return recommendations[:n_samples]
            
        except Exception as e:
            logger.error(f"Error getting contextual recommendations from expert {self.expert_id}: {e}")
            logger.error(f"Error details: {traceback.format_exc()}")
            # Fallback to random selection from available_watch_ids
            random_watches = np.random.choice(
                self.available_watch_ids, 
                size=min(n_samples, len(self.available_watch_ids)), 
                replace=False
            )
            fallback_recommendations = [(int(w), 0.5, self.expert_id) for w in random_watches]
            # Track fallback recommendations too
            self.recent_recommendations = [int(w) for w in random_watches]
            return fallback_recommendations

    def should_handle_negative_feedback(self, watch_id: int) -> bool:
        """Check if this expert should handle negative feedback for this watch."""
        # Check against FULL recommendation set, not just top recommendation
        return watch_id in self.recent_recommendations
    
    def clean_old_recommendations(self, max_history: int = 50):
        """Clean up old recommendations to prevent memory bloat."""
        if len(self.recent_recommendations) > max_history:
            # Keep only the most recent recommendations
            self.recent_recommendations = self.recent_recommendations[-max_history:]

class MABWiserEngine:
    """MABWiser Thompson Sampling engine using precomputed embeddings."""
    
    def __init__(self, 
                 batch_size: int = 3,
                 max_experts: int = 2,
                 similarity_threshold: float = 0.5,
                 data_dir: Optional[str] = None):
        
        if not MABWISER_AVAILABLE:
            raise ImportError("MABWiser is required. Install with: pip install mabwiser")
        
        self.batch_size = batch_size
        self.max_experts = max_experts
        self.similarity_threshold = similarity_threshold
        self.data_dir = data_dir or "data"
        
        # Data storage
        self.watch_data = {}
        self.final_embeddings = {}
        self.available_watches = set()
        self.dim = 200
        
        # Session management
        self.session_experts = {}
        self.session_liked_watches = {}
        self.session_shown_watches = {}
        
        # Expert management
        self.experts = {}
        self.next_expert_id = 1
        
        # Load precomputed data
        self._load_precomputed_data()
        
    def _load_precomputed_data(self) -> None:
        """Load precomputed embeddings and metadata."""
        logger.info("🚀 Loading precomputed embeddings for MABWiser...")
        total_start = time.time()
        
        try:
            precomputed_path = os.path.join(self.data_dir, 'precomputed_embeddings.pkl')
            
            if not os.path.exists(precomputed_path):
                logger.error(f"❌ Precomputed file not found: {precomputed_path}")
                logger.info("💡 Please run: python precompute_embeddings.py")
                self._create_fallback_data()
                return
                
            # Load precomputed data
            load_start = time.time()
            with open(precomputed_path, 'rb') as f:
                precomputed_data = pickle.load(f)
            load_time = time.time() - load_start
            
            # Extract data
            self.watch_data = precomputed_data['watch_data']
            self.final_embeddings = precomputed_data['final_embeddings']
            self.dim = precomputed_data['embedding_dim']
            
            # Build available watches set
            self.available_watches = set(self.final_embeddings.keys())
            
            total_time = time.time() - total_start
            file_size = os.path.getsize(precomputed_path) / (1024 * 1024)
            
            logger.info(f"✅ Loaded precomputed data for MABWiser in {total_time:.2f}s:")
            logger.info(f"   • File size: {file_size:.1f}MB")
            logger.info(f"   • Load time: {load_time:.2f}s")
            logger.info(f"   • Watches: {len(self.watch_data)}")
            logger.info(f"   • Embeddings: {len(self.final_embeddings)}")
            logger.info(f"   • Embedding dim: {self.dim}D")
            
        except Exception as e:
            logger.error(f"❌ Failed to load precomputed data: {e}")
            logger.error(f"❌ Error details: {traceback.format_exc()}")
            self._create_fallback_data()
    
    def create_session(self, session_id: str) -> None:
        """Initialize a new session."""
        self.session_experts[session_id] = []
        self.session_liked_watches[session_id] = []
        self.session_shown_watches[session_id] = set()
        
        logger.info(f"✅ Created MABWiser session {session_id}")
    
    def get_recommendations(self,
                          session_id: str,
                          exclude_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """Get recommendations using MABWiser Thompson Sampling."""
        exclude_ids = exclude_ids or set()
        
        # Ensure session exists
        if session_id not in self.session_experts:
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
            # SMART COLD START: Diverse brand/price exploration instead of random
            selected_watches = self._get_diverse_cold_start_watches(available_watches)
            
            # Track the watches we're actually showing
            session_shown_watches.update(selected_watches)
            
            logger.info(f"🎯 Session {session_id}: No experts yet, using diverse exploration ({len(selected_watches)} watches)")
            return [self._format_recommendation(watch_id, 0.5, "Diverse") for watch_id in selected_watches]
        
        # CLEAN OLD RECOMMENDATIONS periodically to prevent memory bloat
        for expert_id in session_experts:
            if expert_id in self.experts:
                self.experts[expert_id].clean_old_recommendations()
        
        # Get recommendations from all experts using CONTEXTUAL predictions
        logger.info(f"🎯 Session {session_id}: Getting contextual MABWiser recommendations from {len(session_experts)} experts for {len(available_watches)} watches")
        
        mabwiser_start = time.time()
        
        # Prepare contexts for all available watches (engine.available_watches filtered)
        contexts_dict = {
            watch_id: self.final_embeddings[watch_id]
            for watch_id in available_watches
            if watch_id in self.final_embeddings
        }
        
        logger.debug(f"Prepared {len(contexts_dict)} contexts for contextual prediction")
        
        # Collect recommendations from all experts
        all_recommendations = []
        expert_recommendations = {}
        
        for expert_id in session_experts:
            if expert_id not in self.experts:
                continue
                
            expert = self.experts[expert_id]
            # Pass contexts to expert for contextual predictions
            expert_recs = expert.get_recommendations(contexts_dict, n_samples=self.batch_size * 2)
            all_recommendations.extend(expert_recs)
            
            # Track best recommendations per expert for balanced selection
            expert_recs.sort(key=lambda x: x[1], reverse=True)  # Sort by Thompson Sampling score
            expert_recommendations[expert_id] = expert_recs[:3]  # Top 3 per expert
        
        # BALANCED EXPERT RECOMMENDATION STRATEGY
        final_recommendations = []
        seen_watches = set()
        
        # Phase 1: Ensure each expert gets at least 1 recommendation
        for expert_id in session_experts:
            if expert_id in expert_recommendations and len(final_recommendations) < self.batch_size:
                for watch_id, score, exp_id in expert_recommendations[expert_id]:
                    if watch_id not in seen_watches and watch_id in available_watches:
                        seen_watches.add(watch_id)
                        session_expert_number = session_experts.index(expert_id) + 1
                        final_recommendations.append(
                            self._format_recommendation(watch_id, score, f"expert_{session_expert_number}_mab")
                        )
                        break
        
        # Phase 2: Fill remaining slots with best overall MABWiser scores
        if len(final_recommendations) < self.batch_size:
            all_recommendations.sort(key=lambda x: x[1], reverse=True)
            
            for watch_id, score, expert_id in all_recommendations:
                if watch_id not in seen_watches and watch_id in available_watches and len(final_recommendations) < self.batch_size:
                    seen_watches.add(watch_id)
                    session_expert_number = session_experts.index(expert_id) + 1
                    final_recommendations.append(
                        self._format_recommendation(watch_id, score, f"expert_{session_expert_number}_mab")
                    )
        
        # Log MABWiser timing
        mabwiser_time = time.time() - mabwiser_start
        logger.info(f"⚡ MABWiser Thompson Sampling completed in {mabwiser_time:.3f}s")
        
        # Track all watches we're actually showing
        shown_watch_ids = [rec.get('watch_id') for rec in final_recommendations if rec.get('watch_id') is not None]
        session_shown_watches.update(shown_watch_ids)
        
        return final_recommendations
    
    def update(self, session_id: str, watch_id: int, reward: float) -> None:
        """Update system with feedback."""
        # Ensure session exists
        if session_id not in self.session_experts:
            self.create_session(session_id)
        
        # Get precomputed embedding
        if watch_id not in self.final_embeddings:
            logger.warning(f"Watch {watch_id} not found in embeddings")
            return
        
        watch_embedding = self.final_embeddings[watch_id]
        
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
                self.experts[expert_id].add_liked_watch(watch_id, watch_embedding)
                logger.info(f"👤 Created first MABWiser expert {expert_id} for session {session_id}")
                return
        
        # IMPROVED NEGATIVE FEEDBACK: Only update the expert that recommended this watch
        if reward <= 0:
            updated_count = 0
            target_experts = []
            
            # Find ALL experts that recommended this watch
            for expert_id in session_experts:
                if expert_id in self.experts:
                    expert = self.experts[expert_id]
                    if expert.should_handle_negative_feedback(watch_id):
                        target_experts.append(expert_id)
                        # Use binary pseudo-reward (0) for Thompson Sampling compatibility
                        expert.update(watch_id, 0.0, watch_embedding)
                        updated_count += 1
            
            if target_experts:
                logger.debug(f"Targeted negative feedback: Updated {updated_count} experts {target_experts} for watch {watch_id}")
            else:
                # If no expert claims this recommendation, apply binary negative to all (fallback)
                for expert_id in session_experts:
                    if expert_id in self.experts:
                        self.experts[expert_id].update(watch_id, 0.0, watch_embedding)
                        updated_count += 1
                logger.debug(f"Fallback negative feedback: Updated all {updated_count} experts for unclaimed watch {watch_id}")
            return
        
        # For positive feedback, find best matching expert or create new one
        best_expert_id = None
        best_similarity = -1
        
        for expert_id in session_experts:
            if expert_id not in self.experts:
                continue
                
            expert = self.experts[expert_id]
            
            # Calculate similarity with expert's liked watches
            similarities = []
            for liked_watch_id in expert.liked_watches:
                if liked_watch_id in self.final_embeddings:
                    liked_embedding = self.final_embeddings[liked_watch_id]
                    similarity = np.dot(watch_embedding, liked_embedding)
                    similarities.append(similarity)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    best_expert_id = expert_id
        
        # Use existing expert if similarity is high enough
        if best_expert_id and best_similarity > self.similarity_threshold:
            self.experts[best_expert_id].add_liked_watch(watch_id, watch_embedding)
            logger.info(f"👤 Updated MABWiser expert {best_expert_id} (similarity: {best_similarity:.3f})")
        # Create new expert if we haven't reached the limit
        elif len(session_experts) < self.max_experts:
            expert_id = self._create_new_expert()
            self.session_experts[session_id].append(expert_id)
            self.experts[expert_id].add_liked_watch(watch_id, watch_embedding)
            logger.info(f"👤 Created new MABWiser expert {expert_id} for session {session_id} (total: {len(session_experts)+1})")
        # Use best available expert if at limit
        elif best_expert_id:
            self.experts[best_expert_id].add_liked_watch(watch_id, watch_embedding)
            logger.info(f"👤 Used best MABWiser expert {best_expert_id} (similarity: {best_similarity:.3f}, at max experts)")
    
    def _get_unique_watches(self, exclude_ids: Set[int]) -> List[int]:
        """Get available watches with deduplication by brand+model."""
        seen_combinations = set()
        unique_watches = []
        
        for watch_id in self.available_watches:
            if watch_id in exclude_ids:
                continue
                
            watch_data = self.watch_data.get(watch_id, {})
            brand = watch_data.get('brand', 'Unknown')
            model = watch_data.get('model', 'Unknown')
            combination = f"{brand}_{model}"
            
            if combination not in seen_combinations:
                seen_combinations.add(combination)
                unique_watches.append(watch_id)
        
        return unique_watches
    
    def _get_diverse_cold_start_watches(self, available_watches: List[int]) -> List[int]:
        """Get diverse watches for cold start: different brands, price ranges, and styles."""
        if len(available_watches) <= self.batch_size:
            return available_watches
        
        # Group watches by brand and price range
        brand_groups = {}
        price_ranges = {'low': [], 'mid': [], 'high': []}
        
        for watch_id in available_watches:
            watch_data = self.watch_data.get(watch_id, {})
            brand = watch_data.get('brand', 'Unknown')
            price = watch_data.get('price', 0)
            
            # Group by brand
            if brand not in brand_groups:
                brand_groups[brand] = []
            brand_groups[brand].append(watch_id)
            
            # Group by price range
            if price < 1000:
                price_ranges['low'].append(watch_id)
            elif price < 5000:
                price_ranges['mid'].append(watch_id)
            else:
                price_ranges['high'].append(watch_id)
        
        selected = []
        
        # Strategy 1: Pick one from each major brand (up to 3)
        sorted_brands = sorted(brand_groups.items(), key=lambda x: len(x[1]), reverse=True)
        for brand, watches in sorted_brands[:3]:
            if len(selected) < self.batch_size:
                selected.append(np.random.choice(watches))
        
        # Strategy 2: Fill remaining slots with diverse price ranges
        remaining_slots = self.batch_size - len(selected)
        if remaining_slots > 0:
            all_price_watches = []
            for price_range, watches in price_ranges.items():
                if watches:
                    all_price_watches.extend(np.random.choice(watches, 
                                           size=min(remaining_slots // 3 + 1, len(watches)), 
                                           replace=False))
            
            # Add remaining watches, avoiding duplicates
            for watch_id in all_price_watches:
                if watch_id not in selected and len(selected) < self.batch_size:
                    selected.append(watch_id)
        
        # If still need more, add random ones
        if len(selected) < self.batch_size:
            remaining = [w for w in available_watches if w not in selected]
            if remaining:
                additional = np.random.choice(
                    remaining, 
                    size=min(self.batch_size - len(selected), len(remaining)), 
                    replace=False
                )
                selected.extend(additional)
        
        return selected[:self.batch_size]
    
    def _create_new_expert(self) -> int:
        """Create a new MABWiser expert."""
        expert_id = self.next_expert_id
        self.next_expert_id += 1
        
        # Create expert with available watch IDs (careful: engine.available_watches vs expert.available_watch_ids)
        available_watch_ids = list(self.available_watches)  # From engine's loaded watches
        self.experts[expert_id] = MABWiserExpert(expert_id, available_watch_ids)
        
        logger.debug(f"Created expert {expert_id} with {len(available_watch_ids)} available_watch_ids")
        return expert_id
    
    def _format_recommendation(self, watch_id: int, confidence: float, algorithm: str) -> Dict[str, Any]:
        """Format a recommendation with watch data."""
        watch_data = self.watch_data.get(watch_id, {})
        return {
            **watch_data,
            'watch_id': watch_id,
            'confidence': float(confidence),
            'algorithm': algorithm
        }
    
    def _create_fallback_data(self) -> None:
        """Create minimal fallback data if precomputed file is missing."""
        logger.warning("⚠️ Creating fallback data for MABWiser (limited functionality)")
        
        # Create minimal data structure
        self.watch_data = {i: {'watch_id': i, 'brand': 'Sample', 'model': f'Watch {i}'} for i in range(10)}
        self.final_embeddings = {i: np.random.randn(200) for i in range(10)}
        self.available_watches = set(range(10))
        self.dim = 200
        
        logger.info("✅ Created fallback data with 10 sample watches for MABWiser")
    
    def shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("🔄 Shutting down MABWiserEngine...")
        self.session_experts.clear()
        self.experts.clear()
        logger.info("✅ MABWiserEngine shutdown complete")
    
    def get_expert_stats(self) -> Dict[str, Any]:
        """Get statistics about experts across all sessions."""
        total_experts = len(self.experts)
        total_sessions = len(self.session_experts)
        
        expert_likes = []
        expert_interactions = []
        for expert in self.experts.values():
            expert_likes.append(len(expert.liked_watches))
            expert_interactions.append(expert.total_interactions)
        
        return {
            'total_experts': total_experts,
            'total_sessions': total_sessions,
            'avg_likes_per_expert': np.mean(expert_likes) if expert_likes else 0,
            'max_likes_per_expert': max(expert_likes) if expert_likes else 0,
            'avg_interactions_per_expert': np.mean(expert_interactions) if expert_interactions else 0,
            'experts_per_session': {sid: len(experts) for sid, experts in self.session_experts.items()},
            'algorithm': 'MABWiser Thompson Sampling'
        } 