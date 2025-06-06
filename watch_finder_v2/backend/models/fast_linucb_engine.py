#!/usr/bin/env python3
"""
Fast LinUCB Engine with Precomputed Embeddings

This version loads precomputed PCA-reduced and concatenated embeddings 
instead of doing the heavy computation at runtime.

Expected startup time: <30 seconds vs 45+ minutes
"""

import os
import pickle
import time
import logging
import traceback
import numpy as np
from typing import Dict, List, Set, Optional, Any

logger = logging.getLogger(__name__)

class SimplifiedExpert:
    """Lightweight expert for LinUCB multi-armed bandit."""
    
    def __init__(self, expert_id: int, dim: int, alpha: float):
        self.expert_id = expert_id
        self.dim = dim
        self.alpha = alpha
        
        # Initialize matrices
        self.A = np.identity(dim)
        self.b = np.zeros(dim)
        self.theta = np.zeros(dim)
        
        # Cache matrix inversion for performance
        self.A_inv = np.identity(dim)
        self.A_inv_valid = True
        
        # Track liked watches for this expert
        self.liked_watches = []
        
    def add_liked_watch(self, watch_id: int, embedding: np.ndarray):
        """Add a liked watch to this expert's profile."""
        if watch_id not in self.liked_watches:
            self.liked_watches.append(watch_id)
        
        # Update with positive reward
        self.update(embedding, 1.0)
        
    def update(self, context: np.ndarray, reward: float):
        """Update expert parameters with new feedback."""
        context = context.reshape(-1)
        
        # Update A and b matrices
        self.A += np.outer(context, context)
        self.b += reward * context
        
        # Invalidate cached inverse
        self.A_inv_valid = False
        
        # Recompute theta (parameter estimate)
        try:
            self.theta = np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:
            self.theta = np.linalg.lstsq(self.A, self.b, rcond=None)[0]
    
    def get_ucb_score(self, context: np.ndarray) -> float:
        """Calculate Upper Confidence Bound score for a single context."""
        context = context.reshape(-1)
        
        # Update cached inverse if needed
        if not self.A_inv_valid:
            try:
                self.A_inv = np.linalg.inv(self.A)
                self.A_inv_valid = True
            except np.linalg.LinAlgError:
                confidence = 1.0
                predicted_reward = np.dot(self.theta, context)
                return predicted_reward + self.alpha * confidence
        
        # Calculate confidence interval using cached inverse
        try:
            confidence = np.sqrt(np.dot(context, np.dot(self.A_inv, context)))
        except np.linalg.LinAlgError:
            confidence = 1.0
        
        # UCB = predicted reward + confidence interval
        predicted_reward = np.dot(self.theta, context)
        return predicted_reward + self.alpha * confidence
    
    def batch_get_ucb_scores(self, contexts: np.ndarray) -> np.ndarray:
        """Calculate UCB scores for multiple contexts efficiently."""
        if len(contexts.shape) == 1:
            contexts = contexts.reshape(1, -1)
        
        # Update cached inverse if needed
        if not self.A_inv_valid:
            try:
                self.A_inv = np.linalg.inv(self.A)
                self.A_inv_valid = True
            except np.linalg.LinAlgError:
                # Fallback to individual computation
                return np.array([self.get_ucb_score(ctx) for ctx in contexts])
        
        # Vectorized computation using cached inverse
        try:
            # Predicted rewards
            predicted_rewards = np.dot(contexts, self.theta)
            
            # Confidence intervals using cached A_inv
            confidence_terms = np.sqrt(np.sum(contexts * np.dot(contexts, self.A_inv), axis=1))
            
            return predicted_rewards + self.alpha * confidence_terms
            
        except Exception:
            # Fallback to individual computation
            return np.array([self.get_ucb_score(ctx) for ctx in contexts])

class FastLinUCBEngine:
    """Fast LinUCB engine using precomputed embeddings."""
    
    def __init__(self, 
                 alpha: float = 0.1,
                 batch_size: int = 5,
                 max_experts: int = 4,
                 similarity_threshold: float = 0.95,
                 data_dir: Optional[str] = None):
        
        self.alpha = alpha
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
        logger.info("🚀 Loading precomputed embeddings...")
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
            
            logger.info(f"✅ Loaded precomputed data in {total_time:.2f}s:")
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
        """Initialize a new session - much faster now!"""
        # Initialize session tracking only - no need to copy embeddings!
        self.session_experts[session_id] = []
        self.session_liked_watches[session_id] = []
        self.session_shown_watches[session_id] = set()
        
        logger.info(f"✅ Created session {session_id} (fast init - no copying)")
    
    def get_recommendations(self,
                          session_id: str,
                          exclude_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """Get recommendations using precomputed embeddings."""
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
        
        # OPTIMIZED APPROACH: Process all watches at once for all experts
        logger.info(f"🤔 Session {session_id}: Querying {len(session_experts)} experts for {len(available_watches)} watches")
        
        ucb_start = time.time()
        
        # Get all embeddings at once
        all_embeddings = np.array([
            self.final_embeddings[watch_id]
            for watch_id in available_watches
        ])
        
        # Get scores from all experts efficiently
        all_scores = []
        expert_best_scores = {}
        
        for expert_id in session_experts:
            if expert_id not in self.experts:
                continue
                
            expert = self.experts[expert_id]
            
            # Get all scores for this expert at once
            expert_scores_values = expert.batch_get_ucb_scores(all_embeddings)
            
            # Create scored list for this expert
            expert_scores = [(available_watches[i], expert_scores_values[i], expert_id) 
                           for i in range(len(available_watches))]
            
            # Add to global scores
            all_scores.extend(expert_scores)
            
            # Track best scores per expert for balanced selection
            expert_scores.sort(key=lambda x: x[1], reverse=True)
            expert_best_scores[expert_id] = expert_scores[:3]
        
        # BALANCED EXPERT RECOMMENDATION STRATEGY
        final_recommendations = []
        seen_watches = set()
        
        # Phase 1: Ensure each expert gets at least 1 recommendation
        for expert_id in session_experts:
            if expert_id in expert_best_scores and len(final_recommendations) < self.batch_size:
                for watch_id, score, exp_id in expert_best_scores[expert_id]:
                    if watch_id not in seen_watches:
                        seen_watches.add(watch_id)
                        session_expert_number = session_experts.index(expert_id) + 1
                        final_recommendations.append(
                            self._format_recommendation(watch_id, score, f"expert_{session_expert_number}")
                        )
                        break
        
        # Phase 2: Fill remaining slots with best overall scores
        if len(final_recommendations) < self.batch_size:
            all_scores.sort(key=lambda x: x[1], reverse=True)
            
            for watch_id, score, expert_id in all_scores:
                if watch_id not in seen_watches and len(final_recommendations) < self.batch_size:
                    seen_watches.add(watch_id)
                    session_expert_number = session_experts.index(expert_id) + 1
                    final_recommendations.append(
                        self._format_recommendation(watch_id, score, f"expert_{session_expert_number}")
                    )
        
        # Log UCB timing - should be much faster now!
        ucb_time = time.time() - ucb_start
        logger.info(f"⚡ UCB calculations completed in {ucb_time:.3f}s for {len(available_watches)} watches")
        
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
                logger.info(f"👤 Created first expert {expert_id} for session {session_id}")
                return
        
        # Update existing experts with negative feedback
        if reward <= 0:
            for expert_id in session_experts:
                if expert_id in self.experts:
                    self.experts[expert_id].update(watch_embedding, reward)
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
            logger.info(f"👤 Updated expert {best_expert_id} (similarity: {best_similarity:.3f})")
        # Create new expert if we haven't reached the limit
        elif len(session_experts) < self.max_experts:
            expert_id = self._create_new_expert()
            self.session_experts[session_id].append(expert_id)
            self.experts[expert_id].add_liked_watch(watch_id, watch_embedding)
            logger.info(f"👤 Created new expert {expert_id} for session {session_id} (total: {len(session_experts)+1})")
        # Use best available expert if at limit
        elif best_expert_id:
            self.experts[best_expert_id].add_liked_watch(watch_id, watch_embedding)
            logger.info(f"👤 Used best expert {best_expert_id} (similarity: {best_similarity:.3f}, at max experts)")
    
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
    
    def _create_new_expert(self) -> int:
        """Create a new expert."""
        expert_id = self.next_expert_id
        self.next_expert_id += 1
        self.experts[expert_id] = SimplifiedExpert(expert_id, self.dim, self.alpha)
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
        logger.warning("⚠️ Creating fallback data (limited functionality)")
        
        # Create minimal data structure
        self.watch_data = {i: {'watch_id': i, 'brand': 'Sample', 'model': f'Watch {i}'} for i in range(10)}
        self.final_embeddings = {i: np.random.randn(200) for i in range(10)}
        self.available_watches = set(range(10))
        self.dim = 200
        
        logger.info("✅ Created fallback data with 10 sample watches")
    
    def shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("🔄 Shutting down FastLinUCBEngine...")
        self.session_experts.clear()
        self.experts.clear()
        logger.info("✅ FastLinUCBEngine shutdown complete")
    
    def get_expert_stats(self) -> Dict[str, Any]:
        """Get statistics about experts across all sessions."""
        total_experts = len(self.experts)
        total_sessions = len(self.session_experts)
        
        expert_likes = []
        for expert in self.experts.values():
            expert_likes.append(len(expert.liked_watches))
        
        return {
            'total_experts': total_experts,
            'total_sessions': total_sessions,
            'avg_likes_per_expert': np.mean(expert_likes) if expert_likes else 0,
            'max_likes_per_expert': max(expert_likes) if expert_likes else 0,
            'experts_per_session': {sid: len(experts) for sid, experts in self.session_experts.items()}
        } 