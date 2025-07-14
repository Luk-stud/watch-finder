#!/usr/bin/env python3
"""
Type-Based MABWiser Engine

Simple type-based LinUCB using real watch metadata for type classification
and the same 200D embeddings as FastLinUCB.
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

# Only define the class if MABWiser is available
if MABWISER_AVAILABLE:
    class TypeBasedMABWiserEngine:
        """Simple type-based MABWiser LinUCB engine."""
        
        def __init__(self, 
                     alpha: float = 0.1,
                     batch_size: int = 5,
                     data_dir: Optional[str] = None):
            
            self.alpha = alpha
            self.batch_size = batch_size
            self.data_dir = data_dir or "data"
        
        # Data storage
        self.watch_data = {}
        self.final_embeddings = {}
        self.available_watches = set()
        self.dim = 200
        
        # Type-based organization
        self.type_of = {}  # watch_id → watch_type
        self.type_to_watch_ids = defaultdict(list)  # watch_type → [watch_ids]
        self.experts = {}  # watch_type → MAB instance
        
        # Session tracking (for shown watches)
        self.session_shown_watches = {}
        
        # Load data and initialize type-based MABs
        self._load_embeddings()
        self._load_watch_metadata()
        self._organize_by_type()
        self._create_type_experts()
        
    def _load_embeddings(self) -> None:
        """Load the same 200D embeddings as FastLinUCB."""
        logger.info("🚀 Loading 200D embeddings (same as FastLinUCB)...")
        
        try:
            precomputed_path = os.path.join(self.data_dir, 'precomputed_embeddings.pkl')
            
            if not os.path.exists(precomputed_path):
                logger.error(f"❌ Precomputed embeddings not found: {precomputed_path}")
                self._create_fallback_embeddings()
                return
                
            with open(precomputed_path, 'rb') as f:
                precomputed_data = pickle.load(f)
            
            self.final_embeddings = precomputed_data['final_embeddings']
            self.dim = precomputed_data['embedding_dim']
            
            logger.info(f"✅ Loaded {len(self.final_embeddings)} embeddings, dim: {self.dim}D")
            
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {e}")
            self._create_fallback_embeddings()
    
    def _load_watch_metadata(self) -> None:
        """Load real watch metadata for type classification."""
        logger.info("📊 Loading real watch metadata...")
        
        try:
            metadata_path = os.path.join(self.data_dir, 'watch_text_metadata.pkl')
            
            if not os.path.exists(metadata_path):
                logger.error(f"❌ Watch metadata not found: {metadata_path}")
                self._create_fallback_metadata()
                return
                
            with open(metadata_path, 'rb') as f:
                watch_list = pickle.load(f)
            
            # Convert list to dict and extract basic info
            for watch in watch_list:
                watch_id = watch.get('index')
                if watch_id is not None:
                    self.watch_data[watch_id] = {
                        'watch_id': watch_id,
                        'brand': watch.get('brand', 'Unknown'),
                        'model': watch.get('model', 'Unknown'),
                        'price': watch.get('price', 0),
                        'specs': watch.get('specs', {})
                    }
            
            # Only use watches that have both metadata and embeddings
            self.available_watches = set(self.watch_data.keys()) & set(self.final_embeddings.keys())
            
            logger.info(f"✅ Loaded metadata for {len(self.watch_data)} watches")
            logger.info(f"✅ {len(self.available_watches)} watches have both metadata and embeddings")
            
        except Exception as e:
            logger.error(f"❌ Failed to load metadata: {e}")
            self._create_fallback_metadata()
    
    def _organize_by_type(self) -> None:
        """Organize watches by type from specs."""
        logger.info("🏷️ Organizing watches by watch_type from specs...")
        
        for watch_id in self.available_watches:
            watch_data = self.watch_data.get(watch_id, {})
            specs = watch_data.get('specs', {})
            
            # Get watch type from specs
            watch_type = self._extract_watch_type(specs)
            
            # Map watch to type
            self.type_of[watch_id] = watch_type
            self.type_to_watch_ids[watch_type].append(watch_id)
        
        # Log type distribution
        type_counts = {t: len(watches) for t, watches in self.type_to_watch_ids.items()}
        logger.info(f"✅ Organized {len(self.available_watches)} watches into {len(type_counts)} types:")
        for watch_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   • {watch_type}: {count} watches")
    
    def _extract_watch_type(self, specs: Dict[str, Any]) -> str:
        """Extract watch type from specs."""
        # Primary: use explicit watch_type field
        watch_type = specs.get('watch_type', '').strip()
        if watch_type and watch_type != '-':
            return watch_type.lower()
        
        # Secondary: use second_watch_type
        second_type = specs.get('second_watch_type', '').strip()
        if second_type and second_type != '-' and second_type.lower() != 'no':
            return second_type.lower()
        
        # Fallback: use general
        return 'general'
    
    def _create_type_experts(self) -> None:
        """Create one LinUCB MAB expert per watch type."""
        logger.info("🧠 Creating LinUCB experts per watch type...")
        
        for watch_type, watch_ids in self.type_to_watch_ids.items():
            if len(watch_ids) == 0:
                continue
                
            # Create LinUCB MAB for this type
            try:
                self.experts[watch_type] = MAB(
                    arms=watch_ids,
                    learning_policy=LearningPolicy.LinUCB(alpha=self.alpha),
                    neighborhood_policy=None
                )
                logger.debug(f"   • Created LinUCB expert for '{watch_type}' with {len(watch_ids)} arms")
                
            except Exception as e:
                logger.error(f"❌ Failed to create expert for type '{watch_type}': {e}")
        
        logger.info(f"✅ Created {len(self.experts)} LinUCB experts")
    
    def create_session(self, session_id: str) -> None:
        """Initialize a new session."""
        self.session_shown_watches[session_id] = set()
        logger.info(f"✅ Created Type-Based MABWiser session {session_id}")
    
    def get_recommendations(self,
                          session_id: str,
                          exclude_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """Get recommendations using type-based LinUCB strategy."""
        exclude_ids = exclude_ids or set()
        
        # Ensure session exists
        if session_id not in self.session_shown_watches:
            self.create_session(session_id)
        
        # Exclude session-specific shown watches and provided excludes
        session_shown_watches = self.session_shown_watches[session_id]
        all_excludes = exclude_ids | session_shown_watches
        
        # Get available watches with deduplication by brand+model
        available_watches = self._get_unique_watches(all_excludes)
        
        if not available_watches:
            return []
        
        logger.info(f"🎯 Session {session_id}: Getting type-based LinUCB recommendations")
        
        mabwiser_start = time.time()
        recommendations = []
        
        # Check if we need cold start (no experts trained yet)
        trained_experts = []
        for watch_type, mab in self.experts.items():
            if hasattr(mab, '_is_initial_fit') and mab._is_initial_fit:
                trained_experts.append(watch_type)
        
        if not trained_experts:
            # Cold start: random exploration
            selected_watches = np.random.choice(
                available_watches,
                size=min(self.batch_size, len(available_watches)),
                replace=False
            ).tolist()
            
            session_shown_watches.update(selected_watches)
            
            logger.info(f"🎲 Cold start: random exploration ({len(selected_watches)} watches)")
            return [self._format_recommendation(watch_id, 0.5, "cold_start") for watch_id in selected_watches]
        
        # === STAGE 1: One recommendation per type (ensures diversity) ===
        type_recommendations = {}
        
        for watch_type, mab in self.experts.items():
            if watch_type not in trained_experts:
                continue
                
            # Get available watches for this type
            type_watch_ids = [w for w in self.type_to_watch_ids[watch_type] if w in available_watches]
            
            if not type_watch_ids:
                continue
            
            try:
                # Build contexts for this type
                contexts = np.array([self.final_embeddings[w] for w in type_watch_ids])
                
                # Get LinUCB predictions
                predictions = mab.predict(contexts)
                
                # Take the best prediction for this type
                if len(predictions) > 0:
                    best_idx = 0  # MAB.predict returns best action first
                    best_watch_id = type_watch_ids[best_idx]
                    
                    # Get confidence score
                    try:
                        expectations = mab.predict_expectations(contexts)
                        confidence = float(expectations[best_idx]) if len(expectations) > best_idx else 0.5
                    except:
                        confidence = 0.5
                    
                    type_recommendations[watch_type] = (best_watch_id, confidence)
                    
            except Exception as e:
                logger.warning(f"Error getting recommendation for type '{watch_type}': {e}")
                continue
        
        # Add type recommendations to final list
        for watch_type, (watch_id, confidence) in type_recommendations.items():
            if len(recommendations) < self.batch_size:
                recommendations.append(
                    self._format_recommendation(watch_id, confidence, f"type_{watch_type}")
                )
        
        # === STAGE 2: Fill remaining slots with global best UCB scores ===
        if len(recommendations) < self.batch_size:
            global_scores = []
            
            for watch_type, mab in self.experts.items():
                if watch_type not in trained_experts:
                    continue
                    
                type_watch_ids = [w for w in self.type_to_watch_ids[watch_type] if w in available_watches]
                
                if not type_watch_ids:
                    continue
                
                try:
                    contexts = np.array([self.final_embeddings[w] for w in type_watch_ids])
                    expectations = mab.predict_expectations(contexts)
                    
                    for i, watch_id in enumerate(type_watch_ids):
                        score = float(expectations[i]) if len(expectations) > i else 0.5
                        global_scores.append((watch_id, score, watch_type))
                        
                except Exception as e:
                    logger.warning(f"Error getting global scores for type '{watch_type}': {e}")
                    continue
            
            # Sort by score descending and fill remaining slots
            global_scores.sort(key=lambda x: x[1], reverse=True)
            
            recommended_ids = {rec['watch_id'] for rec in recommendations}
            for watch_id, score, watch_type in global_scores:
                if watch_id not in recommended_ids and len(recommendations) < self.batch_size:
                    recommendations.append(
                        self._format_recommendation(watch_id, score, f"global_{watch_type}")
                    )
                    recommended_ids.add(watch_id)
        
        # Log timing
        mabwiser_time = time.time() - mabwiser_start
        logger.info(f"⚡ Type-based LinUCB completed in {mabwiser_time:.3f}s")
        logger.info(f"   • Stage 1 (type diversity): {len(type_recommendations)} recommendations")
        logger.info(f"   • Total recommendations: {len(recommendations)}")
        
        # Track shown watches
        shown_watch_ids = [rec.get('watch_id') for rec in recommendations if rec.get('watch_id') is not None]
        session_shown_watches.update(shown_watch_ids)
        
        return recommendations
    
    def update(self, session_id: str, watch_id: int, reward: float) -> None:
        """Update system with feedback - only update relevant type's MAB."""
        # Get watch type
        if watch_id not in self.type_of:
            logger.warning(f"Watch {watch_id} not found in type mapping")
            return
        
        watch_type = self.type_of[watch_id]
        
        # Get the relevant MAB
        if watch_type not in self.experts:
            logger.warning(f"No expert found for type '{watch_type}'")
            return
        
        # Get embedding context
        if watch_id not in self.final_embeddings:
            logger.warning(f"Watch {watch_id} not found in embeddings")
            return
        
        context = self.final_embeddings[watch_id]
        mab = self.experts[watch_type]
        
        try:
            # Check if this is the first feedback for this expert
            if not hasattr(mab, '_is_initial_fit') or not mab._is_initial_fit:
                # Initialize with this feedback
                mab.fit(
                    decisions=[watch_id],
                    rewards=[reward],
                    contexts=context.reshape(1, -1)
                )
                logger.info(f"🔥 Initialized '{watch_type}' LinUCB expert with first feedback: watch {watch_id}, reward {reward}")
            else:
                # Update with partial_fit
                mab.partial_fit(
                    decisions=[watch_id],
                    rewards=[reward],
                    contexts=context.reshape(1, -1)
                )
                logger.info(f"👤 Updated '{watch_type}' LinUCB expert: watch {watch_id}, reward {reward}")
            
        except Exception as e:
            logger.error(f"Error updating type '{watch_type}' expert: {e}")
    
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
    
    def _format_recommendation(self, watch_id: int, confidence: float, algorithm: str) -> Dict[str, Any]:
        """Format a recommendation with watch data."""
        watch_data = self.watch_data.get(watch_id, {})
        return {
            **watch_data,
            'watch_id': watch_id,
            'confidence': float(confidence),
            'algorithm': algorithm
        }
    
    def _create_fallback_embeddings(self) -> None:
        """Create minimal fallback embeddings."""
        logger.warning("⚠️ Creating fallback embeddings")
        self.final_embeddings = {i: np.random.randn(200) for i in range(10)}
        self.dim = 200
    
    def _create_fallback_metadata(self) -> None:
        """Create minimal fallback metadata."""
        logger.warning("⚠️ Creating fallback metadata")
        for i in range(10):
            self.watch_data[i] = {
                'watch_id': i,
                'brand': f'Brand{i%3}',
                'model': f'Model{i}',
                'price': 1000 + i*100,
                'specs': {'watch_type': ['dress', 'sport', 'dive'][i%3]}
            }
        self.available_watches = set(range(10))
    
    def shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("🔄 Shutting down Type-Based MABWiserEngine...")
        self.experts.clear()
        self.session_shown_watches.clear()
        logger.info("✅ Type-Based MABWiserEngine shutdown complete")
    
    def get_expert_stats(self) -> Dict[str, Any]:
        """Get statistics about type-based experts."""
        type_stats = {}
        for watch_type, watch_ids in self.type_to_watch_ids.items():
            type_stats[watch_type] = {
                'watch_count': len(watch_ids),
                'has_expert': watch_type in self.experts
            }
        
        return {
            'total_types': len(self.type_to_watch_ids),
            'total_experts': len(self.experts),
            'total_watches': len(self.available_watches),
            'type_breakdown': type_stats,
            'algorithm': 'Type-Based MABWiser LinUCB'
        } 