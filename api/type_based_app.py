#!/usr/bin/env python3
"""
Type-Based MABWiser FastAPI App

Simple type-based LinUCB using real watch metadata for type classification
and the same 200D embeddings as FastLinUCB.
"""

import os
import pickle
import time
import logging
import uvicorn
import numpy as np
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# MABWiser imports
from mabwiser.mab import MAB, LearningPolicy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TypeBasedMABWiserEngine:
    """Simple type-based MABWiser LinUCB engine."""
    
    def __init__(self, alpha: float = 0.1, batch_size: int = 5):
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Data storage
        self.watch_data = {}
        self.final_embeddings = {}
        self.available_watches = set()
        self.dim = 200
        
        # Type-based organization
        self.type_of = {}
        self.type_to_watch_ids = defaultdict(list)
        self.experts = {}
        
        # Session tracking
        self.session_shown_watches = {}
        
        # Load data
        self._load_embeddings()
        self._load_watch_metadata()
        self._organize_by_type()
        self._create_type_experts()
        
    def _load_embeddings(self):
        """Load 200D embeddings."""
        logger.info("🚀 Loading 200D embeddings...")
        
        with open('data/precomputed_embeddings.pkl', 'rb') as f:
            data = pickle.load(f)
        
        self.final_embeddings = data['final_embeddings']
        self.dim = data['embedding_dim']
        
        logger.info(f"✅ Loaded {len(self.final_embeddings)} embeddings, dim: {self.dim}D")
    
    def _load_watch_metadata(self):
        """Load real watch metadata."""
        logger.info("📊 Loading real watch metadata...")
        
        with open('data/watch_text_metadata.pkl', 'rb') as f:
            watch_list = pickle.load(f)
        
        # Convert to dict
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
        
        # Only use watches with both metadata and embeddings
        self.available_watches = set(self.watch_data.keys()) & set(self.final_embeddings.keys())
        
        logger.info(f"✅ Loaded metadata for {len(self.watch_data)} watches")
        logger.info(f"✅ {len(self.available_watches)} watches have both metadata and embeddings")
    
    def _organize_by_type(self):
        """Organize watches by type from specs."""
        logger.info("🏷️ Organizing watches by watch_type from specs...")
        
        for watch_id in self.available_watches:
            watch_data = self.watch_data.get(watch_id, {})
            specs = watch_data.get('specs', {})
            
            # Get watch type from specs
            watch_type = self._extract_watch_type(specs)
            
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
    
    def _create_type_experts(self):
        """Create LinUCB experts per type."""
        logger.info("🧠 Creating LinUCB experts per watch type...")
        
        for watch_type, watch_ids in self.type_to_watch_ids.items():
            if len(watch_ids) == 0:
                continue
                
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
    
    def create_session(self, session_id: str):
        """Initialize a new session."""
        self.session_shown_watches[session_id] = set()
        logger.info(f"✅ Created Type-Based MABWiser session {session_id}")
    
    def get_recommendations(self, session_id: str, exclude_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
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
    
    def update(self, session_id: str, watch_id: int, reward: float):
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
    
    def get_expert_stats(self):
        """Get stats about experts."""
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

# Pydantic models
class FeedbackRequest(BaseModel):
    session_id: str
    watch_id: int
    feedback: str

class RecommendationRequest(BaseModel):
    session_id: str
    num_recommendations: Optional[int] = 5

# Initialize the engine
logger.info("🚀 Initializing Type-Based MABWiser Engine...")
engine = None

try:
    engine = TypeBasedMABWiserEngine()
    logger.info("✅ Type-Based MABWiser Engine initialized successfully!")
except Exception as e:
    logger.error(f"❌ Failed to initialize engine: {e}")
    import traceback
    traceback.print_exc()

# Create FastAPI app
app = FastAPI(
    title="Type-Based MABWiser Watch Recommendation API",
    description="Advanced watch recommendation system using type-based MABWiser LinUCB",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Type-Based MABWiser Watch Recommendation API",
        "version": "2.0.0",
        "algorithm": "Type-Based MABWiser LinUCB",
        "status": "ready" if engine else "engine_failed"
    }

@app.get("/stats")
async def get_stats():
    """Get engine statistics."""
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    try:
        stats = engine.get_expert_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Get watch recommendations."""
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    try:
        # Update batch size if requested
        if request.num_recommendations:
            engine.batch_size = request.num_recommendations
        
        recommendations = engine.get_recommendations(request.session_id)
        
        return {
            "session_id": request.session_id,
            "recommendations": recommendations,
            "algorithm": "Type-Based MABWiser LinUCB",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendations for session {request.session_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a watch."""
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    try:
        # Convert feedback to reward
        reward = 1.0 if request.feedback.lower() in ['like', 'love', 'positive', '1'] else 0.0
        
        # Update the engine
        engine.update(request.session_id, request.watch_id, reward)
        
        return {
            "message": "Feedback received",
            "session_id": request.session_id,
            "watch_id": request.watch_id,
            "feedback": request.feedback,
            "reward": reward,
            "algorithm": "Type-Based MABWiser LinUCB"
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("🚀 Starting Type-Based MABWiser Watch Recommendation Server...")
    uvicorn.run(
        "type_based_app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) 