#!/usr/bin/env python3
"""
Modern Session Manager for Watch Recommendation System
=====================================================

Advanced session management with:
- Async/await support for high performance
- Real-time session state synchronization
- Intelligent session cleanup and optimization
- Advanced caching and memory management
- Multi-threaded session handling
- Comprehensive session analytics

Author: Watch Finder AI Team
Version: 3.0.0
"""

import asyncio
import uuid
import time
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import logging
from dataclasses import dataclass, field
import pickle
import json
import weakref

from .modern_recommendation_engine import ModernRecommendationEngine, RecommendationRequest, RecommendationResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SessionState:
    """Advanced session state with comprehensive tracking."""
    session_id: str
    created_at: datetime
    last_activity: datetime
    liked_indices: List[int] = field(default_factory=list)
    disliked_indices: List[int] = field(default_factory=list)
    seen_indices: Set[int] = field(default_factory=set)
    step: int = 0
    total_interactions: int = 0
    user_agent: str = ""
    ip_address: str = ""
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
    
    def add_feedback(self, watch_index: int, feedback_type: str):
        """Add feedback and update metrics."""
        self.update_activity()
        self.total_interactions += 1
        self.seen_indices.add(watch_index)
        
        if feedback_type == 'like':
            if watch_index not in self.liked_indices:
                self.liked_indices.append(watch_index)
        elif feedback_type == 'dislike':
            if watch_index not in self.disliked_indices:
                self.disliked_indices.append(watch_index)
    
    def get_session_duration(self) -> float:
        """Get session duration in minutes."""
        return (self.last_activity - self.created_at).total_seconds() / 60
    
    def is_expired(self, timeout_minutes: int = 120) -> bool:
        """Check if session is expired."""
        return (datetime.now() - self.last_activity).total_seconds() > (timeout_minutes * 60)

class ModernSessionManager:
    """
    Modern session manager with advanced features for high-performance
    recommendation serving.
    """
    
    def __init__(self, 
                 embeddings, 
                 watch_data: List[Dict[str, Any]],
                 embeddings_pre_normalized: bool = False,
                 use_faiss: bool = False):
        """
        Initialize the modern session manager.
        
        Args:
            embeddings: Watch embeddings matrix
            watch_data: List of watch metadata
            embeddings_pre_normalized: Whether embeddings are pre-normalized
            use_faiss: Whether to use FAISS for similarity search
        """
        # Initialize the modern recommendation engine
        self.recommendation_engine = ModernRecommendationEngine(
            embeddings=embeddings,
            watch_data=watch_data,
            embeddings_pre_normalized=embeddings_pre_normalized,
            use_faiss=use_faiss
        )
        
        # Session management
        self.sessions: Dict[str, SessionState] = {}
        self.session_lock = threading.RLock()
        
        # Performance tracking
        self.global_metrics = {
            'total_sessions_created': 0,
            'total_recommendations_served': 0,
            'avg_session_duration': 0.0,
            'total_user_interactions': 0,
            'avg_recommendations_per_session': 0.0,
            'peak_concurrent_sessions': 0,
            'cache_hit_rate': 0.0
        }
        
        # Session cleanup
        self.cleanup_interval = 300  # 5 minutes
        self.session_timeout = 120   # 2 hours
        self._cleanup_thread = None
        self._start_cleanup_thread()
        
        # Advanced features
        self.session_cache = {}  # Recommendation caching
        self.popular_watches = defaultdict(int)  # Global popularity tracking
        self.session_analytics = deque(maxlen=1000)  # Recent session analytics
        
        logger.info("🚀 Modern Session Manager initialized")
        logger.info(f"📊 Dataset: {len(watch_data)} watches")
        logger.info(f"🔍 Recommendation engine: {type(self.recommendation_engine).__name__}")

    def _start_cleanup_thread(self):
        """Start background thread for session cleanup."""
        def cleanup_sessions():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self._cleanup_expired_sessions()
                except Exception as e:
                    logger.error(f"Session cleanup error: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
        self._cleanup_thread.start()
        logger.info("🧹 Started session cleanup thread")

    def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        with self.session_lock:
            expired_sessions = [
                session_id for session_id, session in self.sessions.items()
                if session.is_expired(self.session_timeout)
            ]
            
            for session_id in expired_sessions:
                session = self.sessions.pop(session_id)
                
                # Record session analytics before cleanup
                self.session_analytics.append({
                    'session_id': session_id,
                    'duration_minutes': session.get_session_duration(),
                    'total_interactions': session.total_interactions,
                    'likes_count': len(session.liked_indices),
                    'dislikes_count': len(session.disliked_indices),
                    'ended_at': datetime.now().isoformat(),
                    'reason': 'expired'
                })
                
                # Clean up session cache
                if session_id in self.session_cache:
                    del self.session_cache[session_id]
            
            if expired_sessions:
                logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")

    async def create_session(self, 
                           user_agent: str = "", 
                           ip_address: str = "",
                           metadata: Dict[str, Any] = None) -> str:
        """
        Create a new session.
        
        Args:
            user_agent: User agent string
            ip_address: Client IP address
            metadata: Additional session metadata
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        session = SessionState(
            session_id=session_id,
            created_at=current_time,
            last_activity=current_time,
            user_agent=user_agent,
            ip_address=ip_address,
            session_metadata=metadata or {}
        )
        
        with self.session_lock:
            self.sessions[session_id] = session
            self.global_metrics['total_sessions_created'] += 1
            
            # Update peak concurrent sessions
            current_sessions = len(self.sessions)
            if current_sessions > self.global_metrics['peak_concurrent_sessions']:
                self.global_metrics['peak_concurrent_sessions'] = current_sessions
        
        logger.info(f"➕ Created session {session_id}")
        return session_id

    async def get_smart_seeds(self, 
                            session_id: str, 
                            num_seeds: int = 7) -> List[Dict[str, Any]]:
        """
        Get smart seeds for session initialization.
        
        Args:
            session_id: Session identifier
            num_seeds: Number of seeds to return
            
        Returns:
            List of seed watches
        """
        if not self._validate_session(session_id):
            raise ValueError(f"Invalid session ID: {session_id}")
        
        # Get smart seeds from recommendation engine
        seeds = self.recommendation_engine.get_smart_seeds(num_seeds)
        
        # Update session state
        session = self.sessions[session_id]
        session.update_activity()
        session.step = 0
        
        # Track seen watches
        for seed in seeds:
            session.seen_indices.add(seed['index'])
        
        logger.info(f"🌱 Provided {len(seeds)} smart seeds for session {session_id}")
        return seeds

    async def get_recommendations(self, 
                                session_id: str,
                                liked_indices: List[int],
                                disliked_indices: List[int],
                                current_candidates: List[int],
                                num_recommendations: int = 7,
                                step: int = 1) -> Dict[str, Any]:
        """
        Get recommendations for a session.
        
        Args:
            session_id: Session identifier
            liked_indices: List of liked watch indices
            disliked_indices: List of disliked watch indices
            current_candidates: Current watch candidates to exclude
            num_recommendations: Number of recommendations to return
            step: Current recommendation step
            
        Returns:
            Recommendation response with metadata
        """
        if not self._validate_session(session_id):
            raise ValueError(f"Invalid session ID: {session_id}")
        
        start_time = time.time()
        session = self.sessions[session_id]
        
        # Update session state with feedback
        self._update_session_feedback(session, liked_indices, disliked_indices)
        
        # Create recommendation request
        request = RecommendationRequest(
            user_id=session_id,
            liked_indices=liked_indices,
            disliked_indices=disliked_indices,
            current_candidates=current_candidates,
            num_recommendations=num_recommendations,
            exploration_factor=0.3,
            diversity_threshold=0.7,
            context={
                'step': step,
                'session_metadata': session.session_metadata,
                'total_interactions': session.total_interactions
            }
        )
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        if cache_key in self.session_cache:
            logger.info(f"💾 Cache hit for session {session_id}")
            self.global_metrics['cache_hit_rate'] += 0.1
            cached_result = self.session_cache[cache_key]
            # Update cache timestamp
            cached_result['cached_at'] = datetime.now().isoformat()
            return cached_result
        
        # Get recommendations from engine
        try:
            result: RecommendationResult = await self.recommendation_engine.get_recommendations(request)
            
            # Update session state
            session.step = step
            session.update_activity()
            
            # Track seen watches
            for watch in result.watches:
                session.seen_indices.add(watch['index'])
            
            # Update global popularity
            for watch in result.watches:
                self.popular_watches[watch['index']] += 1
            
            # Prepare response
            response = {
                'status': 'success',
                'recommendations': result.watches,
                'step': step,
                'session_id': session_id,
                'algorithm_used': result.algorithm_used,
                'diversity_score': result.diversity_score,
                'exploration_rate': result.exploration_rate,
                'confidence_scores': result.confidence_scores,
                'user_profile_summary': result.user_profile_summary,
                'next_exploration_suggestions': result.next_exploration_suggestions,
                'processing_time': result.processing_time,
                'performance_metrics': {
                    'total_seen_watches': len(session.seen_indices),
                    'session_duration_minutes': session.get_session_duration(),
                    'total_interactions': session.total_interactions,
                    'likes_count': len(session.liked_indices),
                    'dislikes_count': len(session.disliked_indices)
                },
                'generated_at': datetime.now().isoformat()
            }
            
            # Cache the result
            self.session_cache[cache_key] = response.copy()
            
            # Update global metrics
            self.global_metrics['total_recommendations_served'] += len(result.watches)
            processing_time = time.time() - start_time
            
            logger.info(f"📊 Generated {len(result.watches)} recommendations for session {session_id} "
                       f"using {result.algorithm_used} in {processing_time:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations for session {session_id}: {e}")
            
            # Fallback to smart seeds
            try:
                fallback_seeds = await self.get_smart_seeds(session_id, num_recommendations)
                return {
                    'status': 'success',
                    'recommendations': fallback_seeds,
                    'step': step,
                    'session_id': session_id,
                    'algorithm_used': 'fallback_smart_seeds',
                    'diversity_score': 0.8,
                    'exploration_rate': 1.0,
                    'processing_time': time.time() - start_time,
                    'fallback_reason': str(e),
                    'generated_at': datetime.now().isoformat()
                }
            except Exception as fallback_error:
                logger.error(f"❌ Fallback also failed for session {session_id}: {fallback_error}")
                raise Exception(f"Both recommendation engine and fallback failed: {e}")

    def _validate_session(self, session_id: str) -> bool:
        """Validate that session exists and is not expired."""
        with self.session_lock:
            if session_id not in self.sessions:
                return False
            
            session = self.sessions[session_id]
            if session.is_expired(self.session_timeout):
                # Clean up expired session
                del self.sessions[session_id]
                if session_id in self.session_cache:
                    del self.session_cache[session_id]
                return False
            
            return True

    def _update_session_feedback(self, 
                               session: SessionState, 
                               liked_indices: List[int], 
                               disliked_indices: List[int]):
        """Update session with new feedback."""
        # Process new likes
        for idx in liked_indices:
            if idx not in session.liked_indices:
                session.add_feedback(idx, 'like')
                # Add feedback to recommendation engine
                self.recommendation_engine.add_feedback(session.session_id, idx, 'like')
        
        # Process new dislikes
        for idx in disliked_indices:
            if idx not in session.disliked_indices:
                session.add_feedback(idx, 'dislike')
                # Add feedback to recommendation engine
                self.recommendation_engine.add_feedback(session.session_id, idx, 'dislike')
        
        # Update global metrics
        self.global_metrics['total_user_interactions'] += (
            len(liked_indices) + len(disliked_indices)
        )

    def _generate_cache_key(self, request: RecommendationRequest) -> str:
        """Generate cache key for recommendation request."""
        # Create a hashable representation of the request
        key_data = {
            'user_id': request.user_id,
            'liked_indices': tuple(sorted(request.liked_indices)),
            'disliked_indices': tuple(sorted(request.disliked_indices)),
            'current_candidates': tuple(sorted(request.current_candidates)),
            'num_recommendations': request.num_recommendations,
            'step': request.context.get('step', 0)
        }
        return str(hash(str(key_data)))

    async def add_feedback(self, 
                         session_id: str, 
                         watch_index: int, 
                         feedback_type: str,
                         confidence: float = 0.8) -> bool:
        """
        Add user feedback to session.
        
        Args:
            session_id: Session identifier
            watch_index: Index of the watch
            feedback_type: 'like' or 'dislike'
            confidence: Confidence score for the feedback
            
        Returns:
            Success status
        """
        if not self._validate_session(session_id):
            logger.error(f"❌ Invalid session: {session_id}")
            return False
        
        try:
            # Validate and convert watch_index to ensure it's an integer
            logger.debug(f"🔍 Processing feedback: session={session_id}, watch_index={watch_index} (type: {type(watch_index)}), feedback_type={feedback_type}")
            
            try:
                watch_index = int(watch_index)
                logger.debug(f"✅ Converted watch_index to int: {watch_index}")
            except (ValueError, TypeError) as e:
                logger.error(f"❌ Invalid watch_index type in session {session_id}: {type(watch_index)}, value: {watch_index}")
                return False
            
            # Validate watch_index range
            if watch_index < 0 or watch_index >= len(self.recommendation_engine.watch_data):
                logger.error(f"❌ Invalid watch_index in session {session_id}: {watch_index} (valid range: 0-{len(self.recommendation_engine.watch_data)-1})")
                return False
            
            # Validate feedback_type
            if feedback_type not in ['like', 'dislike']:
                logger.error(f"❌ Invalid feedback_type in session {session_id}: {feedback_type}")
                return False
            
            logger.debug(f"🎯 Adding feedback to session state...")
            session = self.sessions[session_id]
            session.add_feedback(watch_index, feedback_type)
            logger.debug(f"✅ Session state updated successfully")
            
            # Add to recommendation engine
            logger.debug(f"🎯 Adding feedback to recommendation engine...")
            try:
                self.recommendation_engine.add_feedback(session_id, watch_index, feedback_type, confidence)
                logger.debug(f"✅ Recommendation engine updated successfully")
            except Exception as rec_error:
                logger.error(f"❌ Error in recommendation engine add_feedback: {rec_error}")
                # Continue with the rest of the process even if recommendation engine fails
            
            # Update global popularity
            logger.debug(f"🎯 Updating global popularity...")
            if feedback_type == 'like':
                self.popular_watches[watch_index] += 2  # Weighted for likes
            logger.debug(f"✅ Global popularity updated")
            
            # Clear relevant cache entries
            logger.debug(f"🎯 Invalidating session cache...")
            self._invalidate_session_cache(session_id)
            logger.debug(f"✅ Session cache invalidated")
            
            logger.info(f"👍 Added {feedback_type} feedback for watch {watch_index} in session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding feedback for session {session_id}: {e}")
            logger.error(f"❌ Error details: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return False

    def _invalidate_session_cache(self, session_id: str):
        """Invalidate cache entries for a session."""
        keys_to_remove = [
            key for key in self.session_cache.keys()
            if session_id in str(key)
        ]
        for key in keys_to_remove:
            del self.session_cache[key]

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        if not self._validate_session(session_id):
            return None
        
        session = self.sessions[session_id]
        return {
            'session_id': session_id,
            'created_at': session.created_at.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'session_duration_minutes': session.get_session_duration(),
            'total_interactions': session.total_interactions,
            'likes_count': len(session.liked_indices),
            'dislikes_count': len(session.disliked_indices),
            'seen_watches_count': len(session.seen_indices),
            'current_step': session.step,
            'user_agent': session.user_agent,
            'metadata': session.session_metadata
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self.session_lock:
            active_sessions = len(self.sessions)
            
            # Calculate average session duration
            if self.session_analytics:
                avg_duration = sum(s['duration_minutes'] for s in self.session_analytics) / len(self.session_analytics)
            else:
                avg_duration = 0.0
            
            # Calculate average recommendations per session
            total_sessions = self.global_metrics['total_sessions_created']
            total_recommendations = self.global_metrics['total_recommendations_served']
            avg_recs_per_session = total_recommendations / max(1, total_sessions)
            
            return {
                'status': 'healthy',
                'active_sessions': active_sessions,
                'global_metrics': {
                    **self.global_metrics,
                    'avg_session_duration': avg_duration,
                    'avg_recommendations_per_session': avg_recs_per_session
                },
                'recommendation_engine_metrics': self.recommendation_engine.get_performance_summary(),
                'cache_size': len(self.session_cache),
                'popular_watches_tracked': len(self.popular_watches),
                'recent_sessions_analyzed': len(self.session_analytics),
                'system_uptime': time.time(),
                'timestamp': datetime.now().isoformat()
            }

    def get_popular_watches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular watches globally."""
        sorted_watches = sorted(
            self.popular_watches.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        popular_list = []
        for watch_index, popularity_score in sorted_watches[:limit]:
            if watch_index < len(self.recommendation_engine.watch_data):
                watch_data = self.recommendation_engine.watch_data[watch_index].copy()
                watch_data['popularity_score'] = popularity_score
                watch_data['index'] = watch_index
                popular_list.append(watch_data)
        
        return popular_list

    async def cleanup_session(self, session_id: str) -> bool:
        """Manually cleanup a specific session."""
        with self.session_lock:
            if session_id in self.sessions:
                session = self.sessions.pop(session_id)
                
                # Record analytics
                self.session_analytics.append({
                    'session_id': session_id,
                    'duration_minutes': session.get_session_duration(),
                    'total_interactions': session.total_interactions,
                    'likes_count': len(session.liked_indices),
                    'dislikes_count': len(session.disliked_indices),
                    'ended_at': datetime.now().isoformat(),
                    'reason': 'manual_cleanup'
                })
                
                # Clean cache
                self._invalidate_session_cache(session_id)
                
                logger.info(f"🧹 Manually cleaned up session {session_id}")
                return True
        
        return False

    def get_session_analytics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent session analytics."""
        return list(self.session_analytics)[-limit:]

    async def shutdown(self):
        """Graceful shutdown of session manager."""
        logger.info("🛑 Shutting down Modern Session Manager...")
        
        # Stop cleanup thread
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            # Note: In a production system, you'd want a more graceful thread shutdown
            logger.info("🧹 Stopping cleanup thread...")
        
        # Save session analytics if needed
        if self.session_analytics:
            try:
                analytics_data = {
                    'analytics': list(self.session_analytics),
                    'global_metrics': self.global_metrics,
                    'shutdown_at': datetime.now().isoformat()
                }
                with open('session_analytics_backup.json', 'w') as f:
                    json.dump(analytics_data, f, indent=2)
                logger.info("💾 Saved session analytics backup")
            except Exception as e:
                logger.error(f"❌ Failed to save analytics backup: {e}")
        
        logger.info("✅ Session Manager shutdown complete") 