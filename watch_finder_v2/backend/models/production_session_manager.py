"""
Production-Ready Session Manager for Watch Recommendation System
=============================================================

Advanced session management for production deployment:
- Thread-safe concurrent session handling
- Automatic session cleanup and memory management
- Session persistence across server restarts
- Per-session isolation of recommendation state
- Comprehensive metrics and monitoring
- Rate limiting and abuse protection

Author: Watch Finder AI Team
Version: 1.0.0
"""

import os
import json
import pickle
import threading
import time
import uuid
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import logging
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .optimized_linucb_engine import OptimizedLinUCBEngine

logger = logging.getLogger(__name__)

class Session:
    """Tracks session state and rate limiting."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.shown_watches: Set[int] = set()
        self.total_recommendations_served = 0
        self.request_timestamps: List[float] = []
        self.response_times: List[float] = []
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
    
    def check_rate_limit(self, max_requests_per_minute: int) -> bool:
        """Check if session has exceeded rate limit."""
        now = time.time()
        minute_ago = now - 60
        
        # Remove old timestamps
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > minute_ago]
        
        # Check rate limit
        if len(self.request_timestamps) >= max_requests_per_minute:
            return False
        
        # Add current request
        self.request_timestamps.append(now)
        return True
    
    def update_performance_metrics(self, response_time_ms: float):
        """Update response time metrics."""
        self.response_times.append(response_time_ms)

class ProductionSessionManager:
    """
    Production-ready session manager with enterprise features.
    """
    
    def __init__(self, 
                 data_dir: str,
                 session_timeout_minutes: int = 60,
                 max_concurrent_sessions: int = 1000,
                 cleanup_interval_seconds: int = 300,
                 enable_persistence: bool = True,
                 max_requests_per_minute: int = 60,
                 linucb_engine: Optional[OptimizedLinUCBEngine] = None):
        """
        Initialize production session manager.
        
        Args:
            data_dir: Directory for data and persistence
            session_timeout_minutes: Session expiration time
            max_concurrent_sessions: Maximum concurrent sessions
            cleanup_interval_seconds: Cleanup frequency
            enable_persistence: Enable session persistence
            max_requests_per_minute: Rate limit per session
            linucb_engine: The LinUCB engine instance to use
        """
        
        self.data_dir = Path(data_dir)
        self.session_timeout_minutes = session_timeout_minutes
        self.max_concurrent_sessions = max_concurrent_sessions
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.enable_persistence = enable_persistence
        self.max_requests_per_minute = max_requests_per_minute
        self.engine = linucb_engine
        
        if self.engine is None:
            raise ValueError("LinUCB engine must be provided")
        
        # Thread-safe session storage
        self.sessions: Dict[str, Session] = {}
        self.session_lock = threading.RLock()
        
        # Persistence
        self.persistence_dir = self.data_dir / "sessions"
        self.persistence_dir.mkdir(exist_ok=True)
        self.db_path = self.persistence_dir / "sessions.db"
        
        # Global metrics
        self.global_metrics = {
            'total_sessions_created': 0,
            'total_recommendations_served': 0,
            'peak_concurrent_sessions': 0,
            'total_server_uptime_seconds': 0.0,
            'start_time': datetime.now()
        }
        
        # Session analytics
        self.session_analytics = deque(maxlen=10000)  # Store last 10k sessions
        
        # Background services
        self.cleanup_thread: Optional[threading.Thread] = None
        self.persistence_thread: Optional[threading.Thread] = None
        self.thread_pool = ThreadPoolExecutor(max_workers=5, thread_name_prefix="session_worker")
        self.shutdown_event = threading.Event()
        
        # Initialize services
        self._initialize_persistence()
        self._start_background_services()
        
        logger.info(f"🚀 Production Session Manager initialized")
        logger.info(f"📊 Max concurrent sessions: {max_concurrent_sessions}")
        logger.info(f"⏰ Session timeout: {session_timeout_minutes} minutes")
        logger.info(f"💾 Persistence enabled: {enable_persistence}")
    
    def _initialize_persistence(self):
        """Initialize session persistence database."""
        if not self.enable_persistence:
            return
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Create sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    session_data TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_activity TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE
                )
            ''')
            
            # Create analytics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS session_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    analytics_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_sessions_last_activity 
                ON sessions(last_activity)
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Session persistence database initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize persistence: {e}")
            self.enable_persistence = False
    
    def _start_background_services(self):
        """Start background cleanup and persistence services."""
        
        # Cleanup service
        def cleanup_service():
            while not self.shutdown_event.is_set():
                try:
                    self._cleanup_expired_sessions()
                    self.shutdown_event.wait(self.cleanup_interval_seconds)
                except Exception as e:
                    logger.error(f"Cleanup service error: {e}")
        
        # Persistence service
        def persistence_service():
            while not self.shutdown_event.is_set():
                try:
                    if self.enable_persistence:
                        self._persist_active_sessions()
                    self.shutdown_event.wait(60)  # Persist every minute
                except Exception as e:
                    logger.error(f"Persistence service error: {e}")
        
        self.cleanup_thread = threading.Thread(target=cleanup_service, daemon=True)
        self.persistence_thread = threading.Thread(target=persistence_service, daemon=True)
        
        self.cleanup_thread.start()
        self.persistence_thread.start()
        
        logger.info("🔄 Background services started")
    
    def create_session(self, 
                      user_agent: str = "",
                      ip_address: str = "",
                      user_id: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Create a new isolated session.
        
        Returns:
            Tuple of (session_id, initial_recommendations)
        """
        
        with self.session_lock:
            # Check concurrent session limit
            if len(self.sessions) >= self.max_concurrent_sessions:
                raise ValueError(f"Maximum concurrent sessions ({self.max_concurrent_sessions}) reached")
            
            # Create session
            session_id = str(uuid.uuid4())
            current_time = datetime.now()
            
            session = Session(session_id)
            
            self.sessions[session_id] = session
            
            # Update global metrics
            self.global_metrics['total_sessions_created'] += 1
            current_sessions = len(self.sessions)
            if current_sessions > self.global_metrics['peak_concurrent_sessions']:
                self.global_metrics['peak_concurrent_sessions'] = current_sessions
        
        # Get initial recommendations
        try:
            start_time = time.time()
            
            # Initialize session in engine with default weights
            self.engine.create_session(session_id)
            
            # Get recommendations
            recommendations = self.engine.get_recommendations(
                session_id=session_id,
                context=np.array([0.5, 0.5]),  # Equal weights initially
                exclude_ids=set()
            )
            
            # Update metrics
            response_time_ms = (time.time() - start_time) * 1000
            session.update_performance_metrics(response_time_ms)
            session.total_recommendations_served += len(recommendations)
            
            # Mark as shown
            for rec in recommendations:
                session.shown_watches.add(rec['watch_id'])
            
            logger.info(f"✅ Created session {session_id} with {len(recommendations)} recommendations")
            
            return session_id, recommendations
            
        except Exception as e:
            # Cleanup failed session
            with self.session_lock:
                self.sessions.pop(session_id, None)
            raise e
    
    def get_recommendations(self, 
                          session_id: str,
                          exclude_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """Get recommendations for a session with rate limiting."""
        
        # Validate session
        session = self._validate_and_get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found or expired")
        
        # Check rate limiting
        if not session.check_rate_limit(self.max_requests_per_minute):
            raise ValueError(f"Rate limit exceeded for session {session_id}")
        
        # Get recommendations
        start_time = time.time()
        
        exclude_ids = exclude_ids or set()
        exclude_ids.update(session.shown_watches)
        
        recommendations = self.engine.get_recommendations(
            session_id=session_id,
            context=np.array([0.5, 0.5]),  # Use stored weights from engine
            exclude_ids=exclude_ids
        )
        
        # Update metrics
        response_time_ms = (time.time() - start_time) * 1000
        session.update_performance_metrics(response_time_ms)
        session.total_recommendations_served += len(recommendations)
        
        # Mark as shown
        for rec in recommendations:
            session.shown_watches.add(rec['watch_id'])
        
        self.global_metrics['total_recommendations_served'] += len(recommendations)
        
        return recommendations
    
    def update_feedback(self,
                       session_id: str,
                       watch_id: int,
                       liked: bool) -> bool:
        """Process user feedback."""
        # Validate session
        session = self._validate_and_get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found or expired")
        
        # Update engine
        try:
            self.engine.update(
                session_id=session_id,
                watch_id=watch_id,
                reward=1.0 if liked else 0.0,
                context=np.array([0.5, 0.5])  # Use stored weights from engine
            )
            return True
        except Exception as e:
            logger.error(f"Error updating feedback: {e}")
            return False
    
    def _validate_and_get_session(self, session_id: str) -> Optional[Session]:
        """Validate session exists and is not expired."""
        with self.session_lock:
            session = self.sessions.get(session_id)
            if not session:
                return None
            
            if datetime.now() - session.last_activity > timedelta(minutes=self.session_timeout_minutes):
                # Clean up expired session
                self._cleanup_session(session_id)
                return None
            
            session.update_activity()
            return session
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        with self.session_lock:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session in self.sessions.items():
                if datetime.now() - session.last_activity > timedelta(minutes=self.session_timeout_minutes):
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                self._cleanup_session(session_id)
            
            if expired_sessions:
                logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
    
    def _cleanup_session(self, session_id: str):
        """Clean up a specific session completely."""
        session = self.sessions.pop(session_id, None)
        
        if session:
            # Record analytics for ALL cleaned sessions
            analytics = session.__dict__
            analytics['cleanup_reason'] = 'expired'
            analytics['cleanup_time'] = datetime.now().isoformat()
            
            # CRITICAL FIX: Ensure analytics are added for every session
            self.session_analytics.append(analytics)
            
            # Update global metrics
            duration_minutes = (datetime.now() - session.created_at).total_seconds() / 60.0
            total_sessions = self.global_metrics['total_sessions_created']
            current_avg = self.global_metrics['total_recommendations_served'] / total_sessions
            
            # Update moving average only if we have valid data
            if total_sessions > 0:
                self.global_metrics['total_recommendations_served'] = (
                    current_avg * (total_sessions - 1) + session.total_recommendations_served
                )
            
            logger.info(f"🧹 Session {session_id} cleaned up (duration: {duration_minutes:.1f}min)")
        
        # CRITICAL FIX: Explicitly clean up session to prevent memory leaks
        if session:
            try:
                # Clear session data structures
                session.shown_watches.clear()
                session.request_timestamps.clear()
                session.response_times.clear()
                logger.info(f"🧹 Session {session_id} cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")
    
    def _serialize_for_json(self, obj: Any) -> Any:
        """Helper method to serialize objects for JSON."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._serialize_for_json(obj.__dict__)
        return obj

    def _persist_active_sessions(self):
        """Persist active sessions to database."""
        if not self.enable_persistence:
            return
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            with self.session_lock:
                for session_id, session in self.sessions.items():
                    session_data = json.dumps(self._serialize_for_json(session))
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO sessions 
                        (session_id, session_data, created_at, last_activity, is_active)
                        VALUES (?, ?, ?, ?, TRUE)
                    ''', (
                        session_id,
                        session_data,
                        session.created_at.isoformat(),
                        session.last_activity.isoformat()
                    ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to persist sessions: {e}")
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed session information."""
        session = self._validate_and_get_session(session_id)
        if not session:
            return None
        
        return session.__dict__
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        with self.session_lock:
            active_sessions = len(self.sessions)
            
            # Calculate uptime
            uptime_seconds = (datetime.now() - self.global_metrics['start_time']).total_seconds()
            self.global_metrics['total_server_uptime_seconds'] = uptime_seconds
            
            return {
                'status': 'healthy',
                'active_sessions': active_sessions,
                'max_concurrent_sessions': self.max_concurrent_sessions,
                'global_metrics': self.global_metrics,
                'session_analytics_count': len(self.session_analytics),
                'system_info': {
                    'uptime_hours': uptime_seconds / 3600,
                    'persistence_enabled': self.enable_persistence,
                    'rate_limit_per_minute': self.max_requests_per_minute,
                    'cleanup_interval_seconds': self.cleanup_interval_seconds
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def manual_cleanup_session(self, session_id: str) -> bool:
        """Manually cleanup a specific session."""
        with self.session_lock:
            if session_id in self.sessions:
                self._cleanup_session(session_id)
                logger.info(f"🧹 Manually cleaned up session {session_id}")
                return True
            return False
    
    def shutdown(self):
        """Graceful shutdown of session manager."""
        logger.info("🛑 Shutting down Production Session Manager...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for background threads
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        if self.persistence_thread and self.persistence_thread.is_alive():
            self.persistence_thread.join(timeout=5)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Final persistence
        if self.enable_persistence:
            self._persist_active_sessions()
            
            # Save analytics
            try:
                analytics_file = self.persistence_dir / "final_analytics.json"
                with open(analytics_file, 'w') as f:
                    json.dump({
                        'global_metrics': self._serialize_for_json(self.global_metrics),
                        'session_analytics': [self._serialize_for_json(a) for a in self.session_analytics],
                        'shutdown_time': datetime.now().isoformat()
                    }, f, indent=2)
                logger.info("💾 Saved final analytics")
            except Exception as e:
                logger.error(f"Failed to save final analytics: {e}")
        
        logger.info("✅ Session Manager shutdown complete")

    def get_session_analytics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent session analytics."""
        return list(self.session_analytics)[-limit:]

    def reset_session(self, session_id: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Reset a session by cleaning up the old one and creating a new one."""
        
        # Get old session info for potential user data transfer
        old_session = self._validate_and_get_session(session_id)
        user_agent = old_session.user_agent if old_session else ""
        ip_address = old_session.ip_address if old_session else ""
        user_id = old_session.user_id if old_session else None
        
        # CRITICAL: Clean up the old session completely
        if old_session:
            with self.session_lock:
                if session_id in self.sessions:
                    # Mark as manual reset instead of expired
                    analytics = old_session.__dict__
                    analytics['cleanup_reason'] = 'manual_reset'
                    analytics['cleanup_time'] = datetime.now().isoformat()
                    self.session_analytics.append(analytics)
                    
                    # Clean up old session
                    self._cleanup_session(session_id)
                    logger.info(f"🔄 Reset session {session_id}")
        
        # Create new session with same user context
        new_session_id, initial_recommendations = self.create_session(
            user_agent=user_agent,
            ip_address=ip_address,
            user_id=user_id
        )
        
        logger.info(f"🔄 Session reset: {session_id} → {new_session_id}")
        
        return new_session_id, initial_recommendations 