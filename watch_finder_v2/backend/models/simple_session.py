"""
Simple Session Manager for LinUCB Recommendations
==============================================

Handles basic session state without complex tracking.
"""

import uuid
from datetime import datetime
from typing import Set, Dict, Any, List
import numpy as np

class SimpleSession:
    def __init__(self):
        self.id: str = str(uuid.uuid4())
        self.created_at: datetime = datetime.now()
        self.last_activity: datetime = datetime.now()
        
        # Core state
        self.liked_watches: Set[int] = set()
        self.shown_watches: Set[int] = set()
        
        # Simple context
        self.context_vector: np.ndarray = np.zeros(50)  # Match LinUCB dimension
        self.context_updates: int = 0
        
        # Limit shown watches to prevent running out of recommendations
        self.max_shown_watches: int = 200  # Keep track of last 200 shown watches
    
    def add_liked_watch(self, watch_id: int) -> None:
        """Add a watch to liked watches."""
        self.liked_watches.add(watch_id)
        self.last_activity = datetime.now()
    
    def add_shown_watch(self, watch_id: int) -> None:
        """Add a watch to shown watches with limit to prevent exhausting all watches."""
        self.shown_watches.add(watch_id)
        
        # If we exceed the limit, remove the oldest watches
        # This is a simple approximation since sets don't preserve order
        if len(self.shown_watches) > self.max_shown_watches:
            # Convert to list, keep last max_shown_watches items
            shown_list = list(self.shown_watches)
            # Remove roughly 20% of the oldest entries 
            remove_count = len(shown_list) - self.max_shown_watches + 20
            for _ in range(remove_count):
                if shown_list:
                    # Remove from the beginning (approximately oldest)
                    self.shown_watches.discard(shown_list.pop(0))
        
        self.last_activity = datetime.now()
    
    def get_context(self) -> np.ndarray:
        """Get current context vector."""
        return self.context_vector
    
    def update_context(self, new_features: np.ndarray) -> None:
        """Update context vector with new features."""
        # Simple exponential moving average
        alpha = 0.7
        self.context_vector = alpha * new_features + (1 - alpha) * self.context_vector
        self.context_updates += 1
        self.last_activity = datetime.now()
    
    def is_expired(self, timeout_seconds: int = 3600) -> bool:
        """Check if session has expired."""
        elapsed = (datetime.now() - self.last_activity).total_seconds()
        return elapsed > timeout_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'liked_watches': list(self.liked_watches),
            'shown_watches': list(self.shown_watches),
            'context_updates': self.context_updates
        } 