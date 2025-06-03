"""
LinUCB Mixture of Experts implementation for watch recommendations.
Each expert corresponds to a taste cluster and uses LinUCB for contextual bandits.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class LinUCBExpert:
    """Individual LinUCB expert for a specific taste cluster."""
    dimension: int
    alpha: float = 1.0
    lambda_: float = 1.0
    
    # LinUCB state
    A: np.ndarray = field(init=False)  # (d x d) matrix
    b: np.ndarray = field(init=False)  # (d) vector
    theta: np.ndarray = field(init=False)  # (d) vector - current estimate
    
    # Expert metadata
    cluster_centroid: Optional[np.ndarray] = None
    total_rewards: float = 0.0
    num_pulls: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Initialize LinUCB matrices."""
        self.A = self.lambda_ * np.eye(self.dimension)
        self.b = np.zeros(self.dimension)
        self.theta = np.zeros(self.dimension)
    
    def get_ucb_score(self, context: np.ndarray) -> Tuple[float, float]:
        """
        Calculate UCB score for given context.
        Returns (expected_reward, uncertainty)
        """
        A_inv = np.linalg.inv(self.A)
        uncertainty = np.sqrt(context.dot(A_inv).dot(context))
        expected_reward = context.dot(self.theta)
        
        return expected_reward, self.alpha * uncertainty
    
    def update(self, context: np.ndarray, reward: float):
        """Update expert with new observation."""
        self.A += np.outer(context, context)
        self.b += reward * context
        self.theta = np.linalg.solve(self.A, self.b)
        
        self.total_rewards += reward
        self.num_pulls += 1
        self.last_update = datetime.now()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get expert's performance statistics."""
        return {
            'total_rewards': self.total_rewards,
            'avg_reward': self.total_rewards / max(1, self.num_pulls),
            'num_pulls': self.num_pulls,
            'last_update': self.last_update.isoformat()
        }

class LinUCBMixtureExperts:
    """
    Mixture of Experts using LinUCB for watch recommendations.
    Each expert specializes in a different taste cluster.
    """
    
    def __init__(self, 
                 embeddings: np.ndarray,
                 watch_data: List[Dict[str, Any]],
                 n_clusters: int = 10,
                 alpha: float = 1.0,
                 lambda_: float = 1.0,
                 min_cluster_size: int = 3):
        """
        Initialize LinUCB Mixture of Experts.
        
        Args:
            embeddings: Watch embeddings matrix (N x d)
            watch_data: List of watch metadata
            n_clusters: Number of initial taste clusters
            alpha: Exploration parameter
            lambda_: Ridge regression parameter
            min_cluster_size: Minimum cluster size
        """
        self.embeddings = embeddings
        self.watch_data = watch_data
        self.dimension = embeddings.shape[1]
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.lambda_ = lambda_
        self.min_cluster_size = min_cluster_size
        
        # Normalize embeddings
        self.normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Initialize experts
        self.experts: List[LinUCBExpert] = []
        self.initialize_experts()
        
        # Meta-UCB for expert selection
        self.meta_counts = np.zeros(len(self.experts))
        self.meta_rewards = np.zeros(len(self.experts))
        
        logger.info(f"Initialized LinUCB Mixture with {len(self.experts)} experts")
    
    def initialize_experts(self):
        """Initialize experts using clustering."""
        from sklearn.cluster import KMeans
        
        # Cluster the embeddings
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.normalized_embeddings)
        
        # Create experts for sufficiently large clusters
        for i in range(self.n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            if len(cluster_indices) >= self.min_cluster_size:
                expert = LinUCBExpert(
                    dimension=self.dimension,
                    alpha=self.alpha,
                    lambda_=self.lambda_
                )
                expert.cluster_centroid = kmeans.cluster_centers_[i]
                self.experts.append(expert)
    
    def select_expert(self, context: np.ndarray) -> Tuple[int, LinUCBExpert]:
        """
        Select best expert using meta-UCB.
        Returns (expert_index, expert)
        """
        n_experts = len(self.experts)
        total_pulls = np.sum(self.meta_counts)
        
        # Calculate meta-UCB scores
        meta_scores = np.zeros(n_experts)
        for i in range(n_experts):
            if self.meta_counts[i] == 0:
                meta_scores[i] = float('inf')
            else:
                avg_reward = self.meta_rewards[i] / self.meta_counts[i]
                exploration = np.sqrt(2 * np.log(total_pulls) / self.meta_counts[i])
                meta_scores[i] = avg_reward + exploration
        
        expert_idx = np.argmax(meta_scores)
        return expert_idx, self.experts[expert_idx]
    
    def get_recommendations(self, 
                          user_state: Dict[str, Any],
                          n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Get recommendations using the mixture of experts.
        
        Args:
            user_state: Current user state
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended watches with scores
        """
        recommendations = []
        seen_watches = user_state.get('seen_watches', set())
        
        # Get user context (can be enhanced based on available features)
        user_context = self._extract_user_context(user_state)
        
        while len(recommendations) < n_recommendations:
            # Select expert
            expert_idx, expert = self.select_expert(user_context)
            
            # Get recommendations from expert
            expert_recs = self._get_expert_recommendations(
                expert, user_context, seen_watches, n=3
            )
            
            if expert_recs:
                recommendations.extend(expert_recs)
            
            if len(recommendations) >= n_recommendations:
                break
        
        # Sort by UCB score and return top N
        recommendations.sort(key=lambda x: x['ucb_score'], reverse=True)
        return recommendations[:n_recommendations]
    
    def _extract_user_context(self, user_state: Dict[str, Any]) -> np.ndarray:
        """Extract context vector from user state."""
        # This can be enhanced with more features
        if not user_state['feedback_history']:
            return np.zeros(self.dimension)
        
        # Average embedding of liked watches
        liked_embeddings = []
        for feedback in user_state['feedback_history']:
            if feedback['type'] == 'like':
                idx = feedback['watch_index']
                liked_embeddings.append(self.normalized_embeddings[idx])
        
        if liked_embeddings:
            context = np.mean(liked_embeddings, axis=0)
            return context / np.linalg.norm(context)
        
        return np.zeros(self.dimension)
    
    def _get_expert_recommendations(self,
                                  expert: LinUCBExpert,
                                  context: np.ndarray,
                                  seen_watches: set,
                                  n: int = 3) -> List[Dict[str, Any]]:
        """Get recommendations from a specific expert."""
        recommendations = []
        
        # Calculate UCB scores for all watches
        ucb_scores = []
        for i, watch_embedding in enumerate(self.normalized_embeddings):
            if i in seen_watches:
                continue
                
            expected_reward, uncertainty = expert.get_ucb_score(watch_embedding)
            ucb_score = expected_reward + uncertainty
            ucb_scores.append((i, ucb_score, expected_reward, uncertainty))
        
        # Sort by UCB score and get top N
        ucb_scores.sort(key=lambda x: x[1], reverse=True)
        
        for idx, ucb_score, expected_reward, uncertainty in ucb_scores[:n]:
            watch = self.watch_data[idx].copy()
            watch.update({
                'index': int(idx),
                'ucb_score': float(ucb_score),
                'expected_reward': float(expected_reward),
                'uncertainty': float(uncertainty),
                'expert_id': id(expert),
                'algorithm': 'linucb_expert'
            })
            recommendations.append(watch)
        
        return recommendations
    
    def update(self, watch_index: int, reward: float, user_state: Dict[str, Any]):
        """Update experts with new feedback."""
        context = self._extract_user_context(user_state)
        watch_embedding = self.normalized_embeddings[watch_index]
        
        # Find expert that recommended this watch
        expert_found = False
        for i, expert in enumerate(self.experts):
            if expert.num_pulls == 0:
                continue
                
            expected_reward, _ = expert.get_ucb_score(watch_embedding)
            if expected_reward > 0:  # This expert would have recommended it
                expert.update(watch_embedding, reward)
                self.meta_counts[i] += 1
                self.meta_rewards[i] += reward
                expert_found = True
        
        if not expert_found:
            # Update expert closest to watch embedding
            distances = [np.linalg.norm(expert.cluster_centroid - watch_embedding) 
                       for expert in self.experts]
            closest_expert_idx = np.argmin(distances)
            
            self.experts[closest_expert_idx].update(watch_embedding, reward)
            self.meta_counts[closest_expert_idx] += 1
            self.meta_rewards[closest_expert_idx] += reward
    
    def get_expert_stats(self) -> List[Dict[str, Any]]:
        """Get performance statistics for all experts."""
        stats = []
        for i, expert in enumerate(self.experts):
            expert_stats = expert.get_performance_stats()
            expert_stats.update({
                'expert_id': id(expert),
                'meta_pulls': int(self.meta_counts[i]),
                'meta_rewards': float(self.meta_rewards[i])
            })
            stats.append(expert_stats)
        return stats 