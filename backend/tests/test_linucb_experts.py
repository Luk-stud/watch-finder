"""Tests for LinUCB mixture of experts implementation."""

import numpy as np
import pytest
from datetime import datetime
from typing import Dict, Any

from backend.models.linucb_experts import LinUCBExpert, LinUCBMixtureExperts

def create_test_data(n_watches: int = 100, dimension: int = 32):
    """Create test data for LinUCB experts."""
    # Create random embeddings
    embeddings = np.random.randn(n_watches, dimension)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create watch data
    watch_data = []
    for i in range(n_watches):
        watch = {
            'index': i,
            'brand': f'Brand_{i % 10}',
            'model': f'Model_{i}',
            'price': float(1000 + i * 100),
            'specs': {
                'watch_type': f'Type_{i % 5}'
            }
        }
        watch_data.append(watch)
    
    return embeddings, watch_data

def create_test_user_state() -> Dict[str, Any]:
    """Create test user state."""
    return {
        'feedback_history': [],
        'seen_watches': set(),
        'session_start': datetime.now(),
        'engagement_level': 'exploring'
    }

def test_linucb_expert_initialization():
    """Test LinUCB expert initialization."""
    dimension = 32
    expert = LinUCBExpert(dimension=dimension)
    
    assert expert.A.shape == (dimension, dimension)
    assert expert.b.shape == (dimension,)
    assert expert.theta.shape == (dimension,)
    assert expert.total_rewards == 0.0
    assert expert.num_pulls == 0

def test_linucb_expert_ucb_score():
    """Test UCB score calculation."""
    dimension = 32
    expert = LinUCBExpert(dimension=dimension)
    context = np.random.randn(dimension)
    context = context / np.linalg.norm(context)
    
    expected_reward, uncertainty = expert.get_ucb_score(context)
    assert isinstance(expected_reward, float)
    assert isinstance(uncertainty, float)
    assert uncertainty > 0  # Should have positive uncertainty

def test_linucb_expert_update():
    """Test expert update with feedback."""
    dimension = 32
    expert = LinUCBExpert(dimension=dimension)
    context = np.random.randn(dimension)
    context = context / np.linalg.norm(context)
    
    # Initial state
    initial_theta = expert.theta.copy()
    
    # Update with positive reward
    expert.update(context, reward=1.0)
    
    assert expert.num_pulls == 1
    assert expert.total_rewards == 1.0
    assert not np.array_equal(expert.theta, initial_theta)  # Theta should update

def test_mixture_experts_initialization():
    """Test mixture of experts initialization."""
    embeddings, watch_data = create_test_data()
    mixture = LinUCBMixtureExperts(embeddings, watch_data)
    
    assert len(mixture.experts) > 0
    assert mixture.meta_counts.shape == (len(mixture.experts),)
    assert mixture.meta_rewards.shape == (len(mixture.experts),)

def test_mixture_experts_recommendations():
    """Test getting recommendations from mixture of experts."""
    embeddings, watch_data = create_test_data()
    mixture = LinUCBMixtureExperts(embeddings, watch_data)
    user_state = create_test_user_state()
    
    # Get initial recommendations
    recommendations = mixture.get_recommendations(user_state)
    assert len(recommendations) > 0
    for rec in recommendations:
        assert 'index' in rec
        assert 'ucb_score' in rec
        assert 'expected_reward' in rec
        assert 'uncertainty' in rec
        assert 'expert_id' in rec

def test_mixture_experts_feedback():
    """Test updating experts with feedback."""
    embeddings, watch_data = create_test_data()
    mixture = LinUCBMixtureExperts(embeddings, watch_data)
    user_state = create_test_user_state()
    
    # Get initial recommendations
    recommendations = mixture.get_recommendations(user_state)
    first_rec = recommendations[0]
    
    # Add positive feedback
    mixture.update(
        watch_index=first_rec['index'],
        reward=1.0,
        user_state=user_state
    )
    
    # Check that at least one expert was updated
    expert_stats = mixture.get_expert_stats()
    assert any(stat['num_pulls'] > 0 for stat in expert_stats)
    assert any(stat['total_rewards'] > 0 for stat in expert_stats)

def test_mixture_experts_cold_start():
    """Test cold start recommendations."""
    embeddings, watch_data = create_test_data()
    mixture = LinUCBMixtureExperts(embeddings, watch_data)
    user_state = create_test_user_state()
    
    # Get recommendations without any feedback
    recommendations = mixture.get_recommendations(user_state)
    assert len(recommendations) > 0
    
    # Should have high uncertainty in cold start
    assert all(rec['uncertainty'] > 0 for rec in recommendations)

def test_mixture_experts_expert_selection():
    """Test expert selection mechanism."""
    embeddings, watch_data = create_test_data()
    mixture = LinUCBMixtureExperts(embeddings, watch_data)
    user_state = create_test_user_state()
    
    # Initial selection should prioritize exploration
    context = np.zeros(embeddings.shape[1])
    expert_idx, expert = mixture.select_expert(context)
    assert expert_idx >= 0
    assert expert_idx < len(mixture.experts)
    
    # Add feedback to make some experts more promising
    for i in range(5):
        mixture.update(i, reward=1.0, user_state=user_state)
    
    # Selection should now favor experts with good performance
    expert_idx2, expert2 = mixture.select_expert(context)
    assert expert_idx2 >= 0  # Should still be valid
    
def test_mixture_experts_context_extraction():
    """Test user context extraction."""
    embeddings, watch_data = create_test_data()
    mixture = LinUCBMixtureExperts(embeddings, watch_data)
    user_state = create_test_user_state()
    
    # Empty feedback should give zero context
    context = mixture._extract_user_context(user_state)
    assert np.allclose(context, np.zeros(embeddings.shape[1]))
    
    # Add some feedback
    user_state['feedback_history'].append({
        'type': 'like',
        'watch_index': 0,
        'timestamp': datetime.now(),
        'confidence': 0.8
    })
    
    # Should now have non-zero context
    context = mixture._extract_user_context(user_state)
    assert not np.allclose(context, np.zeros(embeddings.shape[1]))
    assert np.allclose(np.linalg.norm(context), 1.0)  # Should be normalized 