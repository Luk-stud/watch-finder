# Dynamic Multi-Expert LinUCB Configuration - HYBRID APPROACH
DYNAMIC_MULTI_EXPERT_CONFIG = {
    'dim': 68,  # OPTIMIZED: 80.2% information retention (vs 50.4% at 20D)
    'alpha': 1.2,  # Balanced exploration vs exploitation
    'batch_size': 4,  # Good batch size for quality
    'num_experts': 2,  # Start with 2 pre-clustered experts
    'max_experts': 6,  # Can grow to 6 total experts
    'similarity_threshold': 0.7,  # Higher threshold for quality
    'min_expert_size': 50,  # Larger expert sizes for better specialization
    'unassigned_ratio': 0.3,  # Keep 30% unassigned for exploration
    'pre_cluster_method': 'kmeans',  # Use K-means for initial clustering
    'initialization_strategy': 'hybrid',  # NEW: Hybrid pre-clustered + dynamic
} 