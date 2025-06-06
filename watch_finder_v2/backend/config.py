"""
Configuration settings for the Watch Finder backend.
"""

import os
from pathlib import Path

# Base directory of the backend
BASE_DIR = Path(__file__).resolve().parent

# Data directory
DATA_DIR = BASE_DIR / "data"

# API settings
API_VERSION = "v2"
DEFAULT_PORT = 5001

# LinUCB parameters - OPTIMIZED FOR FASTER EXPLOITATION WITH DIVERSITY
LINUCB_DIMENSION = 68  # Optimal dimension for 80.2% information retention
LINUCB_ALPHA = 0.5  # REDUCED from 1.0 - less exploration, faster exploitation
RECOMMENDATION_BATCH_SIZE = 5

# Multi-Expert LinUCB settings - LIKE-DRIVEN DYNAMIC CLUSTERING
NUM_EXPERTS = 0  # CHANGED: Start with no experts, create from likes
MAX_EXPERTS = 4  # UPDATED: Increased to 4 experts for better preference clustering
EXPERT_RECOMMENDATION_SIZE = 3  # INCREASED from 2 - more exploitation recommendations
ENABLE_MULTI_EXPERT = True  # Enable/disable multi-expert system

# Like-Driven Dynamic Clustering settings - NEW
MIN_LIKES_FOR_FIRST_EXPERT = 10  # Minimum likes before creating first expert
MIN_LIKES_FOR_NEW_EXPERT = 5   # Minimum likes for additional experts
LIKE_CLUSTERING_SIMILARITY_THRESHOLD = 0.75  # UPDATED: Raised for tighter clusters (proper cosine similarity)
PURE_EXPLORATION_MODE = True  # Start in pure exploration until enough likes

# Dynamic expert creation settings - PREFERENCE-BASED
EXPERT_SIMILARITY_THRESHOLD = 0.75  # UPDATED: Raised for more distinct clusters (proper cosine similarity)
MIN_EXPERT_SIZE = 3  # Minimum watches needed to justify a new expert
UNASSIGNED_EXPLORATION_RATIO = 1.0  # 100% exploration initially, reduces after likes
MAX_EXPLORATION_ROUNDS = 50  # INCREASED - allow more exploration for likes
DIVERSITY_BOOST_ROUNDS = 10  # INCREASED - more diversity for like collection

# Session settings
SESSION_TIMEOUT_SECONDS = 3600  # 1 hour
SESSION_TIMEOUT_MINUTES = 60    # 1 hour in minutes

# Feature extraction settings
FEATURE_CACHE_SIZE = 1000

# Development settings
DEBUG = os.getenv("FLASK_ENV") == "development"

# Ensure required directories exist
DATA_DIR.mkdir(exist_ok=True) 