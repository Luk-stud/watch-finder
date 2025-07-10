#!/usr/bin/env python3
"""
Simple MABWiser Test with Real Data

Test MABWiser LinUCB with actual watch data to understand what types we have.
"""

import os
import pickle
import time
import logging
import numpy as np
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict

# MABWiser imports
from mabwiser.mab import MAB, LearningPolicy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_watch_data():
    """Load the precomputed watch data."""
    print("🚀 Loading watch data...")
    
    precomputed_path = os.path.join("data", 'precomputed_embeddings.pkl')
    
    if not os.path.exists(precomputed_path):
        print(f"❌ File not found: {precomputed_path}")
        return None, None
    
    with open(precomputed_path, 'rb') as f:
        data = pickle.load(f)
    
    watch_data = data['watch_data']
    embeddings = data['final_embeddings']
    
    print(f"✅ Loaded {len(watch_data)} watches with {len(embeddings)} embeddings")
    return watch_data, embeddings

def extract_watch_type(watch_data: Dict[str, Any]) -> str:
    """Extract watch type from watch data."""
    # Method 1: Use category field if available
    if 'category' in watch_data:
        return watch_data['category'].lower()
    
    # Method 2: Use type field if available  
    if 'type' in watch_data:
        return watch_data['type'].lower()
    
    # Method 3: Infer from description/model
    description = str(watch_data.get('description', '')).lower()
    model = str(watch_data.get('model', '')).lower()
    brand = str(watch_data.get('brand', '')).lower()
    text = f"{description} {model} {brand}"
    
    # Simple rule-based type classification
    if any(word in text for word in ['dive', 'diving', 'submariner', 'seamaster']):
        return 'dive'
    elif any(word in text for word in ['dress', 'formal', 'classic', 'heritage']):
        return 'dress'
    elif any(word in text for word in ['sport', 'chrono', 'racing', 'speedmaster']):
        return 'sport'
    elif any(word in text for word in ['pilot', 'aviation', 'aviator', 'flieger']):
        return 'pilot'
    elif any(word in text for word in ['gmt', 'travel', 'dual']):
        return 'travel'
    else:
        return 'general'

def analyze_watch_types():
    """Analyze what types of watches we have in the dataset."""
    print("=" * 80)
    print("🔍 ANALYZING WATCH TYPES IN DATASET")
    print("=" * 80)
    
    # Load data
    watch_data, embeddings = load_watch_data()
    if watch_data is None:
        return False
    
    # Organize by type
    type_to_watches = defaultdict(list)
    type_of = {}
    
    for watch_id, data in watch_data.items():
        watch_type = extract_watch_type(data)
        type_of[watch_id] = watch_type
        type_to_watches[watch_type].append(watch_id)
    
    # Print type breakdown
    print(f"\n🏷️ TYPE BREAKDOWN:")
    total_watches = len(watch_data)
    for watch_type, watch_ids in sorted(type_to_watches.items(), key=lambda x: len(x[1]), reverse=True):
        count = len(watch_ids)
        percentage = (count / total_watches) * 100
        print(f"   • {watch_type}: {count} watches ({percentage:.1f}%)")
    
    # Show sample watches per type
    print(f"\n🔍 SAMPLE WATCHES PER TYPE:")
    for watch_type, watch_ids in sorted(type_to_watches.items()):
        print(f"\n   {watch_type.upper()}:")
        sample_ids = watch_ids[:3]  # Show first 3
        for watch_id in sample_ids:
            data = watch_data[watch_id]
            brand = data.get('brand', 'Unknown')
            model = data.get('model', 'Unknown')
            print(f"     • {brand} {model}")
    
    return type_to_watches, type_of, embeddings

def test_mabwiser_with_types():
    """Test MABWiser LinUCB with our watch types."""
    print("\n" + "=" * 80)
    print("🧪 TESTING MABWISER LINUCB WITH WATCH TYPES")
    print("=" * 80)
    
    # Get type data
    result = analyze_watch_types()
    if not result:
        return False
    
    type_to_watches, type_of, embeddings = result
    
    # Test with one type (the largest one)
    largest_type = max(type_to_watches.keys(), key=lambda t: len(type_to_watches[t]))
    watch_ids = type_to_watches[largest_type]
    
    print(f"\n🎯 Testing with '{largest_type}' type ({len(watch_ids)} watches)")
    
    try:
        # Create MAB for this type
        mab = MAB(
            arms=watch_ids,
            learning_policy=LearningPolicy.LinUCB(alpha=0.1),
            neighborhood_policy=None
        )
        print(f"✅ Created LinUCB MAB for '{largest_type}' type")
        
        # Initialize with some training data first
        sample_watches = watch_ids[:10]  # Use first 10 watches
        contexts = np.array([embeddings[w] for w in sample_watches])
        
        print(f"🔧 Initializing MAB with training data...")
        # Create some initial training data (simulate random exploration)
        initial_decisions = sample_watches[:3]  # First 3 watches
        initial_rewards = [0.5, 1.0, 0.0]  # Mixed feedback
        initial_contexts = np.array([embeddings[w] for w in initial_decisions])
        
        mab.fit(
            decisions=initial_decisions,
            rewards=initial_rewards,
            contexts=initial_contexts
        )
        print(f"✅ Initialized MAB with {len(initial_decisions)} training samples")
        
        # Test prediction after initialization
        print(f"🔮 Testing prediction after initialization...")
        predictions = mab.predict(contexts)
        print(f"✅ Got predictions: {predictions[:5]}...")  # Show first 5
        
        # Test with additional feedback
        print(f"👍 Testing additional feedback update...")
        chosen_watch = sample_watches[5]  # Different watch
        chosen_context = embeddings[chosen_watch].reshape(1, -1)
        
        mab.partial_fit(
            decisions=[chosen_watch],
            rewards=[1.0],
            contexts=chosen_context
        )
        print(f"✅ Updated MAB with positive feedback for watch {chosen_watch}")
        
        # Test prediction after feedback
        print(f"🔮 Testing prediction after additional feedback...")
        new_predictions = mab.predict(contexts)
        print(f"✅ Got new predictions: {new_predictions[:5]}...")
        
        # Test expectations (confidence scores)
        print(f"📊 Testing expectation scores...")
        expectations = mab.predict_expectations(contexts)
        print(f"✅ Got expectation scores: {expectations[:5]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ MABWiser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 MABWISER SIMPLE TEST WITH REAL WATCH DATA")
    
    success = test_mabwiser_with_types()
    
    if success:
        print("\n✅ All tests completed successfully!")
        print("🎉 MABWiser works with our watch data!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 