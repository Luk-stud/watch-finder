#!/usr/bin/env python3
"""
Working Type-Based Engine Test

Test the type-based engine directly with real data.
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

class SimpleTypeBasedEngine:
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

def test_working_engine():
    """Test the working type-based engine."""
    
    print("=" * 80)
    print("🧪 TESTING WORKING TYPE-BASED ENGINE")
    print("=" * 80)
    
    # Test engine creation
    try:
        start_time = time.time()
        engine = SimpleTypeBasedEngine()
        init_time = time.time() - start_time
        
        print(f"✅ Engine initialized in {init_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Get stats
    try:
        stats = engine.get_expert_stats()
        print(f"\n📊 ENGINE STATS:")
        print(f"   • Total watches: {stats['total_watches']}")
        print(f"   • Total types detected: {stats['total_types']}")
        print(f"   • LinUCB experts created: {stats['total_experts']}")
        print(f"   • Algorithm: {stats['algorithm']}")
        
        print(f"\n🏷️ TYPE BREAKDOWN:")
        for watch_type, info in sorted(stats['type_breakdown'].items(), key=lambda x: x[1]['watch_count'], reverse=True):
            status = "✅" if info['has_expert'] else "❌"
            print(f"   • {watch_type}: {info['watch_count']} watches {status}")
            
    except Exception as e:
        print(f"❌ Failed to get stats: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Sample watches per type
    print(f"\n🔍 SAMPLE WATCHES PER TYPE:")
    for watch_type, watch_ids in sorted(engine.type_to_watch_ids.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        print(f"\n   {watch_type.upper()} ({len(watch_ids)} watches):")
        for watch_id in watch_ids[:3]:  # Show first 3
            watch_data = engine.watch_data.get(watch_id, {})
            brand = watch_data.get('brand', 'Unknown')
            model = watch_data.get('model', 'Unknown')
            price = watch_data.get('price', 0)
            print(f"     • {brand} {model} (${price})")
    
    print(f"\n✅ Type-based engine working with real data!")
    return True

def main():
    """Main test function."""
    success = test_working_engine()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 