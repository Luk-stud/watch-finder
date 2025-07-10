#!/usr/bin/env python3
"""
Fix embeddings structure to match backend expectations
"""

import pickle
import numpy as np
from pathlib import Path

def fix_embeddings_structure():
    """Convert embeddings to match backend SimpleSgdEngine expectations"""
    
    # Load current embeddings
    input_path = "watch_finder_v2/backend/data/precomputed_embeddings.pkl"
    output_path = "watch_finder_v2/backend/data/precomputed_embeddings.pkl"
    
    print(f"📖 Loading embeddings from {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"📊 Current structure:")
    for k, v in data.items():
        print(f"  {k}: {type(v)} - {len(v) if hasattr(v, '__len__') else 'N/A'}")
    
    # Convert to backend-expected format
    print(f"\n🔄 Converting structure...")
    
    # Create final_embeddings dict from items_matrix
    final_embeddings = {}
    for watch_id, idx in data['watch_id_to_idx'].items():
        final_embeddings[watch_id] = data['items_matrix'][idx]
    
    # Create new structure
    new_data = {
        'watch_data': data['watch_data'],
        'final_embeddings': final_embeddings,
        'embedding_dim': data['dim']
    }
    
    print(f"✅ New structure:")
    for k, v in new_data.items():
        print(f"  {k}: {type(v)} - {len(v) if hasattr(v, '__len__') else 'N/A'}")
    
    # Save fixed embeddings
    print(f"\n💾 Saving fixed embeddings to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(new_data, f)
    
    print(f"✅ Embeddings structure fixed successfully!")
    print(f"   • Total watches: {len(new_data['watch_data'])}")
    print(f"   • Embeddings: {len(new_data['final_embeddings'])}")
    print(f"   • Dimension: {new_data['embedding_dim']}")

if __name__ == "__main__":
    fix_embeddings_structure() 