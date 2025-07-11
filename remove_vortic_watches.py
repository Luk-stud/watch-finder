#!/usr/bin/env python3
"""
Remove Vortic Watch Company watches from the embeddings dataset
"""

import pickle
import numpy as np
import os
from pathlib import Path

def remove_vortic_watches():
    """Remove all Vortic Watch Company watches from the embeddings."""
    
    # Paths
    input_path = "watch_finder_v2/backend/data/precomputed_embeddings.pkl"
    output_path = "watch_finder_v2/backend/data/precomputed_embeddings.pkl"
    backup_path = "watch_finder_v2/backend/data/precomputed_embeddings_backup.pkl"
    
    print(f"📖 Loading embeddings from {input_path}")
    
    # Load current embeddings
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"📊 Current dataset stats:")
    print(f"  - Total watches: {len(data['watch_data'])}")
    print(f"  - final_embeddings type: {type(data['final_embeddings'])}")
    print(f"  - Embeddings keys sample: {list(data['final_embeddings'].keys())[:5]}")
    print(f"  - Dimension: {data['embedding_dim']}")
    
    # Find Vortic watches
    vortic_watch_ids = []
    for watch_id, watch_data in data['watch_data'].items():
        if watch_data.get('brand', '').lower() == 'vortic watch company':
            vortic_watch_ids.append(watch_id)
    
    print(f"🔍 Found {len(vortic_watch_ids)} Vortic Watch Company watches to remove")
    
    if len(vortic_watch_ids) == 0:
        print("✅ No Vortic watches found to remove")
        return
    
    # Create backup
    print(f"💾 Creating backup at {backup_path}")
    with open(backup_path, 'wb') as f:
        pickle.dump(data, f)
    
    # Remove Vortic watches from watch_data and final_embeddings
    print("🗑️ Removing Vortic watches from watch_data and final_embeddings...")
    for watch_id in vortic_watch_ids:
        data['watch_data'].pop(watch_id, None)
        data['final_embeddings'].pop(watch_id, None)
    
    # Rebuild mappings
    print("🔄 Rebuilding watch ID mappings...")
    new_watch_id_to_idx = {}
    new_idx_to_watch_id = {}
    new_available_watches = set()
    
    for new_idx, watch_id in enumerate(data['final_embeddings'].keys()):
        new_watch_id_to_idx[watch_id] = new_idx
        new_idx_to_watch_id[new_idx] = watch_id
        new_available_watches.add(watch_id)
    
    data['watch_id_to_idx'] = new_watch_id_to_idx
    data['idx_to_watch_id'] = new_idx_to_watch_id
    data['available_watches'] = new_available_watches
    
    # Update embedding dimension (unchanged)
    print(f"📊 New dataset stats:")
    print(f"  - Total watches: {len(data['watch_data'])}")
    print(f"  - Embeddings keys sample: {list(data['final_embeddings'].keys())[:5]}")
    print(f"  - Dimension: {data['embedding_dim']}")
    print(f"  - Watches removed: {len(vortic_watch_ids)}")
    
    # Save filtered embeddings
    print(f"💾 Saving filtered embeddings to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print("✅ Successfully removed Vortic Watch Company watches!")
    print(f"📁 Backup saved at: {backup_path}")

if __name__ == "__main__":
    remove_vortic_watches() 