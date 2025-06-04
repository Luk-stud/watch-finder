#!/usr/bin/env python3
"""
Remove Vortic Watch Company entries from the dataset
"""

import pickle
import numpy as np
import shutil
from datetime import datetime

def backup_files():
    """Create backups of original files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("Creating backups...")
    shutil.copy('watch_text_metadata.pkl', f'watch_text_metadata_backup_{timestamp}.pkl')
    shutil.copy('watch_text_embeddings.pkl', f'watch_text_embeddings_backup_{timestamp}.pkl')
    print(f"✅ Backups created with timestamp {timestamp}")

def remove_vortic_watches():
    """Remove all Vortic watches from metadata and embeddings"""
    
    # Load current data
    print("Loading current data...")
    with open('watch_text_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    with open('watch_text_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    
    original_count = len(metadata)
    print(f"Original dataset: {original_count} watches")
    print(f"Original embeddings shape: {embeddings.shape}")
    
    # Find non-Vortic watches
    keep_indices = []
    vortic_count = 0
    
    for i, watch in enumerate(metadata):
        brand = watch.get('brand', '').lower()
        if 'vortic' in brand:
            vortic_count += 1
            print(f"Removing: {watch.get('brand')} - {watch.get('model', 'N/A')}")
        else:
            keep_indices.append(i)
    
    print(f"\n📊 Summary:")
    print(f"   Total watches: {original_count}")
    print(f"   Vortic watches: {vortic_count}")
    print(f"   Keeping: {len(keep_indices)}")
    print(f"   Removal percentage: {vortic_count/original_count*100:.1f}%")
    
    # Create filtered datasets
    print("\nFiltering datasets...")
    filtered_metadata = [metadata[i] for i in keep_indices]
    filtered_embeddings = embeddings[keep_indices]
    
    # Update indices in metadata to be sequential
    print("Updating indices...")
    for new_idx, watch in enumerate(filtered_metadata):
        watch['index'] = new_idx
    
    print(f"✅ New dataset: {len(filtered_metadata)} watches")
    print(f"✅ New embeddings shape: {filtered_embeddings.shape}")
    
    # Save filtered data
    print("\nSaving filtered data...")
    with open('watch_text_metadata.pkl', 'wb') as f:
        pickle.dump(filtered_metadata, f)
    
    with open('watch_text_embeddings.pkl', 'wb') as f:
        pickle.dump(filtered_embeddings, f)
    
    print("✅ Successfully removed Vortic watches!")
    print(f"   Dataset reduced from {original_count} to {len(filtered_metadata)} watches")
    
    return len(filtered_metadata)

if __name__ == "__main__":
    backup_files()
    final_count = remove_vortic_watches()
    print(f"\n🎉 Dataset cleanup complete! Now have {final_count} high-quality watches.") 