#!/usr/bin/env python3
"""
Use the largest dataset with 3,320 watches for better duplicate prevention
"""

import os
import pickle
import numpy as np
import shutil

def main():
    """Copy the largest available embeddings to backend data directory."""
    
    # Source files (largest available from embedding generation v2)
    source_embeddings = "embedding_generation_v2/output/embeddings_intermediate_3320.npy"
    source_metadata = "embedding_generation_v2/output/enhanced_watches_intermediate_3320.pkl"
    
    # Destination files
    dest_embeddings = "backend/data/watch_embeddings.pkl"
    dest_metadata = "backend/data/watch_metadata.pkl"
    
    print("🔄 Updating to MASSIVE dataset with 3,320 watches...")
    
    try:
        # Check source files exist
        if not os.path.exists(source_embeddings):
            raise FileNotFoundError(f"Source embeddings not found: {source_embeddings}")
        if not os.path.exists(source_metadata):
            raise FileNotFoundError(f"Source metadata not found: {source_metadata}")
        
        # Load and convert embeddings
        print(f"📊 Loading embeddings from: {source_embeddings}")
        embeddings = np.load(source_embeddings)
        
        print(f"📋 Loading metadata from: {source_metadata}")
        with open(source_metadata, 'rb') as f:
            watch_data = pickle.load(f)
        
        print(f"✅ Loaded {len(watch_data)} watches with {embeddings.shape[1]}D embeddings")
        
        # Backup current files before overwriting
        if os.path.exists(dest_embeddings):
            backup_embeddings = "backend/data/watch_embeddings_500_backup.pkl"
            shutil.copy2(dest_embeddings, backup_embeddings)
            print(f"📦 Backed up current embeddings to: {backup_embeddings}")
        
        if os.path.exists(dest_metadata):
            backup_metadata = "backend/data/watch_metadata_500_backup.pkl"
            shutil.copy2(dest_metadata, backup_metadata)
            print(f"📦 Backed up current metadata to: {backup_metadata}")
        
        # Save new embeddings
        print(f"💾 Saving embeddings to: {dest_embeddings}")
        with open(dest_embeddings, 'wb') as f:
            pickle.dump(embeddings, f)
        
        # Save new metadata
        print(f"💾 Saving metadata to: {dest_metadata}")
        with open(dest_metadata, 'wb') as f:
            pickle.dump(watch_data, f)
        
        print(f"\n🎉 Successfully updated to MASSIVE dataset!")
        print(f"📊 Dataset size: {len(watch_data)} watches (was 500 - 6.6x larger!)")
        print(f"📐 Embedding dimension: {embeddings.shape[1]}")
        
        # Show brand and style diversity
        brands = set(watch.get('brand', 'Unknown') for watch in watch_data)
        styles = set()
        for watch in watch_data:
            # Use same style classification as beam search
            specs = watch.get('specs', {})
            watch_type = specs.get('watch_type', '').lower()
            if watch_type:
                styles.add(watch_type)
            else:
                description = watch.get('ai_description', '').lower()
                if any(term in description for term in ['dive', 'water', 'ocean']):
                    styles.add('diver')
                elif any(term in description for term in ['dress', 'formal', 'elegant']):
                    styles.add('dress')
                elif any(term in description for term in ['sport', 'athletic', 'performance']):
                    styles.add('sport')
                elif any(term in description for term in ['pilot', 'aviation', 'field']):
                    styles.add('field')
                else:
                    styles.add('casual')
        
        print(f"🏭 Unique brands: {len(brands)}")
        print(f"🎨 Unique styles: {len(styles)}")
        
        # Show file sizes
        emb_size = os.path.getsize(dest_embeddings) / (1024 * 1024)
        meta_size = os.path.getsize(dest_metadata) / (1024 * 1024)
        print(f"📁 Embeddings file size: {emb_size:.1f} MB")
        print(f"📁 Metadata file size: {meta_size:.1f} MB")
        
        print(f"\n🔄 Backend restart required to use new massive dataset")
        print(f"🚀 This will dramatically improve duplicate prevention!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    if not success:
        import sys
        sys.exit(1) 