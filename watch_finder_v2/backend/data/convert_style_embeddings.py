#!/usr/bin/env python3
"""
Convert Style-Focused Embeddings for LinUCB Integration
=====================================================

This script converts our brand-balanced style-focused embeddings to the format
expected by the LinUCB recommendation system.
"""

import pickle
import numpy as np
from pathlib import Path

def convert_style_embeddings():
    """Convert style-focused embeddings to LinUCB format."""
    
    # Paths
    style_output_dir = "../../../text_embedding_generation/brand_balanced_500_output"
    style_metadata_path = f"{style_output_dir}/watch_style_metadata.pkl"
    style_embeddings_path = f"{style_output_dir}/watch_style_embeddings.pkl"
    
    print("🔄 Converting style-focused embeddings for LinUCB...")
    
    try:
        # Load style-focused data
        with open(style_metadata_path, 'rb') as f:
            style_watches = pickle.load(f)
        
        with open(style_embeddings_path, 'rb') as f:
            style_embeddings = pickle.load(f)
        
        print(f"✅ Loaded {len(style_watches)} style-focused watches")
        print(f"✅ Loaded embeddings shape: {style_embeddings.shape}")
        
        # Convert to LinUCB format
        linucb_metadata = []
        linucb_embeddings = []
        
        for i, watch in enumerate(style_watches):
            # Extract price properly
            price = 0
            if 'specs' in watch and watch['specs']:
                if 'price_usd' in watch['specs']:
                    try:
                        price = float(watch['specs']['price_usd'])
                    except:
                        price = 0
            
            # Convert to LinUCB expected format
            linucb_watch = {
                'index': i,
                'brand': watch['brand'],
                'model': watch['model'], 
                'price': price,
                'image_url': watch.get('main_image_url', ''),
                'description': watch.get('visual_style_description', ''),
                'source': 'style_focused_visual_analysis',
                'specs': watch.get('specs', {}),
                'ai_description': watch.get('visual_style_description', ''),
                'text_embedding': style_embeddings[i],  # Include embedding in metadata
                'style_embedding': style_embeddings[i]   # Keep reference to style embedding
            }
            
            linucb_metadata.append(linucb_watch)
            linucb_embeddings.append(style_embeddings[i])
        
        # Convert embeddings to numpy array
        linucb_embeddings_array = np.array(linucb_embeddings)
        
        print(f"✅ Converted to LinUCB format:")
        print(f"   - Metadata: {len(linucb_metadata)} watches")
        print(f"   - Embeddings: {linucb_embeddings_array.shape}")
        
        # Save in LinUCB format
        print("💾 Saving new style-focused embeddings...")
        
        with open('watch_text_metadata.pkl', 'wb') as f:
            pickle.dump(linucb_metadata, f)
        
        with open('watch_text_embeddings.pkl', 'wb') as f:
            pickle.dump(linucb_embeddings_array, f)
        
        print("✅ Style-focused embeddings saved!")
        print("🎯 LinUCB system will now use visual design-based recommendations!")
        
        # Print sample comparison
        print("\n📊 SAMPLE DESCRIPTIONS:")
        for i in range(min(3, len(style_watches))):
            watch = style_watches[i]
            print(f"\n{i+1}. {watch['brand']} {watch['model']}")
            print(f"   Style Description: {watch['visual_style_description'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting embeddings: {e}")
        return False

if __name__ == "__main__":
    convert_style_embeddings() 