#!/usr/bin/env python3
"""
Test Different Vision Encoders

Simple script to test different vision encoders and show nearest neighbors
for specific watch samples.
"""

import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import torch
from PIL import Image
import requests
from io import BytesIO

def load_embeddings(filepath: str) -> Dict:
    """Load embeddings from file."""
    print(f"📊 Loading embeddings from {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def find_nearest_neighbors(embeddings: Dict, query_watch_id: str, n_neighbors: int = 5) -> List[Tuple]:
    """Find nearest neighbors for a given watch."""
    if query_watch_id not in embeddings:
        print(f"❌ Watch {query_watch_id} not found in embeddings")
        return []
    
    query_embedding = embeddings[query_watch_id]
    
    # Calculate similarities
    similarities = []
    for watch_id, embedding in embeddings.items():
        if watch_id != query_watch_id:
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append((watch_id, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:n_neighbors]

def print_watch_info(watch_data: Dict, watch_id: str):
    """Print watch information."""
    data = watch_data.get(watch_id, {})
    brand = data.get('brand', 'Unknown')
    model = data.get('model', 'Unknown')
    specs = data.get('specs', {})
    watch_type = specs.get('watch_type', 'Unknown')
    price = data.get('price', 'Unknown')
    
    print(f"  📱 {brand} {model}")
    print(f"     Type: {watch_type}")
    print(f"     Price: {price}")

def test_encoder_comparison():
    """Test different encoders on the same samples."""
    print("🔍 Vision Encoder Comparison Test")
    print("="*50)
    
    # Load DINO embeddings
    try:
        dino_data = load_embeddings('data/precomputed_embeddings.pkl')
        dino_embeddings = dino_data['final_embeddings']
        watch_data = dino_data['watch_data']
        print(f"✅ Loaded {len(dino_embeddings)} DINO embeddings")
    except Exception as e:
        print(f"❌ Failed to load DINO embeddings: {e}")
        return
    
    # Load CLIP embeddings if available
    clip_embeddings = {}
    try:
        clip_data = load_embeddings('clip_image_embeddings/watch_image_embeddings_final_scrape.pkl')
        # Convert to same format
        for watch_id, embedding in clip_data.items():
            if watch_id in watch_data:
                clip_embeddings[watch_id] = embedding
        print(f"✅ Loaded {len(clip_embeddings)} CLIP embeddings")
    except Exception as e:
        print(f"⚠️ CLIP embeddings not available: {e}")
    
    # Select some interesting test cases
    test_cases = []
    
    # Find some dress watches
    dress_watches = []
    for watch_id, data in watch_data.items():
        specs = data.get('specs', {})
        if specs.get('watch_type', '').lower() == 'dress':
            dress_watches.append(watch_id)
            if len(dress_watches) >= 3:
                break
    
    # Find some dive watches
    dive_watches = []
    for watch_id, data in watch_data.items():
        specs = data.get('specs', {})
        if specs.get('watch_type', '').lower() == 'diver':
            dive_watches.append(watch_id)
            if len(dive_watches) >= 3:
                break
    
    # Find some chronograph watches
    chrono_watches = []
    for watch_id, data in watch_data.items():
        specs = data.get('specs', {})
        if 'chronograph' in specs.get('watch_type', '').lower():
            chrono_watches.append(watch_id)
            if len(chrono_watches) >= 3:
                break
    
    test_cases.extend(dress_watches[:2])
    test_cases.extend(dive_watches[:2])
    test_cases.extend(chrono_watches[:2])
    
    print(f"\n🎯 Testing {len(test_cases)} watch samples:")
    for watch_id in test_cases:
        print_watch_info(watch_data, watch_id)
    
    # Test each encoder
    encoders = {'DINO': dino_embeddings}
    if clip_embeddings:
        encoders['CLIP'] = clip_embeddings
    
    for encoder_name, embeddings in encoders.items():
        print(f"\n" + "="*60)
        print(f"🔍 {encoder_name.upper()} ENCODER RESULTS")
        print("="*60)
        
        for test_watch_id in test_cases:
            if test_watch_id not in embeddings:
                print(f"⚠️ {test_watch_id} not found in {encoder_name} embeddings")
                continue
            
            print(f"\n📱 Query: {watch_data[test_watch_id].get('brand', 'Unknown')} {watch_data[test_watch_id].get('model', 'Unknown')}")
            print(f"   Type: {watch_data[test_watch_id].get('specs', {}).get('watch_type', 'Unknown')}")
            
            # Find nearest neighbors
            neighbors = find_nearest_neighbors(embeddings, test_watch_id, n_neighbors=5)
            
            print(f"   🔗 Nearest neighbors:")
            for i, (neighbor_id, similarity) in enumerate(neighbors, 1):
                neighbor_data = watch_data[neighbor_id]
                neighbor_brand = neighbor_data.get('brand', 'Unknown')
                neighbor_model = neighbor_data.get('model', 'Unknown')
                neighbor_type = neighbor_data.get('specs', {}).get('watch_type', 'Unknown')
                
                print(f"     {i}. {neighbor_brand} {neighbor_model} ({neighbor_type}) - similarity: {similarity:.3f}")
    
    # Cross-encoder comparison
    if len(encoders) > 1:
        print(f"\n" + "="*60)
        print("🔄 CROSS-ENCODER COMPARISON")
        print("="*60)
        
        # Find common watches
        common_watches = set(dino_embeddings.keys()) & set(clip_embeddings.keys())
        print(f"📊 {len(common_watches)} watches have both DINO and CLIP embeddings")
        
        # Compare similarities for a few test cases
        for test_watch_id in test_cases[:3]:
            if test_watch_id not in common_watches:
                continue
            
            print(f"\n📱 Comparing similarities for: {watch_data[test_watch_id].get('brand', 'Unknown')} {watch_data[test_watch_id].get('model', 'Unknown')}")
            
            dino_neighbors = find_nearest_neighbors(dino_embeddings, test_watch_id, n_neighbors=3)
            clip_neighbors = find_nearest_neighbors(clip_embeddings, test_watch_id, n_neighbors=3)
            
            print(f"   DINO top matches:")
            for neighbor_id, similarity in dino_neighbors:
                neighbor_data = watch_data[neighbor_id]
                print(f"     {neighbor_data.get('brand', 'Unknown')} {neighbor_data.get('model', 'Unknown')} ({similarity:.3f})")
            
            print(f"   CLIP top matches:")
            for neighbor_id, similarity in clip_neighbors:
                neighbor_data = watch_data[neighbor_id]
                print(f"     {neighbor_data.get('brand', 'Unknown')} {neighbor_data.get('model', 'Unknown')} ({similarity:.3f})")

def generate_clip_embeddings_for_samples():
    """Generate CLIP embeddings for a few sample watches."""
    print("🎨 Generating CLIP embeddings for samples...")
    
    try:
        import clip
        from torchvision import transforms
        
        # Load DINO data to get watch info
        with open('data/precomputed_embeddings.pkl', 'rb') as f:
            dino_data = pickle.load(f)
        
        watch_data = dino_data['watch_data']
        
        # Select a few diverse samples
        sample_watches = []
        for watch_id, data in watch_data.items():
            specs = data.get('specs', {})
            watch_type = specs.get('watch_type', 'Unknown')
            if watch_type in ['Dress', 'Diver', 'Chronograph']:
                sample_watches.append(watch_id)
                if len(sample_watches) >= 10:
                    break
        
        print(f"📊 Generating CLIP embeddings for {len(sample_watches)} samples...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        clip_embeddings = {}
        
        for i, watch_id in enumerate(sample_watches):
            try:
                data = watch_data[watch_id]
                image_url = data.get('image_url')
                
                if not image_url:
                    continue
                
                # Download and process image
                response = requests.get(image_url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                
                # Generate embedding
                with torch.no_grad():
                    image_features = model.encode_image(image_tensor)
                    embedding = image_features.cpu().numpy().flatten()
                
                clip_embeddings[watch_id] = embedding
                print(f"  ✅ {data.get('brand', 'Unknown')} {data.get('model', 'Unknown')}")
                
            except Exception as e:
                print(f"  ❌ Failed to process {watch_id}: {e}")
                continue
        
        # Save CLIP embeddings
        output_path = 'clip_sample_embeddings.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(clip_embeddings, f)
        
        print(f"✅ Saved {len(clip_embeddings)} CLIP embeddings to {output_path}")
        
    except ImportError:
        print("❌ CLIP not available. Install with: pip install ftfy regex clip")

if __name__ == "__main__":
    # First try to generate CLIP embeddings if they don't exist
    if not os.path.exists('clip_image_embeddings/watch_image_embeddings_final_scrape.pkl'):
        print("⚠️ CLIP embeddings not found. Generating sample embeddings...")
        generate_clip_embeddings_for_samples()
    
    # Run the comparison test
    test_encoder_comparison() 