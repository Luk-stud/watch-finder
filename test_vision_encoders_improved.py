#!/usr/bin/env python3
"""
Improved Vision Encoder Comparison

Better comparison that excludes identical watches and shows meaningful differences.
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

def find_nearest_neighbors_diverse(embeddings: Dict, watch_data: Dict, query_watch_id: str, n_neighbors: int = 5) -> List[Tuple]:
    """Find nearest neighbors excluding identical watches."""
    if query_watch_id not in embeddings:
        print(f"❌ Watch {query_watch_id} not found in embeddings")
        return []
    
    query_embedding = embeddings[query_watch_id]
    query_data = watch_data[query_watch_id]
    query_brand = query_data.get('brand', '').lower()
    query_model = query_data.get('model', '').lower()
    
    # Calculate similarities
    similarities = []
    for watch_id, embedding in embeddings.items():
        if watch_id == query_watch_id:
            continue
            
        # Skip identical watches (same brand+model)
        neighbor_data = watch_data[watch_id]
        neighbor_brand = neighbor_data.get('brand', '').lower()
        neighbor_model = neighbor_data.get('model', '').lower()
        
        if query_brand == neighbor_brand and query_model == neighbor_model:
            continue
        
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

def analyze_embedding_quality(embeddings: Dict, watch_data: Dict):
    """Analyze the quality of embeddings by type separation."""
    print("\n🔍 Analyzing embedding quality...")
    
    # Group watches by type
    type_groups = {}
    for watch_id, data in watch_data.items():
        if watch_id not in embeddings:
            continue
            
        specs = data.get('specs', {})
        watch_type = specs.get('watch_type', 'Unknown')
        if watch_type not in type_groups:
            type_groups[watch_type] = []
        type_groups[watch_type].append(watch_id)
    
    # Calculate intra-type and inter-type similarities
    print("📊 Type separation analysis:")
    for watch_type, watch_ids in type_groups.items():
        if len(watch_ids) < 2:
            continue
            
        # Intra-type similarity (within same type)
        intra_similarities = []
        for i, watch1 in enumerate(watch_ids):
            for j, watch2 in enumerate(watch_ids[i+1:], i+1):
                # Skip identical watches
                data1 = watch_data[watch1]
                data2 = watch_data[watch2]
                if (data1.get('brand', '').lower() == data2.get('brand', '').lower() and 
                    data1.get('model', '').lower() == data2.get('model', '').lower()):
                    continue
                    
                similarity = cosine_similarity([embeddings[watch1]], [embeddings[watch2]])[0][0]
                intra_similarities.append(similarity)
        
        # Inter-type similarity (with other types)
        inter_similarities = []
        other_watches = [w for w in watch_data.keys() if w in embeddings and w not in watch_ids]
        
        for watch1 in watch_ids[:5]:  # Sample first 5
            for watch2 in other_watches[:20]:  # Sample first 20 from other types
                similarity = cosine_similarity([embeddings[watch1]], [embeddings[watch2]])[0][0]
                inter_similarities.append(similarity)
        
        if intra_similarities and inter_similarities:
            mean_intra = np.mean(intra_similarities)
            mean_inter = np.mean(inter_similarities)
            separation = mean_intra - mean_inter
            
            print(f"  {watch_type}:")
            print(f"    Intra-type similarity: {mean_intra:.3f} (n={len(intra_similarities)})")
            print(f"    Inter-type similarity: {mean_inter:.3f} (n={len(inter_similarities)})")
            print(f"    Separation score: {separation:.3f}")

def test_encoder_comparison():
    """Test different encoders on the same samples."""
    print("🔍 Improved Vision Encoder Comparison Test")
    print("="*60)
    
    # Load DINO embeddings
    try:
        dino_data = load_embeddings('data/precomputed_embeddings.pkl')
        dino_embeddings = dino_data['final_embeddings']
        watch_data = dino_data['watch_data']
        print(f"✅ Loaded {len(dino_embeddings)} DINO embeddings")
    except Exception as e:
        print(f"❌ Failed to load DINO embeddings: {e}")
        return
    
    # Analyze embedding quality
    analyze_embedding_quality(dino_embeddings, watch_data)
    
    # Select diverse test cases (one per brand to avoid duplicates)
    test_cases = []
    seen_brands = set()
    
    # Find diverse watches
    for watch_id, data in watch_data.items():
        if watch_id not in dino_embeddings:
            continue
            
        brand = data.get('brand', '').lower()
        if brand not in seen_brands:
            specs = data.get('specs', {})
            watch_type = specs.get('watch_type', 'Unknown')
            
            # Only include watches with clear types
            if watch_type in ['Dress', 'Diver', 'Chronograph', 'Pilot', 'Field']:
                test_cases.append(watch_id)
                seen_brands.add(brand)
                
                if len(test_cases) >= 8:
                    break
    
    print(f"\n🎯 Testing {len(test_cases)} diverse watch samples:")
    for watch_id in test_cases:
        print_watch_info(watch_data, watch_id)
    
    # Test DINO encoder
    print(f"\n" + "="*60)
    print(f"🔍 DINO ENCODER RESULTS (excluding identical watches)")
    print("="*60)
    
    for test_watch_id in test_cases:
        print(f"\n📱 Query: {watch_data[test_watch_id].get('brand', 'Unknown')} {watch_data[test_watch_id].get('model', 'Unknown')}")
        print(f"   Type: {watch_data[test_watch_id].get('specs', {}).get('watch_type', 'Unknown')}")
        
        # Find nearest neighbors (excluding identical watches)
        neighbors = find_nearest_neighbors_diverse(dino_embeddings, watch_data, test_watch_id, n_neighbors=5)
        
        print(f"   🔗 Nearest neighbors (different watches):")
        for i, (neighbor_id, similarity) in enumerate(neighbors, 1):
            neighbor_data = watch_data[neighbor_id]
            neighbor_brand = neighbor_data.get('brand', 'Unknown')
            neighbor_model = neighbor_data.get('model', 'Unknown')
            neighbor_type = neighbor_data.get('specs', {}).get('watch_type', 'Unknown')
            
            print(f"     {i}. {neighbor_brand} {neighbor_model} ({neighbor_type}) - similarity: {similarity:.3f}")
    
    # Test type-based clustering
    print(f"\n" + "="*60)
    print("🎯 TYPE-BASED CLUSTERING ANALYSIS")
    print("="*60)
    
    # Test dress watch clustering
    dress_watches = []
    for watch_id, data in watch_data.items():
        if watch_id in dino_embeddings:
            specs = data.get('specs', {})
            if specs.get('watch_type', '').lower() == 'dress':
                dress_watches.append(watch_id)
                if len(dress_watches) >= 5:
                    break
    
    if dress_watches:
        print(f"\n👔 Dress watch clustering (testing {len(dress_watches)} dress watches):")
        for dress_watch in dress_watches[:3]:
            print(f"\n  Query: {watch_data[dress_watch].get('brand', 'Unknown')} {watch_data[dress_watch].get('model', 'Unknown')}")
            neighbors = find_nearest_neighbors_diverse(dino_embeddings, watch_data, dress_watch, n_neighbors=3)
            
            dress_neighbors = 0
            for neighbor_id, similarity in neighbors:
                neighbor_type = watch_data[neighbor_id].get('specs', {}).get('watch_type', 'Unknown')
                if neighbor_type.lower() == 'dress':
                    dress_neighbors += 1
                print(f"    {watch_data[neighbor_id].get('brand', 'Unknown')} {watch_data[neighbor_id].get('model', 'Unknown')} ({neighbor_type}) - {similarity:.3f}")
            
            print(f"    Dress watch matches: {dress_neighbors}/{len(neighbors)}")
    
    # Test dive watch clustering
    dive_watches = []
    for watch_id, data in watch_data.items():
        if watch_id in dino_embeddings:
            specs = data.get('specs', {})
            if specs.get('watch_type', '').lower() == 'diver':
                dive_watches.append(watch_id)
                if len(dive_watches) >= 5:
                    break
    
    if dive_watches:
        print(f"\n🏊 Dive watch clustering (testing {len(dive_watches)} dive watches):")
        for dive_watch in dive_watches[:3]:
            print(f"\n  Query: {watch_data[dive_watch].get('brand', 'Unknown')} {watch_data[dive_watch].get('model', 'Unknown')}")
            neighbors = find_nearest_neighbors_diverse(dino_embeddings, watch_data, dive_watch, n_neighbors=3)
            
            dive_neighbors = 0
            for neighbor_id, similarity in neighbors:
                neighbor_type = watch_data[neighbor_id].get('specs', {}).get('watch_type', 'Unknown')
                if neighbor_type.lower() == 'diver':
                    dive_neighbors += 1
                print(f"    {watch_data[neighbor_id].get('brand', 'Unknown')} {watch_data[neighbor_id].get('model', 'Unknown')} ({neighbor_type}) - {similarity:.3f}")
            
            print(f"    Dive watch matches: {dive_neighbors}/{len(neighbors)}")

if __name__ == "__main__":
    test_encoder_comparison() 