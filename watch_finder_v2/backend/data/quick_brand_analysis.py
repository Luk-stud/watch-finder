#!/usr/bin/env python3
"""
Quick Brand Distance Summary
===========================
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def quick_analysis():
    # Load data
    with open('watch_text_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    with open('watch_text_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    df = pd.DataFrame(metadata)
    
    # Get major brands
    brand_counts = df['brand'].value_counts()
    major_brands = brand_counts[brand_counts >= 20].index.tolist()
    
    print(f"🎯 QUICK BRAND SEPARATION ANALYSIS")
    print(f"=" * 50)
    print(f"📊 Analyzing {len(major_brands)} major brands with 20+ watches:")
    for brand in major_brands:
        count = brand_counts[brand]
        print(f"   {brand}: {count} watches")
    
    # Compute brand centroids for major brands
    brand_centroids = {}
    for brand in major_brands:
        brand_indices = df[df['brand'] == brand].index.values
        brand_embeddings = embeddings[brand_indices]
        centroid = np.mean(brand_embeddings, axis=0)
        brand_centroids[brand] = centroid
    
    # Calculate all pairwise distances
    brands = list(brand_centroids.keys())
    distances = []
    
    for i in range(len(brands)):
        for j in range(i + 1, len(brands)):
            brand_a, brand_b = brands[i], brands[j]
            cos_sim = cosine_similarity([brand_centroids[brand_a]], [brand_centroids[brand_b]])[0, 0]
            distance = 1 - cos_sim
            distances.append(distance)
    
    distances = np.array(distances)
    
    print(f"\n📏 DISTANCE STATISTICS:")
    print(f"   Average inter-brand distance: {np.mean(distances):.4f}")
    print(f"   Standard deviation: {np.std(distances):.4f}")
    print(f"   Minimum distance: {np.min(distances):.4f}")
    print(f"   Maximum distance: {np.max(distances):.4f}")
    print(f"   Median distance: {np.median(distances):.4f}")
    
    # Distance interpretation
    avg_dist = np.mean(distances)
    if avg_dist > 0.4:
        interpretation = "EXTREMELY far apart"
    elif avg_dist > 0.3:
        interpretation = "VERY far apart"
    elif avg_dist > 0.2:
        interpretation = "Moderately separated"
    elif avg_dist > 0.1:
        interpretation = "Somewhat separated"
    else:
        interpretation = "Close together"
    
    print(f"\n🎯 INTERPRETATION: Brands are {interpretation}")
    print(f"   (Distance scale: 0=identical, 1=completely opposite)")
    
    # Show most/least similar
    min_idx = np.argmin(distances)
    max_idx = np.argmax(distances)
    
    # Find which brands these correspond to
    pair_idx = 0
    for i in range(len(brands)):
        for j in range(i + 1, len(brands)):
            if pair_idx == min_idx:
                most_similar = (brands[i], brands[j], distances[min_idx])
            if pair_idx == max_idx:
                most_different = (brands[i], brands[j], distances[max_idx])
            pair_idx += 1
    
    print(f"\n🤝 Most similar brands: {most_similar[0]} ↔ {most_similar[1]} (distance: {most_similar[2]:.4f})")
    print(f"⚡ Most different brands: {most_different[0]} ↔ {most_different[1]} (distance: {most_different[2]:.4f})")

if __name__ == "__main__":
    quick_analysis() 