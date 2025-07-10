#!/usr/bin/env python3
"""
Comprehensive Watch Metadata Explorer

Thoroughly analyze the precomputed_embeddings.pkl file to understand
what data is available for watch classification and recommendations.
"""

import os
import pickle
import numpy as np
from collections import defaultdict, Counter
import json

def load_and_explore_metadata():
    """Load and thoroughly explore the watch metadata."""
    
    print("=" * 80)
    print("🔍 COMPREHENSIVE WATCH METADATA ANALYSIS")
    print("=" * 80)
    
    # Load the data
    precomputed_path = os.path.join("data", 'precomputed_embeddings.pkl')
    
    if not os.path.exists(precomputed_path):
        print(f"❌ File not found: {precomputed_path}")
        return
    
    print(f"📂 Loading file: {precomputed_path}")
    file_size = os.path.getsize(precomputed_path) / (1024 * 1024)
    print(f"📊 File size: {file_size:.1f} MB")
    
    with open(precomputed_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Successfully loaded pickle file")
    
    # Analyze top-level structure
    print(f"\n🏗️ TOP-LEVEL STRUCTURE:")
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"   • {key}: dict with {len(value)} items")
        elif isinstance(value, (list, tuple)):
            print(f"   • {key}: {type(value).__name__} with {len(value)} items")
        elif isinstance(value, np.ndarray):
            print(f"   • {key}: numpy array shape {value.shape}")
        else:
            print(f"   • {key}: {type(value).__name__} = {value}")
    
    # Focus on watch_data
    watch_data = data.get('watch_data', {})
    embeddings = data.get('final_embeddings', {})
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   • Total watches in metadata: {len(watch_data)}")
    print(f"   • Total watches with embeddings: {len(embeddings)}")
    print(f"   • Embedding dimension: {data.get('embedding_dim', 'Unknown')}")
    
    if not watch_data:
        print("❌ No watch_data found!")
        return
    
    # Analyze watch data structure
    print(f"\n🔍 WATCH DATA FIELD ANALYSIS:")
    
    # Collect all fields across all watches
    all_fields = set()
    field_counts = defaultdict(int)
    field_types = defaultdict(set)
    field_samples = defaultdict(list)
    
    for watch_id, watch_info in watch_data.items():
        if isinstance(watch_info, dict):
            for field, value in watch_info.items():
                all_fields.add(field)
                field_counts[field] += 1
                field_types[field].add(type(value).__name__)
                
                # Collect samples (limit to avoid memory issues)
                if len(field_samples[field]) < 10:
                    field_samples[field].append(value)
    
    # Print field analysis
    print(f"\n📋 AVAILABLE FIELDS ({len(all_fields)} total):")
    for field in sorted(all_fields):
        coverage = (field_counts[field] / len(watch_data)) * 100
        types = ", ".join(sorted(field_types[field]))
        print(f"   • {field}: {coverage:.1f}% coverage, types: {types}")
        
        # Show sample values
        samples = field_samples[field][:5]  # Show first 5 samples
        if samples:
            sample_str = ", ".join([f"'{s}'" if isinstance(s, str) else str(s) for s in samples])
            if len(sample_str) > 100:
                sample_str = sample_str[:97] + "..."
            print(f"     Samples: {sample_str}")
        print()
    
    # Analyze specific important fields
    important_fields = ['brand', 'model', 'category', 'type', 'description', 'price', 'movement', 'case_material']
    
    print(f"\n🎯 DETAILED ANALYSIS OF KEY FIELDS:")
    for field in important_fields:
        if field in all_fields:
            print(f"\n   🔸 {field.upper()}:")
            values = []
            for watch_info in watch_data.values():
                if isinstance(watch_info, dict) and field in watch_info:
                    values.append(watch_info[field])
            
            if values:
                # Count unique values
                if isinstance(values[0], str):
                    value_counts = Counter(values)
                    print(f"     • Unique values: {len(value_counts)}")
                    print(f"     • Most common:")
                    for value, count in value_counts.most_common(10):
                        percentage = (count / len(values)) * 100
                        print(f"       - '{value}': {count} ({percentage:.1f}%)")
                else:
                    print(f"     • Sample values: {values[:10]}")
        else:
            print(f"\n   ❌ {field.upper()}: Not found in dataset")
    
    # Sample complete watch records
    print(f"\n📝 SAMPLE COMPLETE WATCH RECORDS:")
    sample_watches = list(watch_data.items())[:5]  # First 5 watches
    
    for i, (watch_id, watch_info) in enumerate(sample_watches, 1):
        print(f"\n   🕰️ WATCH {i} (ID: {watch_id}):")
        if isinstance(watch_info, dict):
            for field, value in sorted(watch_info.items()):
                if isinstance(value, str) and len(value) > 100:
                    value = value[:97] + "..."
                print(f"     • {field}: {value}")
        else:
            print(f"     Raw data: {watch_info}")
    
    # Analyze embeddings
    if embeddings:
        print(f"\n🧠 EMBEDDING ANALYSIS:")
        sample_embedding = next(iter(embeddings.values()))
        print(f"   • Embedding shape: {sample_embedding.shape}")
        print(f"   • Embedding type: {type(sample_embedding)}")
        print(f"   • Sample values: {sample_embedding[:10]}...")
        
        # Check if all embeddings have same shape
        shapes = [emb.shape for emb in embeddings.values()]
        unique_shapes = set(shapes)
        print(f"   • All embeddings same shape: {len(unique_shapes) == 1}")
        if len(unique_shapes) > 1:
            print(f"   • Shape variations: {unique_shapes}")
    
    # Try to identify the best fields for type classification
    print(f"\n🏷️ TYPE CLASSIFICATION RECOMMENDATIONS:")
    
    classification_fields = []
    
    # Check for direct category/type fields
    if 'category' in all_fields:
        classification_fields.append('category')
    if 'type' in all_fields:
        classification_fields.append('type')
    
    # Check for text fields that could be used for classification
    text_fields = []
    for field in ['description', 'model', 'name', 'title']:
        if field in all_fields:
            text_fields.append(field)
    
    print(f"   ✅ Direct classification fields: {classification_fields}")
    print(f"   ✅ Text fields for rule-based classification: {text_fields}")
    
    # Brand analysis for type inference
    if 'brand' in all_fields:
        brands = []
        for watch_info in watch_data.values():
            if isinstance(watch_info, dict) and 'brand' in watch_info:
                brands.append(watch_info['brand'])
        
        brand_counts = Counter(brands)
        print(f"\n   📊 BRAND DISTRIBUTION (Top 10):")
        for brand, count in brand_counts.most_common(10):
            percentage = (count / len(brands)) * 100
            print(f"     • {brand}: {count} watches ({percentage:.1f}%)")
    
    print(f"\n" + "=" * 80)
    print(f"🎉 METADATA EXPLORATION COMPLETE!")
    print(f"=" * 80)

def main():
    """Main exploration function."""
    load_and_explore_metadata()

if __name__ == "__main__":
    main() 