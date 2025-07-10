#!/usr/bin/env python3
"""
Real Watch Data Explorer

Focus on the watch_text_metadata.pkl file with 500 real watches.
"""

import os
import pickle
import numpy as np
from collections import defaultdict, Counter

def explore_real_watch_data():
    """Explore the real watch metadata with 500 watches."""
    
    print("=" * 80)
    print("🔍 REAL WATCH METADATA ANALYSIS (500 WATCHES)")
    print("=" * 80)
    
    # Load the real metadata
    metadata_path = "./data/watch_text_metadata.pkl"
    
    if not os.path.exists(metadata_path):
        print(f"❌ File not found: {metadata_path}")
        return
    
    file_size = os.path.getsize(metadata_path) / (1024 * 1024)
    print(f"📊 File size: {file_size:.1f} MB")
    
    with open(metadata_path, 'rb') as f:
        watch_list = pickle.load(f)
    
    print(f"✅ Loaded {len(watch_list)} watches")
    
    if not watch_list:
        print("❌ No watch data found!")
        return
    
    # Analyze structure
    sample_watch = watch_list[0]
    print(f"\n🏗️ WATCH RECORD STRUCTURE:")
    print(f"   Available fields: {list(sample_watch.keys())}")
    
    # Analyze all fields
    print(f"\n📋 FIELD ANALYSIS:")
    all_fields = set()
    field_counts = defaultdict(int)
    field_types = defaultdict(set)
    field_samples = defaultdict(list)
    
    for watch in watch_list[:100]:  # Sample first 100 for performance
        for field, value in watch.items():
            all_fields.add(field)
            field_counts[field] += 1
            field_types[field].add(type(value).__name__)
            
            # Collect samples
            if len(field_samples[field]) < 5:
                field_samples[field].append(value)
    
    for field in sorted(all_fields):
        coverage = (field_counts[field] / min(len(watch_list), 100)) * 100
        types = ", ".join(sorted(field_types[field]))
        print(f"   • {field}: {coverage:.0f}% coverage, types: {types}")
        
        # Show samples
        samples = field_samples[field][:3]
        if samples:
            sample_str = []
            for s in samples:
                if isinstance(s, str):
                    if len(s) > 50:
                        sample_str.append(f"'{s[:47]}...'")
                    else:
                        sample_str.append(f"'{s}'")
                else:
                    sample_str.append(str(s))
            print(f"     Samples: {', '.join(sample_str)}")
        print()
    
    # Analyze key fields in detail
    print(f"\n🎯 DETAILED FIELD ANALYSIS:")
    
    # Brands
    brands = [w['brand'] for w in watch_list if 'brand' in w]
    brand_counts = Counter(brands)
    print(f"\n   🏷️ BRANDS ({len(brand_counts)} unique):")
    for brand, count in brand_counts.most_common(15):
        percentage = (count / len(brands)) * 100
        print(f"     • {brand}: {count} ({percentage:.1f}%)")
    
    # Models
    models = [w['model'] for w in watch_list if 'model' in w]
    print(f"\n   ⌚ MODELS: {len(set(models))} unique models")
    
    # Descriptions
    descriptions = [w['description'] for w in watch_list if 'description' in w and w['description']]
    if descriptions:
        avg_desc_length = sum(len(d) for d in descriptions) / len(descriptions)
        print(f"\n   📝 DESCRIPTIONS: {len(descriptions)} watches have descriptions")
        print(f"     • Average length: {avg_desc_length:.0f} characters")
        print(f"     • Sample: '{descriptions[0][:200]}...'")
    
    # AI Descriptions
    ai_descriptions = [w['ai_description'] for w in watch_list if 'ai_description' in w and w['ai_description']]
    if ai_descriptions:
        avg_ai_desc_length = sum(len(d) for d in ai_descriptions) / len(ai_descriptions)
        print(f"\n   🤖 AI DESCRIPTIONS: {len(ai_descriptions)} watches have AI descriptions")
        print(f"     • Average length: {avg_ai_desc_length:.0f} characters")
        print(f"     • Sample: '{ai_descriptions[0][:200]}...'")
    
    # Specs
    specs_with_data = [w for w in watch_list if 'specs' in w and w['specs']]
    if specs_with_data:
        print(f"\n   🔧 SPECS: {len(specs_with_data)} watches have specs")
        sample_specs = specs_with_data[0]['specs']
        if isinstance(sample_specs, dict):
            print(f"     • Spec fields: {list(sample_specs.keys())}")
        else:
            print(f"     • Sample specs: '{str(sample_specs)[:200]}...'")
    
    # Price analysis
    prices = [w['price'] for w in watch_list if 'price' in w and w['price'] is not None]
    if prices:
        numeric_prices = []
        for p in prices:
            if isinstance(p, (int, float)):
                numeric_prices.append(p)
            elif isinstance(p, str):
                # Try to extract number from string
                import re
                numbers = re.findall(r'[\d,]+', p.replace(',', ''))
                if numbers:
                    try:
                        numeric_prices.append(int(numbers[0]))
                    except:
                        pass
        
        if numeric_prices:
            print(f"\n   💰 PRICES: {len(numeric_prices)} numeric prices")
            print(f"     • Min: ${min(numeric_prices):,}")
            print(f"     • Max: ${max(numeric_prices):,}")
            print(f"     • Average: ${sum(numeric_prices)/len(numeric_prices):,.0f}")
    
    # Sample complete records
    print(f"\n📝 SAMPLE COMPLETE WATCH RECORDS:")
    for i, watch in enumerate(watch_list[:3], 1):
        print(f"\n   🕰️ WATCH {i}:")
        for field, value in watch.items():
            if isinstance(value, str) and len(value) > 150:
                value = value[:147] + "..."
            elif isinstance(value, dict):
                value = f"Dict with keys: {list(value.keys())}"
            elif isinstance(value, np.ndarray):
                value = f"Array shape: {value.shape}"
            print(f"     • {field}: {value}")
    
    print(f"\n" + "=" * 80)
    print(f"🎉 REAL METADATA EXPLORATION COMPLETE!")
    print(f"   Found {len(watch_list)} watches with rich metadata!")
    print(f"=" * 80)

def main():
    """Main exploration function."""
    explore_real_watch_data()

if __name__ == "__main__":
    main() 