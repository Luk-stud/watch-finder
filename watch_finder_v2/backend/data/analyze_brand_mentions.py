#!/usr/bin/env python3
"""
Analyze Brand Mentions in Text Descriptions
==========================================

Analyzes how frequently brand names appear in the text descriptions
used to create embeddings, which could explain the strong brand separation.
"""

import pickle
import re
from collections import defaultdict, Counter
from typing import Dict, List

def analyze_brand_mentions():
    """Analyze brand name frequency in text descriptions."""
    
    # Load watch metadata
    try:
        with open('watch_text_metadata.pkl', 'rb') as f:
            metadata_list = pickle.load(f)
        print(f"✅ Loaded {len(metadata_list)} watches")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    brand_mention_stats = defaultdict(list)  # brand -> [mention_counts_per_watch]
    description_lengths = []
    total_watches = 0
    
    for watch in metadata_list:
        try:
            brand = watch.get('brand', '').strip()
            if not brand:
                continue
                
            # Get text description (could be in different fields)
            description = ""
            if 'description' in watch:
                description = str(watch['description'])
            elif 'enhanced_description' in watch:
                description = str(watch['enhanced_description'])
            elif 'text_description' in watch:
                description = str(watch['text_description'])
            else:
                # Try to build from available fields
                parts = []
                for field in ['model', 'series', 'case_material', 'movement', 'features']:
                    if field in watch and watch[field]:
                        parts.append(str(watch[field]))
                description = " ".join(parts)
            
            if not description:
                continue
                
            description_lengths.append(len(description))
            total_watches += 1
            
            # Count brand mentions (case-insensitive)
            brand_lower = brand.lower()
            description_lower = description.lower()
            
            # Count exact brand name matches
            exact_matches = len(re.findall(r'\b' + re.escape(brand_lower) + r'\b', description_lower))
            
            # Also check for partial matches (brand as substring)
            partial_matches = description_lower.count(brand_lower)
            
            brand_mention_stats[brand].append({
                'exact_matches': exact_matches,
                'partial_matches': partial_matches,
                'description_length': len(description),
                'description': description[:200] + "..." if len(description) > 200 else description
            })
            
        except Exception as e:
            print(f"Error processing watch: {e}")
            continue
    
    # Analyze results
    print(f"\n📊 BRAND MENTION ANALYSIS")
    print(f"=" * 50)
    print(f"Total watches analyzed: {total_watches}")
    print(f"Unique brands: {len(brand_mention_stats)}")
    print(f"Average description length: {sum(description_lengths)/len(description_lengths):.1f} chars")
    
    # Overall statistics
    all_exact_mentions = []
    all_partial_mentions = []
    
    for brand, mentions in brand_mention_stats.items():
        for mention_data in mentions:
            all_exact_mentions.append(mention_data['exact_matches'])
            all_partial_mentions.append(mention_data['partial_matches'])
    
    print(f"\n🎯 MENTION FREQUENCY:")
    print(f"Average exact mentions per watch: {sum(all_exact_mentions)/len(all_exact_mentions):.2f}")
    print(f"Average partial mentions per watch: {sum(all_partial_mentions)/len(all_partial_mentions):.2f}")
    
    # Distribution of mention counts
    exact_counter = Counter(all_exact_mentions)
    partial_counter = Counter(all_partial_mentions)
    
    print(f"\n📈 EXACT MENTION DISTRIBUTION:")
    for count in sorted(exact_counter.keys())[:10]:  # Top 10
        percentage = (exact_counter[count] / len(all_exact_mentions)) * 100
        print(f"  {count} mentions: {exact_counter[count]} watches ({percentage:.1f}%)")
    
    print(f"\n📈 PARTIAL MENTION DISTRIBUTION:")
    for count in sorted(partial_counter.keys())[:10]:  # Top 10
        percentage = (partial_counter[count] / len(all_partial_mentions)) * 100
        print(f"  {count} mentions: {partial_counter[count]} watches ({percentage:.1f}%)")
    
    # Sample brand analysis
    print(f"\n🔍 SAMPLE BRAND ANALYSIS:")
    print(f"=" * 50)
    
    # Sort brands by total watches
    sorted_brands = sorted(brand_mention_stats.items(), 
                          key=lambda x: len(x[1]), reverse=True)
    
    for brand, mentions in sorted_brands[:5]:  # Top 5 brands by watch count
        exact_mentions = [m['exact_matches'] for m in mentions]
        partial_mentions = [m['partial_matches'] for m in mentions]
        
        print(f"\n📱 {brand} ({len(mentions)} watches):")
        print(f"  Exact mentions - avg: {sum(exact_mentions)/len(exact_mentions):.2f}, "
              f"range: {min(exact_mentions)}-{max(exact_mentions)}")
        print(f"  Partial mentions - avg: {sum(partial_mentions)/len(partial_mentions):.2f}, "
              f"range: {min(partial_mentions)}-{max(partial_mentions)}")
        
        # Show sample description
        sample_watch = mentions[0]
        print(f"  Sample: \"{sample_watch['description']}\"")
    
    # Find extreme cases
    print(f"\n⚡ EXTREME CASES:")
    print(f"=" * 30)
    
    max_exact = max(all_exact_mentions)
    max_partial = max(all_partial_mentions)
    
    print(f"Maximum exact mentions: {max_exact}")
    print(f"Maximum partial mentions: {max_partial}")
    
    # Find watches with highest mention counts
    for brand, mentions in brand_mention_stats.items():
        for mention_data in mentions:
            if mention_data['exact_matches'] == max_exact:
                print(f"\nHighest exact mentions ({max_exact}): {brand}")
                print(f"Description: \"{mention_data['description']}\"")
                break
    
    return brand_mention_stats

if __name__ == "__main__":
    analyze_brand_mentions() 