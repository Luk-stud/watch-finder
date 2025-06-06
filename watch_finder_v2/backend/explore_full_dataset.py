#!/usr/bin/env python3
"""
Comprehensive Dataset Exploration: Understand watch categories and create realistic scenarios
===========================================================================================
"""

import os
import sys
import pickle
import numpy as np
from collections import defaultdict, Counter

# Add backend directory to path
sys.path.append(os.path.dirname(__file__))

def analyze_full_dataset():
    """Comprehensive analysis of the full watch dataset."""
    print("🔍 Comprehensive Dataset Analysis...")
    
    # Load watch metadata
    data_dir = 'data'
    metadata_path = os.path.join(data_dir, 'watch_text_metadata.pkl')
    
    try:
        with open(metadata_path, 'rb') as f:
            metadata_list = pickle.load(f)
        
        print(f"📊 Total watches: {len(metadata_list)}")
        
        # Analyze all watches
        brands = defaultdict(list)
        price_ranges = defaultdict(list)
        categories = defaultdict(list)
        watch_types = defaultdict(list)
        
        for idx, watch_dict in enumerate(metadata_list):
            brand = watch_dict.get('brand', '').strip()
            model = watch_dict.get('model', '').strip()
            price = watch_dict.get('price', 0)
            category = watch_dict.get('category', '').strip()
            
            # Categorize by price
            if price == 0:
                price_range = 'Unknown'
            elif price < 300:
                price_range = 'Budget (<$300)'
            elif price < 800:
                price_range = 'Mid-range ($300-800)'
            elif price < 2000:
                price_range = 'Premium ($800-2000)'
            else:
                price_range = 'Luxury (>$2000)'
            
            price_ranges[price_range].append({
                'id': idx, 'brand': brand, 'model': model, 'price': price
            })
            
            # Categorize by type (based on model name)
            model_lower = model.lower()
            if any(term in model_lower for term in ['diver', 'dive', 'compressor', 'submariner']):
                watch_type = 'Dive Watch'
            elif any(term in model_lower for term in ['field', 'military', 'pilot', 'aviation']):
                watch_type = 'Field/Military Watch'
            elif any(term in model_lower for term in ['chrono', 'chronograph', 'timer', 'racing']):
                watch_type = 'Chronograph'
            elif any(term in model_lower for term in ['gmt', 'world', 'dual', 'travel']):
                watch_type = 'GMT/Travel Watch'
            elif any(term in model_lower for term in ['dress', 'formal', 'classic', 'elegance']):
                watch_type = 'Dress Watch'
            elif any(term in model_lower for term in ['sport', 'racing', 'speed']):
                watch_type = 'Sports Watch'
            else:
                watch_type = 'General/Other'
            
            watch_types[watch_type].append({
                'id': idx, 'brand': brand, 'model': model, 'price': price
            })
            
            if brand:
                brands[brand].append({
                    'id': idx, 'model': model, 'price': price, 'type': watch_type
                })
        
        # Print analysis
        print(f"\n📈 Top 15 Brands:")
        for brand, watches in sorted(brands.items(), key=lambda x: len(x[1]), reverse=True)[:15]:
            avg_price = np.mean([w['price'] for w in watches if w['price'] > 0])
            types = Counter([w['type'] for w in watches])
            main_type = types.most_common(1)[0][0] if types else 'Unknown'
            print(f"  {brand:25} | {len(watches):2d} watches | Avg: ${avg_price:4.0f} | Main: {main_type}")
        
        print(f"\n💰 Price Distribution:")
        for price_range, watches in price_ranges.items():
            print(f"  {price_range:20} | {len(watches):3d} watches")
        
        print(f"\n⌚ Watch Types:")
        for watch_type, watches in sorted(watch_types.items(), key=lambda x: len(x[1]), reverse=True):
            avg_price = np.mean([w['price'] for w in watches if w['price'] > 0])
            print(f"  {watch_type:20} | {len(watches):3d} watches | Avg: ${avg_price:4.0f}")
        
        return brands, price_ranges, watch_types
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}, {}, {}

def create_realistic_scenarios(brands, price_ranges, watch_types):
    """Create realistic test scenarios based on actual dataset."""
    print(f"\n🎯 Creating Realistic Test Scenarios...")
    
    scenarios = []
    
    # Scenario 1: Dive Watch Enthusiast
    dive_watches = watch_types.get('Dive Watch', [])
    if len(dive_watches) >= 2:
        selected = sorted(dive_watches, key=lambda x: x['price'])[:2]  # Start with cheaper ones
        scenarios.append({
            'name': 'Dive Watch Enthusiast',
            'description': 'Loves underwater timekeeping and robust sports watches',
            'seed_watches': selected,
            'expected_behavior': 'Should recommend more dive watches, robust sports watches',
            'test_goals': ['Create dive watch expert', 'Recommend similar water-resistant watches']
        })
    
    # Scenario 2: Field Watch Collector  
    field_watches = watch_types.get('Field/Military Watch', [])
    if len(field_watches) >= 2:
        selected = sorted(field_watches, key=lambda x: x['price'])[:2]
        scenarios.append({
            'name': 'Field Watch Collector',
            'description': 'Appreciates tool watches with military heritage and functionality',
            'seed_watches': selected,
            'expected_behavior': 'Should recommend more field watches, tool watches, military-inspired pieces',
            'test_goals': ['Create field watch expert', 'Avoid dress/luxury watches']
        })
    
    # Scenario 3: Budget-Conscious Buyer
    budget_watches = price_ranges.get('Budget (<$300)', []) + price_ranges.get('Mid-range ($300-800)', [])
    if len(budget_watches) >= 2:
        # Pick diverse types within budget
        selected = []
        seen_brands = set()
        for watch in budget_watches:
            if watch['brand'] not in seen_brands and len(selected) < 2:
                selected.append(watch)
                seen_brands.add(watch['brand'])
        
        scenarios.append({
            'name': 'Budget-Conscious Buyer',
            'description': 'Seeks value and quality under $800',
            'seed_watches': selected,
            'expected_behavior': 'Should recommend affordable watches with good value',
            'test_goals': ['Create budget-focused expert', 'Avoid luxury recommendations']
        })
    
    # Scenario 4: Independent Brand Explorer
    # Pick from top microbrand: Out Of Order, Carpenter, Traska, etc.
    microbrands = ['Out Of Order', 'Carpenter', 'Traska', 'Oak & Oscar', 'Studio Underd0g']
    microbrand_watches = []
    for brand in microbrands:
        if brand in brands:
            microbrand_watches.extend(brands[brand])
    
    if len(microbrand_watches) >= 2:
        selected = microbrand_watches[:2]
        scenarios.append({
            'name': 'Independent Brand Explorer',
            'description': 'Interested in unique microbrands and independent watchmakers',
            'seed_watches': selected,
            'expected_behavior': 'Should recommend other independent brands, unique designs',
            'test_goals': ['Create microbrand expert', 'Recommend diverse independent watches']
        })
    
    # Scenario 5: Luxury Watch Aspirant
    luxury_watches = price_ranges.get('Luxury (>$2000)', []) + price_ranges.get('Premium ($800-2000)', [])
    if len(luxury_watches) >= 2:
        selected = sorted(luxury_watches, key=lambda x: x['price'], reverse=True)[:2]
        scenarios.append({
            'name': 'Luxury Watch Aspirant',
            'description': 'Appreciates high-end craftsmanship and premium materials',
            'seed_watches': selected,
            'expected_behavior': 'Should recommend premium watches, avoid budget options',
            'test_goals': ['Create luxury expert', 'Maintain price preference']
        })
    
    # Print scenarios
    print(f"\n📋 Test Scenarios Created:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}️⃣ **{scenario['name']}**")
        print(f"   📝 {scenario['description']}")
        print(f"   🎯 Seed watches:")
        for watch in scenario['seed_watches']:
            print(f"     • {watch['brand']} {watch['model']} (${watch['price']})")
        print(f"   🤖 Expected: {scenario['expected_behavior']}")
        print(f"   ✅ Goals: {', '.join(scenario['test_goals'])}")
    
    return scenarios

def find_specific_watches(brands, watch_types):
    """Find specific interesting watches for manual testing."""
    print(f"\n🔎 Interesting Watches for Manual Testing:")
    
    interesting = []
    
    # Most expensive watch
    all_watches = []
    for brand_watches in brands.values():
        all_watches.extend(brand_watches)
    
    if all_watches:
        most_expensive = max(all_watches, key=lambda x: x['price'])
        interesting.append(('Most Expensive', most_expensive))
        
        cheapest = min([w for w in all_watches if w['price'] > 0], key=lambda x: x['price'])
        interesting.append(('Cheapest', cheapest))
    
    # Representative from each type
    for watch_type, watches in watch_types.items():
        if watches:
            # Pick mid-price representative
            sorted_watches = sorted([w for w in watches if w['price'] > 0], key=lambda x: x['price'])
            if sorted_watches:
                mid_idx = len(sorted_watches) // 2
                representative = sorted_watches[mid_idx]
                interesting.append((f'Representative {watch_type}', representative))
    
    for category, watch in interesting[:8]:  # Top 8
        print(f"  {category:25} | {watch['brand']} {watch['model']} (${watch['price']})")
    
    return interesting

if __name__ == "__main__":
    brands, price_ranges, watch_types = analyze_full_dataset()
    if brands:
        scenarios = create_realistic_scenarios(brands, price_ranges, watch_types)
        interesting_watches = find_specific_watches(brands, watch_types)
        print(f"\n✅ Created {len(scenarios)} scenarios with real dataset knowledge!")
    else:
        print("❌ Failed to analyze dataset") 