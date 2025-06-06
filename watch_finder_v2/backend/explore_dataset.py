#!/usr/bin/env python3
"""
Explore Watch Dataset: Find recognizable watches for realistic testing
=====================================================================
"""

import os
import sys
import pickle
import numpy as np
from collections import defaultdict

# Add backend directory to path
sys.path.append(os.path.dirname(__file__))

def explore_watch_dataset():
    """Explore the dataset to find recognizable watches."""
    print("🔍 Exploring Watch Dataset...")
    
    # Load watch metadata
    data_dir = 'data'
    metadata_path = os.path.join(data_dir, 'watch_text_metadata.pkl')
    
    try:
        with open(metadata_path, 'rb') as f:
            metadata_list = pickle.load(f)
        
        print(f"📊 Total watches in dataset: {len(metadata_list)}")
        
        # Categorize by brand
        brands = defaultdict(list)
        notable_watches = []
        
        for idx, watch_dict in enumerate(metadata_list[:100]):  # Sample first 100
            brand = watch_dict.get('brand', '').strip()
            model = watch_dict.get('model', '').strip()
            category = watch_dict.get('category', '')
            price = watch_dict.get('price', 0)
            
            if brand:
                brands[brand].append({
                    'id': idx,
                    'model': model,
                    'category': category,
                    'price': price,
                    'full_name': f"{brand} {model}".strip()
                })
                
                # Look for notable/recognizable watches
                brand_lower = brand.lower()
                model_lower = model.lower()
                
                # Famous brands
                if any(famous in brand_lower for famous in [
                    'rolex', 'omega', 'seiko', 'casio', 'citizen', 'tissot', 
                    'hamilton', 'orient', 'timex', 'swatch', 'breitling'
                ]):
                    notable_watches.append({
                        'id': idx,
                        'brand': brand,
                        'model': model,
                        'category': category,
                        'price': price,
                        'reason': 'Famous brand'
                    })
                
                # Iconic models
                if any(iconic in model_lower for iconic in [
                    'submariner', 'speedmaster', 'datejust', 'gmt', 'daytona',
                    'seamaster', 'skx', 'turtle', 'samurai', 'field', 'diver'
                ]):
                    notable_watches.append({
                        'id': idx,
                        'brand': brand,
                        'model': model,
                        'category': category,
                        'price': price,
                        'reason': 'Iconic model'
                    })
        
        # Show brand distribution
        print(f"\n📈 Top 10 Brands by Count:")
        sorted_brands = sorted(brands.items(), key=lambda x: len(x[1]), reverse=True)
        for brand, watches in sorted_brands[:10]:
            print(f"  {brand}: {len(watches)} watches")
        
        # Show notable watches
        print(f"\n⭐ Notable/Recognizable Watches Found:")
        for watch in notable_watches[:20]:  # Top 20
            print(f"  ID {watch['id']:3d}: {watch['brand']} {watch['model']} "
                  f"(${watch['price']}) - {watch['reason']}")
        
        # Sample some interesting categories
        print(f"\n🎯 Sample Watches by Category:")
        categories = defaultdict(list)
        for watch_list in brands.values():
            for watch in watch_list:
                if watch['category']:
                    categories[watch['category']].append(watch)
        
        for category, watches in list(categories.items())[:5]:
            print(f"\n  {category}:")
            for watch in watches[:3]:
                print(f"    • {watch['full_name']} (${watch['price']})")
        
        return notable_watches, brands, categories
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return [], {}, {}

def create_test_scenarios(notable_watches, brands):
    """Create realistic test scenarios based on watch knowledge."""
    print(f"\n🎯 Creating Realistic Test Scenarios...")
    
    scenarios = []
    
    # Scenario 1: Sports Watch Enthusiast
    sports_watches = [w for w in notable_watches if any(
        term in w['model'].lower() for term in ['diver', 'sport', 'chrono', 'speedmaster', 'submariner']
    )]
    if sports_watches:
        scenario = {
            'name': 'Sports Watch Enthusiast',
            'description': 'User who likes robust, sporty watches for active lifestyle',
            'seed_watches': sports_watches[:2],
            'expected_recommendations': 'More sports/dive watches, chronographs, tool watches',
            'avoid': 'Dress watches, minimal designs, precious metals'
        }
        scenarios.append(scenario)
    
    # Scenario 2: Dress Watch Collector
    dress_watches = [w for w in notable_watches if any(
        term in w['model'].lower() for term in ['dress', 'classic', 'elegance', 'gold', 'formal']
    )]
    if dress_watches:
        scenario = {
            'name': 'Dress Watch Collector',
            'description': 'User who prefers elegant, formal timepieces',
            'seed_watches': dress_watches[:2],
            'expected_recommendations': 'Thin profiles, leather straps, precious metals, clean dials',
            'avoid': 'Large sport watches, rubber straps, bold colors'
        }
        scenarios.append(scenario)
    
    # Scenario 3: Budget-Conscious Entry Level
    budget_watches = [w for w in notable_watches if w['price'] < 500 and any(
        brand.lower() in w['brand'].lower() for brand in ['seiko', 'casio', 'citizen', 'timex', 'orient']
    )]
    if budget_watches:
        scenario = {
            'name': 'Budget-Conscious Entry Level',
            'description': 'New to watches, looking for value and reliability',
            'seed_watches': budget_watches[:2],
            'expected_recommendations': 'Reliable movements, good value, established brands under $500',
            'avoid': 'Luxury brands, complications, precious metals'
        }
        scenarios.append(scenario)
    
    # Scenario 4: Luxury Watch Connoisseur
    luxury_watches = [w for w in notable_watches if w['price'] > 2000 and any(
        brand.lower() in w['brand'].lower() for brand in ['rolex', 'omega', 'breitling', 'iwc']
    )]
    if luxury_watches:
        scenario = {
            'name': 'Luxury Watch Connoisseur',
            'description': 'Experienced collector focusing on high-end pieces',
            'seed_watches': luxury_watches[:2],
            'expected_recommendations': 'Swiss movements, precious metals, complications, heritage brands',
            'avoid': 'Quartz watches, plastic cases, unknown brands'
        }
        scenarios.append(scenario)
    
    # Print scenarios
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}️⃣ **{scenario['name']}**")
        print(f"   Description: {scenario['description']}")
        print(f"   Seed watches:")
        for watch in scenario['seed_watches']:
            print(f"     • {watch['brand']} {watch['model']} (${watch['price']})")
        print(f"   Expected: {scenario['expected_recommendations']}")
        print(f"   Should avoid: {scenario['avoid']}")
    
    return scenarios

if __name__ == "__main__":
    notable_watches, brands, categories = explore_watch_dataset()
    if notable_watches:
        scenarios = create_test_scenarios(notable_watches, brands)
        print(f"\n✅ Found {len(scenarios)} realistic test scenarios!")
    else:
        print("❌ No notable watches found for scenario creation") 