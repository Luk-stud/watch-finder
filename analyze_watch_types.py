#!/usr/bin/env python3
import pickle

# Load the embeddings data
with open('data/precomputed_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

# Analyze watch types
watch_types = {}
dress_watches = []
dive_watches = []
chrono_watches = []

for watch_id, watch_data in data['watch_data'].items():
    specs = watch_data.get('specs', {})
    watch_type = specs.get('watch_type', 'Unknown')
    watch_types[watch_type] = watch_types.get(watch_type, 0) + 1
    
    # Categorize for analysis
    brand = watch_data.get('brand', '').lower()
    model = watch_data.get('model', '').lower()
    description = str(specs).lower()
    
    # Look for dress watch indicators
    if any(keyword in description or keyword in model for keyword in ['dress', 'formal', 'elegant', 'minimal']):
        dress_watches.append((watch_id, brand, model, watch_type))
    
    # Look for dive watch indicators
    if any(keyword in description or keyword in model for keyword in ['dive', 'diver', 'submariner', 'seamaster', 'waterproof']):
        dive_watches.append((watch_id, brand, model, watch_type))
    
    # Look for chronograph indicators
    if any(keyword in description or keyword in model for keyword in ['chronograph', 'chrono', 'timer']):
        chrono_watches.append((watch_id, brand, model, watch_type))

print("=== WATCH TYPE DISTRIBUTION ===")
for wt, count in sorted(watch_types.items(), key=lambda x: x[1], reverse=True):
    print(f"  {wt}: {count}")

print(f"\n=== TOTAL WATCHES: {len(data['watch_data'])} ===")

print(f"\n=== DRESS WATCHES ({len(dress_watches)}) ===")
for watch_id, brand, model, wt in dress_watches[:10]:
    print(f"  {brand} {model} ({wt})")

print(f"\n=== DIVE WATCHES ({len(dive_watches)}) ===")
for watch_id, brand, model, wt in dive_watches[:10]:
    print(f"  {brand} {model} ({wt})")

print(f"\n=== CHRONOGRAPH WATCHES ({len(chrono_watches)}) ===")
for watch_id, brand, model, wt in chrono_watches[:10]:
    print(f"  {brand} {model} ({wt})") 