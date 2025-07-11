import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'watch_finder_v2', 'backend'))
from models.simple_sgd_engine import SimpleSgdEngine

engine = SimpleSgdEngine()
session_id = 'test_variants_session'

print("🧪 Testing variant filtering in single batch...")

# Get recommendations
recs = engine.get_recommendations(session_id)
print(f"✅ Got {len(recs)} recommendations")

# Check for duplicates in the same batch
brand_model_counts = {}
for rec in recs:
    brand = rec['brand'].lower().strip()
    model = rec['model'].lower().strip()
    key = f"{brand}|{model}"
    brand_model_counts[key] = brand_model_counts.get(key, 0) + 1

print(f"📊 Brand+model groups in this batch:")
for key, count in brand_model_counts.items():
    brand, model = key.split('|', 1)
    print(f"  - {brand} {model}: {count} watches")
    if count > 1:
        print(f"    ❌ ERROR: Multiple watches from same group in single batch!")

# Check if any have more than 1
duplicates = [key for key, count in brand_model_counts.items() if count > 1]
if duplicates:
    print(f"❌ Found {len(duplicates)} groups with duplicates in single batch")
else:
    print("✅ SUCCESS: No duplicates in single batch") 