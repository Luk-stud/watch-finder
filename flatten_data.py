import json
import pickle

# Load final scrape data
with open('backend/data/final_scrape.json', 'r') as f:
    data = json.load(f)

# Flatten the nested structure
flattened = []
for watch in data:
    flat_watch = watch.get('specs', {}).copy()
    flat_watch['main_image'] = watch.get('main_image', '')
    flat_watch['brand_website'] = watch.get('brand_website', '')
    flattened.append(flat_watch)

print(f'Flattened {len(flattened)} watches')
print(f'Sample fields: {list(flattened[0].keys()) if flattened else "No data"}')
print(f'Sample: brand={flattened[0].get("brand")}, model={flattened[0].get("model")}')

# Save as pickle
with open('backend/data/watch_metadata.pkl', 'wb') as f:
    pickle.dump(flattened, f)

print('Saved flattened metadata to watch_metadata.pkl') 