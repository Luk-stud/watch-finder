import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'watch_finder_v2', 'backend'))
import pickle
from models.simple_sgd_engine import SimpleSgdEngine

# Load data
with open('watch_finder_v2/backend/data/precomputed_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

engine = SimpleSgdEngine()

# Pick one Vallelunga watch
vallelunga_id = 'watch_13'
group = engine._get_similar_watches(vallelunga_id)
print(f'Brand+model group for {vallelunga_id}: {group}')

for wid in group:
    w = data['watch_data'][wid]
    print(f'  - {w.get("brand")} {w.get("model")} (ID: {wid})')
    print(f'    Specs: {w.get("specs")}')
    print(f'    Image: {w.get("image_url") or w.get("main_image")}') 