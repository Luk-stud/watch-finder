from models.optimized_linucb_engine import OptimizedLinUCBEngine
from utils.filter_utils import should_include_watch
import json

# Test the filter function directly
engine = OptimizedLinUCBEngine()
watches = list(engine.watch_data.values())[:10]

# Test diameter filter
filter_prefs = {'maxDiameter': 36}
print('Testing diameter filter with maxDiameter: 36')
print('='*50)

for i, watch in enumerate(watches):
    specs = watch.get('specs', {})
    diameter = specs.get('diameter_mm', 'None')
    result = should_include_watch(watch, filter_prefs)
    print(f'Watch {i+1}: diameter={diameter}mm, passes_filter={result}, brand={watch.get("brand", "Unknown")}')

print('\n' + '='*50)
print('Testing dial color filter with dialColors: ["Black"]')
print('='*50)

# Test dial color filter  
filter_prefs = {'dialColors': ['Black']}
for i, watch in enumerate(watches):
    specs = watch.get('specs', {})
    dial_color = specs.get('dial_color', 'None')
    result = should_include_watch(watch, filter_prefs)
    print(f'Watch {i+1}: dial_color="{dial_color}", passes_filter={result}, brand={watch.get("brand", "Unknown")}') 