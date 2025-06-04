#!/usr/bin/env python3
"""
Check Watch Types in Dataset
"""

import pickle

def analyze_watch_types():
    # Load metadata
    with open('watch_text_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    # Count watch types
    types_count = {}
    total = len(metadata)
    
    for watch in metadata:
        watch_type = watch.get('specs', {}).get('watch_type', '')
        types_count[watch_type] = types_count.get(watch_type, 0) + 1
    
    print(f'Total watches: {total}')
    print('\nWatch type distribution:')
    for watch_type, count in sorted(types_count.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total * 100
        print(f'  {watch_type}: {count} ({percentage:.1f}%)')
    
    # Check how many have actual types (not "-")
    with_types = sum(count for wt, count in types_count.items() if wt and wt != '-')
    print(f'\nWatches with type info: {with_types} ({with_types/total*100:.1f}%)')
    print(f'Watches without type info: {total - with_types} ({(total - with_types)/total*100:.1f}%)')
    
    # Show actual watch types (excluding "-" and empty)
    actual_types = {wt: count for wt, count in types_count.items() if wt and wt != '-'}
    print(f'\nActual watch types ({len(actual_types)} types):')
    for watch_type, count in sorted(actual_types.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total * 100
        print(f'  {watch_type}: {count} ({percentage:.1f}%)')

if __name__ == "__main__":
    analyze_watch_types() 