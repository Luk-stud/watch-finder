#!/usr/bin/env python3
"""
Real Watch Metadata Explorer

Explore ALL pickle files to find the comprehensive watch metadata.
"""

import os
import pickle
import numpy as np
from collections import defaultdict, Counter
import json

def explore_pickle_file(filepath):
    """Explore a single pickle file."""
    
    print(f"\n{'='*60}")
    print(f"📂 EXPLORING: {filepath}")
    print(f"{'='*60}")
    
    try:
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"📊 File size: {file_size:.1f} MB")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Successfully loaded pickle file")
        
        # Analyze top-level structure
        print(f"\n🏗️ TOP-LEVEL STRUCTURE:")
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    print(f"   • {key}: dict with {len(value)} items")
                elif isinstance(value, (list, tuple)):
                    print(f"   • {key}: {type(value).__name__} with {len(value)} items")
                elif isinstance(value, np.ndarray):
                    print(f"   • {key}: numpy array shape {value.shape}")
                else:
                    print(f"   • {key}: {type(value).__name__} = {value}")
        elif isinstance(data, (list, tuple)):
            print(f"   Root is {type(data).__name__} with {len(data)} items")
            if len(data) > 0:
                print(f"   First item type: {type(data[0])}")
                if hasattr(data[0], 'keys'):
                    print(f"   First item keys: {list(data[0].keys())}")
        else:
            print(f"   Root type: {type(data)}")
        
        # If it's a dict with watch data, explore it
        if isinstance(data, dict):
            # Look for watch data - could be direct or nested
            watch_data_keys = []
            for key, value in data.items():
                if isinstance(value, dict) and len(value) > 0:
                    # Check if this looks like watch data
                    sample_key = next(iter(value.keys()))
                    sample_value = next(iter(value.values()))
                    if isinstance(sample_value, dict):
                        watch_data_keys.append(key)
            
            if watch_data_keys:
                print(f"\n🕰️ POTENTIAL WATCH DATA SECTIONS: {watch_data_keys}")
                
                for section_key in watch_data_keys:
                    section_data = data[section_key]
                    print(f"\n   📋 ANALYZING SECTION: {section_key}")
                    print(f"   📊 Contains {len(section_data)} items")
                    
                    # Analyze fields in this section
                    all_fields = set()
                    field_counts = defaultdict(int)
                    field_samples = defaultdict(list)
                    
                    for item_id, item_data in list(section_data.items())[:100]:  # Sample first 100
                        if isinstance(item_data, dict):
                            for field, value in item_data.items():
                                all_fields.add(field)
                                field_counts[field] += 1
                                if len(field_samples[field]) < 3:
                                    field_samples[field].append(value)
                    
                    print(f"   📋 FIELDS IN {section_key} ({len(all_fields)} total):")
                    for field in sorted(all_fields):
                        coverage = (field_counts[field] / min(len(section_data), 100)) * 100
                        samples = field_samples[field]
                        sample_str = ", ".join([f"'{s}'" if isinstance(s, str) else str(s) for s in samples])
                        if len(sample_str) > 80:
                            sample_str = sample_str[:77] + "..."
                        print(f"     • {field}: {coverage:.0f}% | {sample_str}")
                    
                    # Show a complete sample record
                    sample_id, sample_data = next(iter(section_data.items()))
                    print(f"\n   🔍 SAMPLE RECORD (ID: {sample_id}):")
                    if isinstance(sample_data, dict):
                        for field, value in sample_data.items():
                            if isinstance(value, str) and len(value) > 100:
                                value = value[:97] + "..."
                            print(f"     • {field}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Explore all pickle files."""
    
    print("=" * 80)
    print("🔍 COMPREHENSIVE PICKLE FILE EXPLORATION")
    print("=" * 80)
    
    # Find all pickle files
    pickle_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pkl'):
                pickle_files.append(os.path.join(root, file))
    
    print(f"📂 Found {len(pickle_files)} pickle files:")
    for pf in pickle_files:
        print(f"   • {pf}")
    
    # Explore each file
    for pkl_file in pickle_files:
        explore_pickle_file(pkl_file)
    
    print(f"\n" + "=" * 80)
    print(f"🎉 EXPLORATION COMPLETE!")
    print(f"=" * 80)

if __name__ == "__main__":
    main() 