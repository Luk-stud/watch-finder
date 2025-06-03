#!/usr/bin/env python3
"""
Test script for the variants endpoint.
Tests getting variants of specific watches to ensure the endpoint works correctly.
"""

import requests
import json

def test_variants_endpoint():
    """Test the get-variants endpoint."""
    
    base_url = "http://localhost:5001"
    
    print("🧪 Testing Variants Endpoint")
    print("=" * 40)
    
    # Test some watch indices that likely have variants
    test_indices = [0, 100, 500, 1000, 1500]  # Sample various indices
    
    for watch_index in test_indices:
        print(f"\n🔍 Testing variants for watch index {watch_index}:")
        
        try:
            response = requests.get(f"{base_url}/api/get-variants/{watch_index}")
            
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] == 'success':
                    brand = data.get('brand', 'Unknown')
                    model = data.get('model', 'Unknown')
                    variant_count = data.get('variant_count', 0)
                    signature = data.get('signature', 'Unknown')
                    
                    print(f"   ✅ {brand} - {model}")
                    print(f"   📊 Signature: {signature}")
                    print(f"   🔢 Variants found: {variant_count}")
                    
                    if variant_count > 1:
                        print(f"   📝 Variants:")
                        variants = data.get('variants', [])
                        for i, variant in enumerate(variants[:5], 1):  # Show first 5
                            variant_details = variant.get('variant_details', [])
                            is_target = variant.get('is_target', False)
                            target_marker = " (TARGET)" if is_target else ""
                            
                            if variant_details:
                                details_str = " | ".join(variant_details)
                                print(f"      {i}. {variant.get('model', 'Unknown')} - {details_str}{target_marker}")
                            else:
                                print(f"      {i}. {variant.get('model', 'Unknown')}{target_marker}")
                        
                        if len(variants) > 5:
                            print(f"      ... and {len(variants) - 5} more variants")
                    else:
                        print(f"   ℹ️  No variants found (unique model)")
                        
                    # Stop after finding first watch with multiple variants
                    if variant_count > 1:
                        print(f"\n🎯 Found example with variants, stopping test.")
                        break
                        
                else:
                    print(f"   ❌ Error: {data.get('message', 'Unknown error')}")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    print(f"\n✅ Variants endpoint test completed!")

if __name__ == "__main__":
    test_variants_endpoint() 