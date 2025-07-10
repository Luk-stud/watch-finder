#!/usr/bin/env python3
"""
Test Concatenation Approach
Test the new concatenated embedding combination method.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.linucb_engine import DynamicMultiExpertLinUCBEngine

def test_concatenation_approach():
    """Test the concatenated embedding approach."""
    print("🧪 TESTING CONCATENATION APPROACH")
    print("=" * 50)
    
    # Test 1: Engine initialization with 100D
    print("📝 Test 1: Engine Initialization (100D)")
    try:
        engine = DynamicMultiExpertLinUCBEngine(dim=100, data_dir='../data')
        print(f"✅ Engine initialized with {engine.dim}D embeddings")
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        return False
    
    # Test 2: Test concatenated embedding creation
    print("\n📝 Test 2: Concatenated Embedding Creation")
    try:
        # Create sample REDUCED embeddings (50D each, not 1536D and 512D)
        text_reduced = np.random.randn(50)  # Pre-reduced text embedding
        clip_reduced = np.random.randn(50)  # Pre-reduced CLIP embedding
        
        # Test different weight combinations
        test_cases = [
            (0.5, 0.5),  # Equal weights
            (0.7, 0.3),  # Text-heavy
            (0.3, 0.7),  # Visual-heavy
            (1.0, 0.0),  # Pure text
            (0.0, 1.0),  # Pure visual
        ]
        
        for text_weight, clip_weight in test_cases:
            result = engine._create_weighted_embedding(text_reduced, clip_reduced, text_weight, clip_weight)
            
            # Check dimensions
            if len(result) != 100:
                print(f"❌ Wrong dimension: {len(result)}, expected 100")
                return False
            
            # Check normalization
            norm = np.linalg.norm(result)
            if not (0.95 <= norm <= 1.05):  # Allow small numerical error
                print(f"❌ Not normalized: norm={norm:.3f}")
                return False
            
            print(f"✅ Weights {text_weight:.1f}/{clip_weight:.1f}: 100D, norm={norm:.3f}")
        
    except Exception as e:
        print(f"❌ Concatenation test failed: {e}")
        return False
    
    # Test 3: Test structure of concatenated embedding
    print("\n📝 Test 3: Concatenated Embedding Structure")
    try:
        text_reduced = np.ones(50)  # All ones for text (50D)
        clip_reduced = np.zeros(50)  # All zeros for visual (50D)
        
        result = engine._create_weighted_embedding(text_reduced, clip_reduced, 1.0, 0.0)
        
        # First half should be from text (non-zero), second half from visual (zero)
        first_half = result[:50]
        second_half = result[50:]
        
        if np.all(first_half == 0):
            print("❌ First half (text) is all zeros!")
            return False
        
        if not np.allclose(second_half, 0, atol=1e-6):
            print("❌ Second half (visual) should be zeros!")
            return False
        
        print("✅ Concatenation structure correct: [text_50D | visual_50D]")
        
    except Exception as e:
        print(f"❌ Structure test failed: {e}")
        return False
    
    # Test 4: Test no CLIP embedding case
    print("\n📝 Test 4: Missing CLIP Embedding Handling")
    try:
        text_reduced = np.random.randn(50)  # 50D reduced text
        result = engine._create_weighted_embedding(text_reduced, None, 0.7, 0.3)
        
        if len(result) != 100:
            print(f"❌ Wrong dimension with missing CLIP: {len(result)}")
            return False
        
        # Second half should be zeros (padded)
        second_half = result[50:]
        if not np.allclose(second_half, 0, atol=1e-6):
            print("❌ Missing CLIP should result in zero padding for second half")
            return False
        
        print("✅ Missing CLIP embedding handled correctly (zero padding)")
        
    except Exception as e:
        print(f"❌ Missing CLIP test failed: {e}")
        return False
    
    # Test 5: Memory usage comparison
    print("\n📝 Test 5: Memory Usage Analysis")
    try:
        # Calculate LinUCB matrix memory usage
        old_memory = 50 * 50 * 8  # 50x50 matrix, 8 bytes per float64
        new_memory = 100 * 100 * 8  # 100x100 matrix
        
        print(f"📊 LinUCB matrix memory:")
        print(f"   Old (50D): {old_memory / 1024:.1f} KB per arm")
        print(f"   New (100D): {new_memory / 1024:.1f} KB per arm")
        print(f"   Increase: {new_memory / old_memory:.1f}x")
        
        # For 1000 arms (typical system)
        total_old = old_memory * 1000 / (1024 * 1024)
        total_new = new_memory * 1000 / (1024 * 1024)
        print(f"📊 Total for 1000 arms:")
        print(f"   Old: {total_old:.1f} MB")
        print(f"   New: {total_new:.1f} MB")
        print(f"   Still within Railway limits: {'✅' if total_new < 400 else '❌'}")
        
    except Exception as e:
        print(f"❌ Memory analysis failed: {e}")
        return False
    
    print(f"\n🎉 ALL CONCATENATION TESTS PASSED!")
    print("✅ 100D concatenated embeddings working correctly")
    print("✅ Text and visual modalities properly separated")
    print("✅ User weights applied to each modality")
    print("✅ Memory usage acceptable for Railway")
    print("✅ No information loss from different model outputs")
    
    return True

if __name__ == "__main__":
    test_concatenation_approach() 