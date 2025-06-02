#!/usr/bin/env python3
"""
Test script for watch description generation.

This script tests the description generation with a small sample of watches
to verify the system works before running on the full dataset.
"""

import os
import sys
from dotenv import load_dotenv
from generate_text_embeddings import WatchTextEmbeddingGenerator

# Load environment variables
load_dotenv()

def test_single_watch():
    """Test description generation for a single watch."""
    print("🧪 Testing Single Watch Description Generation")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        return False
    
    try:
        # Initialize generator
        generator = WatchTextEmbeddingGenerator()
        
        # Test watches
        test_watches = [
            {"brand": "Rolex", "model": "Submariner"},
            {"brand": "Omega", "model": "Speedmaster"},
            {"brand": "Patek Philippe", "model": "Nautilus"},
            {"brand": "Audemars Piguet", "model": "Royal Oak"},
            {"brand": "Seiko", "model": "5 Sport"}
        ]
        
        for watch in test_watches:
            brand = watch["brand"]
            model = watch["model"]
            
            print(f"\n🕐 Generating description for {brand} {model}...")
            
            # Generate description
            description = generator.generate_watch_description(brand, model)
            print(f"📝 Description: {description}")
            
            # Generate embedding
            print(f"🔢 Generating embedding...")
            embedding = generator.generate_embedding(description)
            
            if embedding:
                print(f"✅ Embedding generated (dimension: {len(embedding)})")
            else:
                print(f"❌ Failed to generate embedding")
            
            print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def test_sample_processing():
    """Test processing a small sample of watches from the production scrape data."""
    print("\n🧪 Testing Sample Watch Processing")
    print("=" * 50)
    
    try:
        # Load a small sample from production scrape data
        from generate_text_embeddings import load_production_scrape_data
        production_scrape_path = "../production_scrape_20250601_175426"
        
        if not os.path.exists(production_scrape_path):
            print(f"❌ Production scrape not found at {production_scrape_path}")
            return False
        
        all_watch_data = load_production_scrape_data(production_scrape_path)
        
        # Take first 3 watches for testing
        sample_watches = all_watch_data[:3]
        print(f"📊 Testing with {len(sample_watches)} sample watches")
        
        # Initialize generator
        generator = WatchTextEmbeddingGenerator()
        
        # Process sample
        embedding_matrix, enhanced_watches = generator.process_watch_data(
            sample_watches, 
            output_dir="test_output"
        )
        
        if len(enhanced_watches) > 0:
            print(f"✅ Successfully processed {len(enhanced_watches)} watches")
            print(f"📊 Embedding matrix shape: {embedding_matrix.shape}")
            
            # Show first result
            first_watch = enhanced_watches[0]
            print(f"\n📝 Sample result:")
            print(f"Brand: {first_watch.get('brand', 'N/A')}")
            print(f"Model: {first_watch.get('model', 'N/A')}")
            print(f"Description: {first_watch.get('ai_description', 'N/A')}")
            
            return True
        else:
            print("❌ No watches were successfully processed")
            return False
            
    except Exception as e:
        print(f"❌ Error during sample processing: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Watch Text Embedding Generator - Test Suite")
    print("=" * 60)
    
    # Test 1: Single watch descriptions
    test1_success = test_single_watch()
    
    # Test 2: Sample processing
    test2_success = test_sample_processing()
    
    # Summary
    print("\n📋 Test Results Summary")
    print("=" * 30)
    print(f"Single Watch Test: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"Sample Processing: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! Ready to run full generation.")
        print("💡 Run: python generate_text_embeddings.py")
    else:
        print("\n⚠️  Some tests failed. Please check your setup.")
        print("💡 Ensure your .env file has a valid OPENAI_API_KEY")

if __name__ == "__main__":
    main() 