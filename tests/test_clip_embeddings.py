#!/usr/bin/env python3
import sys
import os

# Add the backend models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from models.watch_embedder import WatchEmbedder

def test_clip_embeddings():
    """Test the CLIP image embedding system with a small subset of data."""
    
    print("Testing CLIP image embedding system...")
    
    # Initialize embedder with custom CLIP model
    custom_model_path = "best_clip_model_description_model.pt"
    if os.path.exists(custom_model_path):
        print(f"Testing with custom CLIP model: {custom_model_path}")
        embedder = WatchEmbedder(model_path=custom_model_path)
    else:
        print(f"Custom model {custom_model_path} not found, testing with default CLIP model")
        embedder = WatchEmbedder()
    
    # Test with just a few watches first
    data_path = "data/raw/watch_data.json"
    
    try:
        # Load just the first few watches to test
        import json
        with open(data_path, 'r') as f:
            all_data = json.load(f)
        
        # Take first 5 watches for testing
        test_data = all_data[:5]
        
        # Save test data temporarily
        test_data_path = "data/raw/test_watch_data.json"
        with open(test_data_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"Testing with {len(test_data)} watches...")
        
        # Generate embeddings
        embeddings = embedder.generate_embeddings(test_data_path)
        
        print(f"✅ Successfully generated embeddings!")
        print(f"   - Embedding shape: {embeddings.shape}")
        print(f"   - Successfully processed: {len(embedder.watch_data)} watches")
        
        # Test saving and loading
        test_embeddings_path = "embeddings/test_watch_image_embeddings.pkl"
        embedder.save_embeddings(test_embeddings_path)
        
        # Test loading
        embedder2 = WatchEmbedder()
        loaded_embeddings, loaded_data = embedder2.load_embeddings(test_embeddings_path)
        
        print(f"✅ Successfully saved and loaded embeddings!")
        print(f"   - Loaded embedding shape: {loaded_embeddings.shape}")
        print(f"   - Loaded {len(loaded_data)} watches")
        
        # Cleanup test files
        os.remove(test_data_path)
        os.remove(test_embeddings_path)
        
        print("✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_clip_embeddings()
    if success:
        print("\n🎉 CLIP image embedding system is working correctly!")
        print("You can now generate embeddings for your full dataset.")
    else:
        print("\n💥 There was an issue with the CLIP embedding system.")
        print("Please check the error messages above.") 