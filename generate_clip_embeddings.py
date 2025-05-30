#!/usr/bin/env python3
import sys
import os

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from models.watch_embedder import WatchEmbedder

def generate_new_embeddings():
    """Generate embeddings for the new complete v2 dataset (8000 watches)."""
    
    # Use custom CLIP model if available
    custom_model_path = "best_clip_model_description_model.pt"
    
    if os.path.exists(custom_model_path):
        print(f"Using custom CLIP model: {custom_model_path}")
        embedder = WatchEmbedder(model_path=custom_model_path)
    else:
        print("Using default CLIP model")
        embedder = WatchEmbedder()
    
    # Use the new complete converted data
    data_path = "data/raw/watch_data_v2_complete.json"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run convert_new_data.py first.")
        return False
    
    print(f"Loading watch data from {data_path}...")
    
    # Define output path
    output_path = "embeddings/watch_image_embeddings_v2_complete.pkl"
    print(f"Generating embeddings for complete v2 dataset...")
    print(f"Output will be saved to: {output_path}")
    
    try:
        # Generate embeddings
        embeddings = embedder.generate_embeddings(data_path)
        
        # Save embeddings
        embedder.save_embeddings(output_path)
        
        print("\n🎉 Complete dataset embeddings generated successfully!")
        print(f"📊 Dataset size: {len(embedder.watch_data)} watches")
        print(f"📐 Embedding dimension: {embeddings.shape[1]}")
        print(f"💾 Embeddings saved to: {output_path}")
        
        # Show file size
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"📁 File size: {size_mb:.1f} MB")
        
        # Show dataset statistics
        brands = set(watch['brand'] for watch in embedder.watch_data)
        with_eur_prices = sum(1 for watch in embedder.watch_data if watch['price'].startswith('€'))
        
        print(f"🏷️  Unique brands: {len(brands)}")
        print(f"💰 Watches with EUR prices: {with_eur_prices}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Generating CLIP embeddings for new dataset...")
    success = generate_new_embeddings()
    
    if success:
        print("\n🎉 Embedding generation completed!")
        print("The new embeddings are ready to use with enhanced specifications.")
        print("You can now restart the backend server to use the new data.")
    else:
        print("\n❌ Embedding generation failed.")
        print("Please check the error messages above.") 