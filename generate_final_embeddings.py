#!/usr/bin/env python3
"""
Generate embeddings from the final_scrape.csv data
"""
import sys
import os

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from models.watch_embedder import WatchEmbedder

def generate_final_embeddings():
    """Generate embeddings for the final scrape dataset."""
    
    # Use custom CLIP model if available
    custom_model_path = "best_clip_model_description_model.pt"
    
    if os.path.exists(custom_model_path):
        print(f"Using custom CLIP model: {custom_model_path}")
        embedder = WatchEmbedder(model_path=custom_model_path)
    else:
        print("Using default CLIP model")
        embedder = WatchEmbedder()
    
    # Use the converted final scrape data
    data_path = "data/raw/watch_data_final_scrape.json"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run convert_final_scrape.py first.")
        return False
    
    print(f"Loading watch data from {data_path}...")
    
    # Define output path
    output_path = "embeddings/watch_image_embeddings_final_scrape.pkl"
    print(f"Generating embeddings for final scrape dataset...")
    print(f"Output will be saved to: {output_path}")
    
    try:
        # Generate embeddings
        embeddings = embedder.generate_embeddings(data_path)
        
        # Save embeddings
        embedder.save_embeddings(output_path)
        
        print("\n🎉 Final scrape dataset embeddings generated successfully!")
        print(f"📊 Dataset size: {len(embedder.watch_data)} watches")
        print(f"📐 Embedding dimension: {embeddings.shape[1]}")
        print(f"💾 Embeddings saved to: {output_path}")
        
        # Show file size
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"📁 File size: {size_mb:.1f} MB")
        
        # Show dataset statistics
        brands = set(watch['brand'] for watch in embedder.watch_data)
        with_prices = sum(1 for watch in embedder.watch_data if not watch['price'].startswith('Contact'))
        with_specs = sum(1 for watch in embedder.watch_data if 'specs' in watch and watch['specs'])
        with_series = sum(1 for watch in embedder.watch_data 
                         if 'specs' in watch and watch['specs'] and 
                         watch['specs'].get('serie', '') and watch['specs']['serie'] != '-')
        with_images = sum(1 for watch in embedder.watch_data if watch.get('image_url', ''))
        
        print(f"🏷️  Unique brands: {len(brands)}")
        print(f"💰 Watches with prices: {with_prices}")
        print(f"📋 Watches with detailed specs: {with_specs}")
        print(f"📚 Watches with series info: {with_series}")
        print(f"🖼️  Watches with images: {with_images}")
        
        # Show series statistics
        series_counts = {}
        for watch in embedder.watch_data:
            if 'specs' in watch and watch['specs']:
                series = watch['specs'].get('serie', 'Unknown')
                if series and series != '-':
                    series_counts[series] = series_counts.get(series, 0) + 1
        
        print(f"🔗 Unique series: {len(series_counts)}")
        
        # Show top 10 series by watch count
        if series_counts:
            top_series = sorted(series_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"📈 Top series by watch count:")
            for series, count in top_series:
                print(f"   - {series}: {count} watches")
        
        # Show brand distribution
        brand_counts = {}
        for watch in embedder.watch_data:
            brand = watch['brand']
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        top_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"🏭 Top brands by watch count:")
        for brand, count in top_brands:
            print(f"   - {brand}: {count} watches")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Generating embeddings for final scrape data...")
    success = generate_final_embeddings()
    
    if success:
        print("\n🎉 Embedding generation completed!")
        print("You can now update the backend to use the new embeddings.")
    else:
        print("\n❌ Embedding generation failed.")
        print("Please check the error messages above.") 