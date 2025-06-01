#!/usr/bin/env python3
"""
Generate embeddings from the final_scrape.csv data using local images
Optimized for Mac with Apple Silicon and MCP
"""
import sys
import os
import pandas as pd
import json
import pickle
import platform
from PIL import Image
import torch

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from models.watch_embedder import WatchEmbedder

def clean_value(value):
    """Convert pandas NaN values to None for proper JSON serialization"""
    if pd.isna(value):
        return None
    return value

def get_device():
    """Get the best available device for Mac"""
    if platform.system() == 'Darwin':  # macOS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("🍎 Using Apple Silicon MPS (Metal Performance Shaders)")
            return torch.device('mps')
        else:
            print("🍎 Using CPU on Mac (MPS not available)")
            return torch.device('cpu')
    elif torch.cuda.is_available():
        print("🔥 Using CUDA GPU")
        return torch.device('cuda')
    else:
        print("💻 Using CPU")
        return torch.device('cpu')

def get_local_image_path(brand, model):
    """Generate local image path based on brand and model"""
    # Clean brand name (remove spaces, special chars)
    brand_clean = brand.replace(' ', '').replace('/', '_').replace('(', '').replace(')', '').replace('-', '_')
    
    # Clean model name  
    model_clean = model.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('-', '_')
    
    # Local image path
    image_filename = f"{brand_clean}_{model_clean}_main.jpg"
    
    # Check both production and frontend locations
    production_path = f"production_scrape_20250531_231634/images/{image_filename}"
    frontend_path = f"frontend-nextjs/public/images/{image_filename}"
    
    # Prefer frontend path if it exists (for web serving)
    if os.path.exists(frontend_path):
        return frontend_path, image_filename
    elif os.path.exists(production_path):
        return production_path, image_filename
    else:
        return None, image_filename

def load_final_scrape_data():
    """Load and process the final_scrape.csv data"""
    csv_path = "production_scrape_20250531_231634/data/final_scrape.csv"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Final scrape CSV not found: {csv_path}")
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Convert to the format expected by WatchEmbedder
    watch_data = []
    
    for _, row in df.iterrows():
        # Get local image path
        local_image_path, image_filename = get_local_image_path(row['brand'], row['model'])
        
        # Check if local image exists
        if local_image_path:
            # For local images, keep CDN URL as fallback and use local file path
            image_url = row['main_image']  # Keep CDN URL for fallback
            local_file_path = local_image_path  # Actual file path for local access
            has_local_image = True
        else:
            # Fallback to CDN URL only
            image_url = row['main_image']
            local_file_path = None
            has_local_image = False
        
        # Build comprehensive metadata with all available specs
        metadata = {
            'brand': row['brand'],
            'model': row['model'],
            'serie': row.get('serie', ''),  # Add series information
            'reference': row.get('reference', ''),
            'image_url': image_url,
            'local_image_path': local_file_path,
            'specs': {
                # Basic info
                'brand': row['brand'],
                'model': row['model'],
                'serie': row.get('serie', ''),
                'reference': row.get('reference', ''),
                'watch_type': row.get('watch_type', ''),
                'second_watch_type': row.get('second_watch_type', ''),
                'url': row.get('url', ''),
                
                # Pricing
                'price_usd': clean_value(row.get('price_usd', '')),
                'msrp_eur': clean_value(row.get('msrp_eur', '')),
                'launch_price_eur': clean_value(row.get('launch_price_eur', '')),
                'availability': row.get('availability', ''),
                'quantity_produced': row.get('quantity_produced', ''),
                'limited_edition_name': row.get('limited_edition_name', ''),
                'launch_date': row.get('launch_date', ''),
                
                # Case specifications
                'case_material': row.get('case_material', ''),
                'bottom_case_material': row.get('bottom_case_material', ''),
                'diameter_mm': clean_value(row.get('diameter_mm', '')),
                'thickness_with_crystal_mm': clean_value(row.get('thickness_with_crystal_mm', '')),
                'thickness_without_crystal_mm': clean_value(row.get('thickness_without_crystal_mm', '')),
                'lug_to_lug_mm': clean_value(row.get('lug_to_lug_mm', '')),
                'case_shape': row.get('case_shape', ''),
                'case_coating_pvd_dlc': row.get('case_coating_pvd_dlc', ''),
                'case_finishing': row.get('case_finishing', ''),
                'drill_lugs_on_case': row.get('drill_lugs_on_case', ''),
                
                # Dial specifications
                'dial_color': row.get('dial_color', ''),
                'dial_material': row.get('dial_material', ''),
                'dial_type': row.get('dial_type', ''),
                'dial_pattern': row.get('dial_pattern', ''),
                'indices_type': row.get('indices_type', ''),
                'hands_types': row.get('hands_types', ''),
                'full_lume': row.get('full_lume', ''),
                'lume_1': row.get('lume_1', ''),
                'lume_2': row.get('lume_2', ''),
                'color_match_date_wheel': row.get('color_match_date_wheel', ''),
                
                # Movement specifications
                'movement': row.get('movement', ''),
                'winding': row.get('winding', ''),
                'power_reserve_hour': clean_value(row.get('power_reserve_hour', '')),
                'mph': row.get('mph', ''),
                'hacking': row.get('hacking', ''),
                'manual_winding': row.get('manual_winding', ''),
                
                # Complications
                'complication_chronograph': row.get('complication_chronograph', ''),
                'complication_date': row.get('complication_date', ''),
                'complication_dual_time': row.get('complication_dual_time', ''),
                'complication_flying_tourbillon': row.get('complication_flying_tourbillon', ''),
                'complication_gmt': row.get('complication_gmt', ''),
                'complication_jump_hour': row.get('complication_jump_hour', ''),
                'complication_power_reserve': row.get('complication_power_reserve', ''),
                'complication_small_second': row.get('complication_small_second', ''),
                'complication_sub_24_hour': row.get('complication_sub_24_hour', ''),
                'complication_sub_second': row.get('complication_sub_second', ''),
                'complication_2nd_bezel_timezone': row.get('complication_2nd_bezel_timezone', ''),
                'complication_moonphase': row.get('complication_moonphase', ''),
                'complication_world_time_zone': row.get('complication_world_time_zone', ''),
                'complication_alarm': row.get('complication_alarm', ''),
                'complication_chronoscope': row.get('complication_chronoscope', ''),
                
                # Crystal and bezel
                'crystal_material': row.get('crystal_material', ''),
                'crystal_type_shape': row.get('crystal_type_shape', ''),
                'inner_coating': row.get('inner_coating', ''),
                'outside_coating': row.get('outside_coating', ''),
                'bezel_insert_material': row.get('bezel_insert_material', ''),
                'external_bezel_material': row.get('external_bezel_material', ''),
                'bezel_type': row.get('bezel_type', ''),
                'uni_bi_direction_bezel': row.get('uni_bi_direction_bezel', ''),
                'number_of_clicks': row.get('number_of_clicks', ''),
                'internal_bezel': row.get('internal_bezel', ''),
                
                # Crown and strap/bracelet
                'main_crown_type': row.get('main_crown_type', ''),
                'other_crowns_function': row.get('other_crowns_function', ''),
                'strap_1_material': row.get('strap_1_material', ''),
                'strap_2_material': row.get('strap_2_material', ''),
                'strap_3_material': row.get('strap_3_material', ''),
                'width_mm': clean_value(row.get('width_mm', '')),
                'bracelet_tapper_to_clasp_mm': clean_value(row.get('bracelet_tapper_to_clasp_mm', '')),
                'bracelet_type': row.get('bracelet_type', ''),
                'bracelet_links_type': row.get('bracelet_links_type', ''),
                'bracelet_finishing': row.get('bracelet_finishing', ''),
                'strap_bracelet_attachment_system': row.get('strap_bracelet_attachment_system', ''),
                'clasp_type': row.get('clasp_type', ''),
                'clasp_material': row.get('clasp_material', ''),
                
                # Other specifications
                'waterproofing_meters': clean_value(row.get('waterproofing_meters', '')),
                'warranty_year': clean_value(row.get('warranty_year', '')),
                'brand_country': row.get('brand_country', ''),
                'made_in': row.get('made_in', ''),
                'assembled_in': row.get('assembled_in', ''),
                'country': row.get('country', ''),
                'waterproofing': row.get('waterproofing', ''),
                'specific_info_from_brand': row.get('specific_info_from_brand', ''),
                'brand_website': row.get('brand_website', ''),
            }
        }
        
        watch = {
            'brand': row['brand'],
            'model': row['model'],  # Use 'model' not 'model_name' for consistency
            'model_name': row['model'],  # Also include model_name for compatibility
            'price': row.get('price', 'Contact for price'),
            'image_url': image_url,  # This will be local path if available
            'local_image_path': local_file_path,  # Full file system path
            'main_image': row['main_image'],  # Keep original CDN URL as backup
            'product_url': row.get('product_url', ''),
            'description': f"Brand: {row['brand']}. Model: {row['model']}",
            'source': 'final_scrape',
            'local_image': has_local_image,
            'specs': metadata['specs']
        }
        
        watch_data.append(watch)
    
    print(f"Loaded {len(watch_data)} watches")
    
    # Count local vs CDN images
    local_count = sum(1 for w in watch_data if w['local_image'])
    cdn_count = len(watch_data) - local_count
    
    print(f"🖼️  Local images: {local_count}")
    print(f"🌐 CDN images: {cdn_count}")
    
    return watch_data

def generate_final_embeddings_with_local():
    """Generate embeddings for the final scrape dataset using local images when available."""
    
    # Get optimal device for Mac
    device = get_device()
    
    # Use custom CLIP model if available
    custom_model_path = "best_clip_model_description_model.pt"
    
    if os.path.exists(custom_model_path):
        print(f"Using custom CLIP model: {custom_model_path}")
        embedder = WatchEmbedder(model_path=custom_model_path, device=device)
    else:
        print("Using default CLIP model")
        embedder = WatchEmbedder(device=device)
    
    # Mac-specific optimizations
    if platform.system() == 'Darwin':
        print("🍎 Applying Mac optimizations...")
        # Enable memory-efficient attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(False)  # Disable for Mac
        
        # Set optimal thread count for Mac
        if torch.get_num_threads() > 8:
            torch.set_num_threads(8)  # Optimal for most Mac configurations
    
    # Load and process final scrape data
    print("📊 Loading final scrape data...")
    watch_data = load_final_scrape_data()
    
    # Save the processed data as JSON for the backend
    metadata_json_path = "backend/data/final_scrape_processed.json"
    os.makedirs("backend/data", exist_ok=True)
    
    with open(metadata_json_path, 'w') as f:
        json.dump(watch_data, f, indent=2)
    
    print(f"💾 Saved processed metadata to: {metadata_json_path}")
    
    # Generate embeddings using the processed data
    print("🚀 Generating embeddings...")
    
    try:
        # Set the watch data directly
        embedder.watch_data = watch_data
        
        # Generate embeddings for all watches with progress tracking
        total_watches = len(watch_data)
        print(f"Processing {total_watches} watches...")
        
        # Save the data temporarily for the embedder
        temp_data_path = "backend/data/temp_watch_data.json"
        with open(temp_data_path, 'w') as f:
            json.dump(watch_data, f)
        
        embeddings = embedder.generate_embeddings(temp_data_path)
        
        # Clean up temp file
        if os.path.exists(temp_data_path):
            os.remove(temp_data_path)
        
        # Save embeddings in the correct location for the backend
        embeddings_path = "backend/data/watch_embeddings.pkl"
        metadata_path = "backend/data/watch_metadata.pkl"
        
        print("💾 Saving embeddings and metadata...")
        
        # Save embeddings
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(watch_data, f)
        
        print(f"\n🎉 Final scrape embeddings generated successfully!")
        print(f"📊 Dataset size: {len(watch_data)} watches")
        print(f"📐 Embedding dimension: {embeddings.shape[1]}")
        print(f"💾 Embeddings saved to: {embeddings_path}")
        print(f"💾 Metadata saved to: {metadata_path}")
        
        # Show file sizes
        if os.path.exists(embeddings_path):
            size_mb = os.path.getsize(embeddings_path) / (1024 * 1024)
            print(f"📁 Embeddings file size: {size_mb:.1f} MB")
        
        if os.path.exists(metadata_path):
            size_mb = os.path.getsize(metadata_path) / (1024 * 1024)
            print(f"📁 Metadata file size: {size_mb:.1f} MB")
        
        # Show dataset statistics
        brands = set(watch['brand'] for watch in watch_data)
        local_images = sum(1 for watch in watch_data if watch['local_image'])
        
        print(f"\n📈 Dataset Statistics:")
        print(f"🏷️  Unique brands: {len(brands)}")
        print(f"🖼️  Local images used: {local_images}")
        print(f"🌐 CDN images used: {len(watch_data) - local_images}")
        print(f"📱 Device used: {device}")
        
        # Show brand distribution
        brand_counts = {}
        for watch in watch_data:
            brand = watch['brand']
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        top_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n🏭 Top brands by watch count:")
        for brand, count in top_brands:
            print(f"   - {brand}: {count} watches")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Generating embeddings for final scrape data with local images...")
    print("🍎 Optimized for Mac with Apple Silicon and MCP")
    
    success = generate_final_embeddings_with_local()
    
    if success:
        print("\n🎉 Embedding generation completed!")
        print("✅ The backend data files have been updated with final_scrape data and local images.")
        print("🚀 Ready for deployment to Railway!")
    else:
        print("\n❌ Embedding generation failed.")
        print("Please check the error messages above.") 