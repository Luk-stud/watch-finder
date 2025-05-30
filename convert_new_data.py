#!/usr/bin/env python3
import json
import re
import os
from datetime import datetime

def clean_text_field(text):
    """Clean up messy text fields from the scraping."""
    if not text or text == '-':
        return ''
    
    # Remove HTML-like patterns
    text = re.sub(r'capitalize lowercase font-bold[^>]*>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove common scraping artifacts
    text = re.sub(r'\bBrand\b', '', text)
    text = re.sub(r'\bModele\s*:\s*-\s*Serie\s*:\s*', '', text)
    text = re.sub(r'\bColor\s*:\s*', '', text)
    
    # Clean up commas and dashes at start/end
    text = text.strip(' ,-')
    
    return text if text and text != '-' else ''

def extract_specs_info(specs):
    """Extract useful information from specifications."""
    if not isinstance(specs, dict):
        return {}
    
    cleaned_specs = {}
    
    # Water resistance
    water_res = specs.get('water_resistance', '')
    if water_res and water_res.isdigit():
        cleaned_specs['water_resistance'] = f"{water_res}m"
    
    # Movement
    movement = clean_text_field(specs.get('movement', ''))
    if movement and 'waterproofing' not in movement.lower():
        cleaned_specs['movement'] = movement
    
    # Case material
    case_material = clean_text_field(specs.get('case_material', ''))
    if case_material and 'diameter' not in case_material.lower():
        cleaned_specs['case_material'] = case_material
    
    # Case diameter
    diameter = specs.get('diameter', '') or specs.get('case_diameter', '')
    if diameter:
        diameter = clean_text_field(str(diameter))
        if diameter and any(char.isdigit() for char in diameter):
            cleaned_specs['case_size'] = diameter
    
    return cleaned_specs

def convert_new_data_format():
    """Convert all extropian chunk data to our expected format."""
    
    # Find all chunk files
    chunk_files = []
    for i in range(1, 10):  # Check chunks 01-09
        chunk_path = f"watch_data/extropian_chunk_{i:02d}.json"
        if os.path.exists(chunk_path):
            chunk_files.append(chunk_path)
    
    if not chunk_files:
        print("Error: No extropian chunk files found")
        return False
    
    print(f"Found {len(chunk_files)} chunk files: {[os.path.basename(f) for f in chunk_files]}")
    
    # Convert all chunks
    all_converted_watches = []
    
    for chunk_file in chunk_files:
        print(f"\nProcessing {os.path.basename(chunk_file)}...")
        
        with open(chunk_file, 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)
        
        print(f"Loaded {len(chunk_data)} watches from {os.path.basename(chunk_file)}")
        
        # Convert this chunk
        chunk_converted = []
        
        for i, watch in enumerate(chunk_data):
            try:
                # Extract basic info
                basic_info = watch.get('basic_info', {})
                pricing = watch.get('pricing', {})
                specifications = watch.get('specifications', {})
                images = watch.get('images', [])
                
                # Clean up brand name
                brand = clean_text_field(basic_info.get('brand', ''))
                if not brand:
                    continue  # Skip watches without brand
                
                # Clean up model name
                model = clean_text_field(basic_info.get('model', ''))
                if not model:
                    model = clean_text_field(basic_info.get('series', ''))
                
                # Get color info
                color = clean_text_field(basic_info.get('color', ''))
                
                # Use EUR price only
                price_eur = pricing.get('price_eur')
                price_str = f"€{price_eur}" if price_eur else "Contact for price"
                
                # Get first image
                image_url = images[0] if images else ""
                
                # Get specifications
                specs = extract_specs_info(specifications)
                
                # Build description
                description_parts = []
                if color:
                    description_parts.append(f"Color: {color}")
                description_parts.append(f"Brand: {brand}")
                if model:
                    description_parts.append(f"Model: {model}")
                
                # Add specs to description
                for spec_name, spec_value in specs.items():
                    if spec_name == 'water_resistance':
                        description_parts.append(f"Water resistance: {spec_value}")
                    elif spec_name == 'movement':
                        description_parts.append(f"Movement: {spec_value}")
                    elif spec_name == 'case_material':
                        description_parts.append(f"Case: {spec_value}")
                    elif spec_name == 'case_size':
                        description_parts.append(f"Size: {spec_value}")
                
                # Create converted watch entry
                converted_watch = {
                    'brand': brand,
                    'model_name': model,
                    'price': price_str,
                    'image_url': image_url,
                    'product_url': watch.get('url', ''),
                    'description': '. '.join(description_parts),
                    'source': f'extropian_v2_{os.path.basename(chunk_file)}',
                    'timestamp': watch.get('scraped_at', datetime.now().isoformat())
                }
                
                # Add individual spec fields for easy access
                converted_watch.update(specs)
                
                chunk_converted.append(converted_watch)
                
            except Exception as e:
                print(f"Error converting watch {i} in {os.path.basename(chunk_file)}: {e}")
                continue
        
        print(f"Successfully converted {len(chunk_converted)} watches from {os.path.basename(chunk_file)}")
        all_converted_watches.extend(chunk_converted)
        
        if len(chunk_converted) > 0:
            print(f"Running total: {len(all_converted_watches)} watches")
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total watches converted: {len(all_converted_watches)}")
    
    # Save converted data
    output_path = "data/raw/watch_data_v2_complete.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_converted_watches, f, indent=2, ensure_ascii=False)
    
    print(f"Saved complete dataset to {output_path}")
    
    # Show statistics
    brands = set(watch['brand'] for watch in all_converted_watches)
    with_specs = sum(1 for watch in all_converted_watches if any(spec in watch for spec in ['water_resistance', 'case_size', 'movement', 'case_material']))
    with_eur_prices = sum(1 for watch in all_converted_watches if watch['price'].startswith('€'))
    
    print(f"\n=== DATASET STATISTICS ===")
    print(f"Total watches: {len(all_converted_watches)}")
    print(f"Unique brands: {len(brands)}")
    print(f"Watches with specifications: {with_specs}")
    print(f"Watches with EUR prices: {with_eur_prices}")
    print(f"Chunks processed: {len(chunk_files)}")
    
    # Show brand distribution
    brand_counts = {}
    for watch in all_converted_watches:
        brand = watch['brand']
        brand_counts[brand] = brand_counts.get(brand, 0) + 1
    
    print(f"\nTop 10 brands:")
    for brand, count in sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {brand}: {count} watches")
    
    # Show some examples
    print("\nExample converted watches:")
    for i, watch in enumerate(all_converted_watches[:3]):
        print(f"\nWatch {i + 1}:")
        print(f"  Brand: {watch['brand']}")
        print(f"  Model: {watch['model_name']}")
        print(f"  Price: {watch['price']}")
        print(f"  Source: {watch['source']}")
        if 'water_resistance' in watch:
            print(f"  Water Resistance: {watch['water_resistance']}")
        if 'case_size' in watch:
            print(f"  Case Size: {watch['case_size']}")
    
    return True

if __name__ == "__main__":
    success = convert_new_data_format()
    if success:
        print("\n✅ Data conversion completed successfully!")
        print("Next step: Run generate_clip_embeddings.py with the new data")
    else:
        print("❌ Data conversion failed") 