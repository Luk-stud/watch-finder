#!/usr/bin/env python3
import json
import os
from datetime import datetime

def clean_text_field(text):
    """Clean up text fields."""
    if not text or text == '-':
        return ''
    return str(text).strip()

def format_price(price_usd, msrp_eur):
    """Format price information."""
    if price_usd and price_usd != '-':
        return f"${price_usd}"
    elif msrp_eur and msrp_eur != '-':
        return f"€{msrp_eur}"
    else:
        return "Contact for price"

def create_description(watch_data):
    """Create a rich description from the new detailed specs."""
    specs = watch_data.get('specs', {})
    if not specs:
        return ""
    
    description_parts = []
    
    # Basic info
    brand = specs.get('brand', '')
    model = specs.get('model', '')
    if brand:
        description_parts.append(f"Brand: {brand}")
    if model:
        description_parts.append(f"Model: {model}")
    
    # Watch type
    watch_type = specs.get('watch_type', '')
    if watch_type and watch_type != '-':
        description_parts.append(f"Type: {watch_type}")
    
    # Case specifications
    case_material = specs.get('case_material', '')
    diameter_mm = specs.get('diameter_mm', '')
    if case_material and case_material != '-':
        description_parts.append(f"Case: {case_material}")
    if diameter_mm and diameter_mm != '-':
        description_parts.append(f"Size: {diameter_mm}mm")
    
    # Movement
    movement = specs.get('movement', '')
    if movement and movement != '-':
        description_parts.append(f"Movement: {movement}")
    
    # Water resistance
    waterproofing = specs.get('waterproofing_meters', '')
    if waterproofing and waterproofing != '-':
        description_parts.append(f"Water resistance: {waterproofing}m")
    
    # Dial specifications
    dial_color = specs.get('dial_color', '')
    if dial_color and dial_color != '-':
        description_parts.append(f"Dial: {dial_color}")
    
    # Crystal
    crystal_material = specs.get('crystal_material', '')
    if crystal_material and crystal_material != '-':
        description_parts.append(f"Crystal: {crystal_material}")
    
    # Complications
    complications = []
    if specs.get('complication_date') and specs.get('complication_date') != 'No':
        complications.append('Date')
    if specs.get('complication_chronograph') and specs.get('complication_chronograph') != 'No':
        complications.append('Chronograph')
    if specs.get('complication_gmt') and specs.get('complication_gmt') != 'No':
        complications.append('GMT')
    if specs.get('complication_moonphase') and specs.get('complication_moonphase') != 'No':
        complications.append('Moonphase')
    
    if complications:
        description_parts.append(f"Features: {', '.join(complications)}")
    
    return '. '.join(description_parts)

def convert_progress_data():
    """Convert the progress_20250531_232532.json file to embedder format."""
    
    input_file = "watch_data/progress_20250531_232532.json"
    output_file = "data/raw/watch_data_v3_detailed.json"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found")
        return False
    
    print(f"Loading detailed watch data from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Processing {len(raw_data)} detailed watches...")
    
    converted_watches = []
    seen_urls = set()  # To avoid duplicates
    
    for i, watch in enumerate(raw_data):
        try:
            specs = watch.get('specs', {})
            main_image = watch.get('main_image', '')
            brand_website = watch.get('brand_website', '')
            
            # Skip if no specs or essential data
            if not specs:
                print(f"Skipping watch {i}: no specs data")
                continue
            
            brand = clean_text_field(specs.get('brand', ''))
            model = clean_text_field(specs.get('model', ''))
            url = clean_text_field(specs.get('url', ''))
            
            # Skip if missing essential info
            if not brand or not model or not url:
                print(f"Skipping watch {i}: missing essential info (brand: {brand}, model: {model}, url: {url[:50]}...)")
                continue
            
            # Skip duplicates based on URL
            if url in seen_urls:
                print(f"Skipping watch {i}: duplicate URL")
                continue
            seen_urls.add(url)
            
            # Create converted watch entry
            converted_watch = {
                'brand': brand,
                'model_name': model,
                'price': format_price(specs.get('price_usd'), specs.get('msrp_eur')),
                'image_url': main_image,
                'product_url': url,
                'description': create_description(watch),
                'source': 'extropian_v3_detailed',
                'timestamp': datetime.now().isoformat(),
                
                # Legacy compatibility fields
                'case_diameter': specs.get('diameter_mm', ''),
                'case_material': specs.get('case_material', ''),
                'movement': specs.get('movement', ''),
                'water_resistance': f"{specs.get('waterproofing_meters', '')}m" if specs.get('waterproofing_meters', '') and specs.get('waterproofing_meters', '') != '-' else '',
                
                # New detailed specs structure
                'specs': specs,
                'main_image': main_image,
                'brand_website': brand_website
            }
            
            converted_watches.append(converted_watch)
            
        except Exception as e:
            print(f"Error converting watch {i}: {e}")
            continue
    
    print(f"Successfully converted {len(converted_watches)} watches")
    
    # Save converted data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_watches, f, indent=2, ensure_ascii=False)
    
    print(f"Saved detailed dataset to {output_file}")
    
    # Show statistics
    brands = set(watch['brand'] for watch in converted_watches)
    with_prices = sum(1 for watch in converted_watches if not watch['price'].startswith('Contact'))
    with_images = sum(1 for watch in converted_watches if watch['image_url'])
    with_movement = sum(1 for watch in converted_watches if watch['movement'])
    with_complications = sum(1 for watch in converted_watches if any(
        watch['specs'].get(f'complication_{comp}', 'No') != 'No' 
        for comp in ['date', 'chronograph', 'gmt', 'moonphase']
    ))
    
    print(f"\n=== DATASET STATISTICS ===")
    print(f"Total watches: {len(converted_watches)}")
    print(f"Unique brands: {len(brands)}")
    print(f"Watches with prices: {with_prices}")
    print(f"Watches with images: {with_images}")
    print(f"Watches with movement info: {with_movement}")
    print(f"Watches with complications: {with_complications}")
    
    # Show brand distribution
    brand_counts = {}
    for watch in converted_watches:
        brand = watch['brand']
        brand_counts[brand] = brand_counts.get(brand, 0) + 1
    
    print(f"\nTop 10 brands:")
    for brand, count in sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {brand}: {count} watches")
    
    # Show series distribution
    series_counts = {}
    for watch in converted_watches:
        series = watch['specs'].get('serie', 'Unknown')
        if series and series != '-':
            series_counts[series] = series_counts.get(series, 0) + 1
    
    print(f"\nTop 10 series:")
    for series, count in sorted(series_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {series}: {count} watches")
    
    return True

if __name__ == "__main__":
    print("🔄 Converting detailed progress data...")
    success = convert_progress_data()
    
    if success:
        print("\n🎉 Data conversion completed!")
        print("The detailed watch data is ready for embedding generation.")
    else:
        print("\n❌ Data conversion failed.")
        print("Please check the error messages above.") 