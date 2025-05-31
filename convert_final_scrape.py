#!/usr/bin/env python3
"""
Convert final_scrape.csv to JSON format for embedding generation
"""
import pandas as pd
import json
import os
from urllib.parse import urlparse

def clean_text_field(text):
    """Clean text field by removing extra whitespace and handling None values."""
    if pd.isna(text) or text is None:
        return ""
    return str(text).strip()

def clean_price(price_text):
    """Convert price to a standardized format."""
    if pd.isna(price_text) or not str(price_text).strip() or str(price_text).strip() == '-':
        return "Contact for price"
    
    try:
        # Remove any currency symbols and commas, extract number
        price_str = str(price_text).replace(',', '').replace('$', '').replace('€', '').strip()
        if price_str and price_str.replace('.', '').isdigit():
            price_num = float(price_str)
            return f"${int(price_num)}"
    except:
        pass
    
    return f"${price_text}" if not str(price_text).startswith('$') else str(price_text)

def create_watch_description(watch_data):
    """Create a comprehensive description for embedding generation."""
    specs = watch_data.get('specs', {})
    
    # Basic info
    brand = specs.get('brand', '')
    model = specs.get('model', '')
    serie = specs.get('serie', '')
    watch_type = specs.get('watch_type', '')
    
    # Physical characteristics
    case_material = specs.get('case_material', '')
    diameter_mm = specs.get('diameter_mm', '')
    dial_color = specs.get('dial_color', '')
    movement = specs.get('movement', '')
    
    # Complications
    complications = []
    if specs.get('complication_chronograph') == 'Yes':
        complications.append('chronograph')
    if specs.get('complication_date') == 'Yes':
        complications.append('date')
    if specs.get('complication_gmt') == 'Yes':
        complications.append('GMT')
    if specs.get('complication_dual_time') == 'Yes':
        complications.append('dual time')
    
    # Build description
    description_parts = []
    
    if brand:
        description_parts.append(f"Brand: {brand}")
    if model:
        description_parts.append(f"Model: {model}")
    if serie and serie != '-':
        description_parts.append(f"Series: {serie}")
    if watch_type:
        description_parts.append(f"Type: {watch_type}")
    if case_material:
        description_parts.append(f"Case: {case_material}")
    if diameter_mm and str(diameter_mm) != '-':
        description_parts.append(f"Size: {diameter_mm}mm")
    if dial_color:
        description_parts.append(f"Dial: {dial_color}")
    if movement:
        description_parts.append(f"Movement: {movement}")
    if complications:
        description_parts.append(f"Features: {', '.join(complications)}")
    
    description = ". ".join(description_parts)
    
    # Add availability and pricing context
    availability = specs.get('availability', '')
    if availability:
        description += f". Availability: {availability}"
    
    return description

def convert_final_scrape():
    """Convert final_scrape.csv to JSON format for embedding generation."""
    
    input_file = "production_scrape_20250531_231634/data/final_scrape.csv"
    output_file = "data/raw/watch_data_final_scrape.json"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found")
        return False
    
    print(f"Loading watch data from {input_file}...")
    
    # Read CSV
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} watches from CSV")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    converted_watches = []
    seen_urls = set()  # To avoid duplicates
    
    for i, row in df.iterrows():
        try:
            # Extract basic info
            brand = clean_text_field(row.get('brand', ''))
            model = clean_text_field(row.get('model', ''))
            url = clean_text_field(row.get('url', ''))
            main_image = clean_text_field(row.get('main_image', ''))
            
            # Skip if missing essential info
            if not brand or not model or not url:
                print(f"Skipping watch {i}: missing essential info")
                continue
            
            # Skip duplicates based on URL
            if url in seen_urls:
                print(f"Skipping watch {i}: duplicate URL")
                continue
            seen_urls.add(url)
            
            # Clean price
            price_usd = clean_price(row.get('price_usd', ''))
            
            # Create specs dictionary with all available fields
            specs = {}
            for col in df.columns:
                if col not in ['main_image']:  # main_image is handled separately
                    specs[col] = clean_text_field(row.get(col, ''))
            
            # Create the watch entry
            watch_entry = {
                'brand': brand,
                'model': model,
                'price': price_usd,
                'url': url,
                'image_url': main_image,
                'specs': specs,
                'description': create_watch_description({'specs': specs}),
                'index': len(converted_watches)
            }
            
            converted_watches.append(watch_entry)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(df)} watches...")
        
        except Exception as e:
            print(f"Error processing watch {i}: {e}")
            continue
    
    print(f"Successfully converted {len(converted_watches)} watches")
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_watches, f, indent=2, ensure_ascii=False)
    
    print(f"Saved converted data to {output_file}")
    
    # Show statistics
    brands = set(watch['brand'] for watch in converted_watches)
    with_images = sum(1 for watch in converted_watches if watch['image_url'])
    with_series = sum(1 for watch in converted_watches 
                     if watch['specs'].get('serie', '') and watch['specs']['serie'] != '-')
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   🏷️  Total watches: {len(converted_watches)}")
    print(f"   🏭 Unique brands: {len(brands)}")
    print(f"   🖼️  Watches with images: {with_images}")
    print(f"   📚 Watches with series info: {with_series}")
    
    # Show series statistics
    series_counts = {}
    for watch in converted_watches:
        series = watch['specs'].get('serie', '')
        if series and series != '-':
            series_counts[series] = series_counts.get(series, 0) + 1
    
    print(f"   🔗 Unique series: {len(series_counts)}")
    
    # Show top series
    if series_counts:
        top_series = sorted(series_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"   📈 Top series:")
        for series, count in top_series:
            print(f"      - {series}: {count} watches")
    
    return True

if __name__ == "__main__":
    print("🔄 Converting final_scrape.csv to JSON format...")
    success = convert_final_scrape()
    
    if success:
        print("\n🎉 Data conversion completed!")
        print("The watch data is ready for embedding generation.")
        print("Run generate_final_embeddings.py to generate embeddings.")
    else:
        print("\n❌ Data conversion failed.")
        print("Please check the error messages above.") 