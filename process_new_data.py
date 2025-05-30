import json
import os
import re
from urllib.parse import urlparse, unquote

def extract_size_from_url(url):
    """Extract size information from the URL."""
    # Look for size patterns like 39mm, 42mm, etc.
    size_match = re.search(r'(\d+)mm', url)
    return f"{size_match.group(1)}mm" if size_match else "Unknown"

def extract_color_from_url(url):
    """Extract color information from the URL."""
    # Split URL and look for color keywords
    url_parts = url.lower().split('-')
    colors = ['black', 'white', 'blue', 'red', 'green', 'brown', 'gold', 'silver', 
              'bronze', 'grey', 'gray', 'yellow', 'orange', 'purple', 'pink',
              'navy', 'slate', 'crimson', 'teal', 'olive', 'aqua', 'cream']
    
    found_colors = []
    for part in url_parts:
        for color in colors:
            if color in part:
                found_colors.append(color.title())
    
    return ', '.join(found_colors) if found_colors else "Unknown"

def clean_model_name(title, brand):
    """Clean and extract model name from title."""
    # Remove brand from title if it's there
    title_clean = title.replace(f"{brand} / ", "").strip()
    return title_clean

def create_description(watch):
    """Create a rich description from available data."""
    description_parts = []
    
    # Extract size from URL
    size = extract_size_from_url(watch['url'])
    if size != "Unknown":
        description_parts.append(f"Case size: {size}")
    
    # Extract color information
    color = extract_color_from_url(watch['url'])
    if color != "Unknown":
        description_parts.append(f"Color: {color}")
    
    # Add brand and model
    description_parts.append(f"Brand: {watch['brand']}")
    description_parts.append(f"Model: {watch['model']}")
    
    # Add any distinctive features from URL
    url_parts = watch['url'].split('-')
    features = []
    feature_keywords = ['automatic', 'manual', 'quartz', 'chronograph', 'gmt', 'date', 
                       'skeleton', 'diving', 'diver', 'sport', 'dress', 'vintage',
                       'limited', 'edition', 'steel', 'titanium', 'ceramic']
    
    for part in url_parts:
        part_lower = part.lower()
        for keyword in feature_keywords:
            if keyword in part_lower and keyword not in [f.lower() for f in features]:
                features.append(keyword.title())
    
    if features:
        description_parts.append(f"Features: {', '.join(features)}")
    
    return ". ".join(description_parts)

def process_watch_data(input_file, output_file):
    """Process the new watch data format and convert to the expected format."""
    print(f"Loading data from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Processing {len(raw_data)} watches...")
    
    processed_data = []
    seen_combinations = set()  # To avoid duplicates
    
    for i, watch in enumerate(raw_data):
        # Create unique identifier to avoid duplicates
        unique_id = f"{watch['brand']}_{watch['model']}_{watch['url']}"
        
        if unique_id in seen_combinations:
            continue
        seen_combinations.add(unique_id)
        
        # Create processed watch entry
        processed_watch = {
            "brand": watch['brand'],
            "model_name": clean_model_name(watch['title'], watch['brand']),
            "price": "Contact for price",  # No price in original data
            "image_url": watch['main_image'],
            "product_url": watch['url'],
            "description": create_description(watch),
            "index": len(processed_data),
            "source": "extropian",
            "timestamp": watch['timestamp']
        }
        
        processed_data.append(processed_watch)
    
    print(f"Processed {len(processed_data)} unique watches")
    
    # Save processed data
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    # Print some stats
    brands = {}
    for watch in processed_data:
        brand = watch['brand']
        brands[brand] = brands.get(brand, 0) + 1
    
    print(f"\nDataset Statistics:")
    print(f"Total watches: {len(processed_data)}")
    print(f"Unique brands: {len(brands)}")
    print(f"\nTop 10 brands:")
    for brand, count in sorted(brands.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {brand}: {count} watches")
    
    return processed_data

if __name__ == '__main__':
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths
    input_file = os.path.join(project_root, 'data', 'raw', 'watches_progress_1.json')
    output_file = os.path.join(project_root, 'data', 'processed', 'watches_processed.json')
    
    # Create processed directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process the data
    process_watch_data(input_file, output_file) 