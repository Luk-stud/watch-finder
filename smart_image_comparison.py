#!/usr/bin/env python3
"""
Smart Watch Image Comparison Tool

Creates intelligent comparisons of watch images based on brands, types, or similarity.
Pure visual comparison without text - just images side by side.
"""

import os
import json
import random
from PIL import Image
import argparse
from typing import List, Tuple, Dict, Optional

class SmartImageComparator:
    def __init__(self, output_size: Tuple[int, int] = (1200, 600)):
        self.output_size = output_size
        self.watch_data = {}
        
    def load_watch_data(self, data_path: str = "production_scrape_20250601_175426/data/final_scrape.json"):
        """Load watch metadata for intelligent filtering."""
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
                
            # Convert to dict for easier lookup
            self.watch_data = {}
            for watch in data:
                specs = watch.get('specs', {})
                brand = specs.get('brand', '')
                model = specs.get('model', '')
                
                if brand and model:
                    # Create image path
                    image_name = f"{brand}_{model}_main.jpg".replace(' ', '_').replace('-', '_')
                    image_path = f"production_scrape_20250601_175426/images/{image_name}"
                    
                    if os.path.exists(image_path):
                        watch_id = f"{brand}_{model}"
                        self.watch_data[watch_id] = {
                            'brand': brand,
                            'model': model,
                            'watch_type': specs.get('watch_type', ''),
                            'image_path': image_path,
                            'specs': specs
                        }
            
            print(f"✅ Loaded data for {len(self.watch_data)} watches")
            
        except Exception as e:
            print(f"Could not load watch data: {e}")
            print("Using directory-based approach instead")
    
    def get_watches_by_brand(self, brand: str, limit: int = 10) -> List[str]:
        """Get watch image paths for a specific brand."""
        matches = []
        brand_lower = brand.lower()
        
        for watch_id, data in self.watch_data.items():
            if brand_lower in data['brand'].lower():
                matches.append(data['image_path'])
                if len(matches) >= limit:
                    break
        
        return matches
    
    def get_watches_by_type(self, watch_type: str, limit: int = 10) -> List[str]:
        """Get watch image paths for a specific type."""
        matches = []
        type_lower = watch_type.lower()
        
        for watch_id, data in self.watch_data.items():
            if type_lower in data['watch_type'].lower():
                matches.append(data['image_path'])
                if len(matches) >= limit:
                    break
        
        return matches
    
    def get_diverse_sample(self, count: int = 4) -> List[str]:
        """Get a diverse sample of different brands."""
        if not self.watch_data:
            return self.get_random_images(count)
        
        # Group by brand
        brands = {}
        for watch_id, data in self.watch_data.items():
            brand = data['brand']
            if brand not in brands:
                brands[brand] = []
            brands[brand].append(data['image_path'])
        
        # Take one from each brand
        selected = []
        brand_list = list(brands.keys())
        random.shuffle(brand_list)
        
        for brand in brand_list:
            if len(selected) >= count:
                break
            selected.append(random.choice(brands[brand]))
        
        # Fill remaining with random if needed
        while len(selected) < count and len(selected) < len(self.watch_data):
            remaining = [path for watch_id, data in self.watch_data.items() 
                        if data['image_path'] not in selected]
            if remaining:
                selected.append(random.choice(remaining))
            else:
                break
        
        return selected
    
    def get_random_images(self, count: int = 4) -> List[str]:
        """Get random images from the directory."""
        image_dir = "production_scrape_20250601_175426/images"
        if not os.path.exists(image_dir):
            return []
        
        image_files = []
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(image_dir, filename))
        
        if len(image_files) <= count:
            return image_files
        
        return random.sample(image_files, count)
    
    def load_and_resize_image(self, image_path: str, target_size: Tuple[int, int]) -> Image.Image:
        """Load and resize an image to fit the target size while maintaining aspect ratio."""
        try:
            image = Image.open(image_path)
            image = image.convert('RGB')
            
            # Calculate aspect ratio preserving resize
            img_ratio = image.width / image.height
            target_ratio = target_size[0] / target_size[1]
            
            if img_ratio > target_ratio:
                new_width = target_size[0]
                new_height = int(new_width / img_ratio)
            else:
                new_height = target_size[1]
                new_width = int(new_height * img_ratio)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create a centered image on white background
            result = Image.new('RGB', target_size, 'white')
            x_offset = (target_size[0] - new_width) // 2
            y_offset = (target_size[1] - new_height) // 2
            result.paste(image, (x_offset, y_offset))
            
            return result
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            placeholder = Image.new('RGB', target_size, '#f0f0f0')
            return placeholder
    
    def create_comparison(self, image_paths: List[str], output_path: str = "smart_comparison.jpg"):
        """Create a clean comparison with no text or labels."""
        if len(image_paths) < 2:
            print("Need at least 2 images to compare")
            return
        
        if len(image_paths) > 6:
            print("Maximum 6 images supported")
            image_paths = image_paths[:6]
        
        num_images = len(image_paths)
        
        # Calculate layout
        if num_images == 2:
            cols, rows = 2, 1
        elif num_images == 3:
            cols, rows = 3, 1
        elif num_images == 4:
            cols, rows = 2, 2
        elif num_images == 5:
            cols, rows = 3, 2
        elif num_images == 6:
            cols, rows = 3, 2
        
        # Calculate individual image size
        img_width = self.output_size[0] // cols
        img_height = self.output_size[1] // rows
        individual_size = (img_width, img_height)
        
        # Create output image
        comparison = Image.new('RGB', self.output_size, 'white')
        
        # Load and place each image
        for i, image_path in enumerate(image_paths):
            col = i % cols
            row = i // cols
            x = col * img_width
            y = row * img_height
            
            img = self.load_and_resize_image(image_path, individual_size)
            comparison.paste(img, (x, y))
        
        comparison.save(output_path, 'JPEG', quality=95)
        print(f"✅ Comparison saved to: {output_path}")
        
        return comparison

def main():
    parser = argparse.ArgumentParser(description='Smart watch image comparison tool')
    parser.add_argument('--brand', '-b', help='Compare watches from specific brand')
    parser.add_argument('--type', '-t', help='Compare watches of specific type (dress, diver, etc.)')
    parser.add_argument('--diverse', '-d', action='store_true', help='Get diverse sample from different brands')
    parser.add_argument('--random', '-r', action='store_true', help='Get completely random selection')
    parser.add_argument('--count', '-c', type=int, default=4, help='Number of images to compare')
    parser.add_argument('--output', '-o', default='smart_comparison.jpg', help='Output filename')
    parser.add_argument('--list-brands', action='store_true', help='List available brands')
    parser.add_argument('--list-types', action='store_true', help='List available watch types')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = SmartImageComparator()
    comparator.load_watch_data()
    
    # List options
    if args.list_brands:
        brands = set()
        for data in comparator.watch_data.values():
            brands.add(data['brand'])
        print("Available brands:")
        for brand in sorted(brands):
            print(f"  - {brand}")
        return
    
    if args.list_types:
        types = set()
        for data in comparator.watch_data.values():
            if data['watch_type']:
                types.add(data['watch_type'])
        print("Available watch types:")
        for watch_type in sorted(types):
            print(f"  - {watch_type}")
        return
    
    # Get images based on selection criteria
    if args.brand:
        image_paths = comparator.get_watches_by_brand(args.brand, args.count)
        print(f"Comparing {len(image_paths)} {args.brand} watches")
    elif args.type:
        image_paths = comparator.get_watches_by_type(args.type, args.count)
        print(f"Comparing {len(image_paths)} {args.type} watches")
    elif args.diverse:
        image_paths = comparator.get_diverse_sample(args.count)
        print(f"Comparing {len(image_paths)} diverse watches")
    elif args.random:
        image_paths = comparator.get_random_images(args.count)
        print(f"Comparing {len(image_paths)} random watches")
    else:
        # Default to diverse sample
        image_paths = comparator.get_diverse_sample(args.count)
        print(f"Comparing {len(image_paths)} diverse watches (default)")
    
    if not image_paths:
        print("No images found matching criteria")
        return
    
    # Create comparison
    comparator.create_comparison(image_paths, args.output)

if __name__ == "__main__":
    main() 