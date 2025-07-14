#!/usr/bin/env python3
"""
Nearest Neighbor Comparison Tool

Creates one large image showing all watches paired with their nearest neighbors.
Pure visual comparison - no text, just watch images side by side.
"""

import os
import pickle
import numpy as np
from PIL import Image
import json
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import argparse

class NearestNeighborComparison:
    def __init__(self, image_size: Tuple[int, int] = (150, 150)):
        self.image_size = image_size
        self.embeddings = {}
        self.watch_data = {}
        
    def load_embeddings(self, filepath: str = 'data/precomputed_embeddings.pkl'):
        """Load watch embeddings and data."""
        print("📊 Loading embeddings...")
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.watch_data = data['watch_data']
            self.embeddings = data['final_embeddings']
            print(f"✅ Loaded {len(self.embeddings)} embeddings")
            return True
        except Exception as e:
            print(f"❌ Could not load embeddings: {e}")
            return False
    
    def find_nearest_neighbor(self, query_watch_id: str) -> str:
        """Find the nearest neighbor for a given watch."""
        if query_watch_id not in self.embeddings:
            return None
        
        query_embedding = self.embeddings[query_watch_id].reshape(1, -1)
        best_similarity = -1
        best_neighbor = None
        
        for watch_id, embedding in self.embeddings.items():
            if watch_id == query_watch_id:
                continue
            
            similarity = cosine_similarity(query_embedding, embedding.reshape(1, -1))[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_neighbor = watch_id
        
        return best_neighbor
    
    def get_watch_image_path(self, watch_id: str) -> str:
        """Get the image path for a watch."""
        watch_info = self.watch_data.get(watch_id, {})
        image_path = watch_info.get('image_path')
        
        if image_path and os.path.exists(image_path):
            return image_path
        
        # Try to construct path from brand and model
        brand = watch_info.get('brand', '')
        model = watch_info.get('model', '')
        
        if brand and model:
            # Clean brand and model names
            brand_clean = brand.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
            model_clean = model.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
            
            # Try different path patterns
            possible_paths = [
                f"production_scrape_20250601_175426/images/{brand_clean}_{model_clean}_main.jpg",
                f"production_scrape_20250601_175426/images/{brand}_{model}_main.jpg".replace(' ', '_'),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    return path
        
        return None
    
    def load_and_resize_image(self, image_path: str) -> Image.Image:
        """Load and resize an image."""
        try:
            image = Image.open(image_path)
            image = image.convert('RGB')
            
            # Calculate aspect ratio preserving resize
            img_ratio = image.width / image.height
            target_ratio = self.image_size[0] / self.image_size[1]
            
            if img_ratio > target_ratio:
                new_width = self.image_size[0]
                new_height = int(new_width / img_ratio)
            else:
                new_height = self.image_size[1]
                new_width = int(new_height * img_ratio)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create centered image on white background
            result = Image.new('RGB', self.image_size, 'white')
            x_offset = (self.image_size[0] - new_width) // 2
            y_offset = (self.image_size[1] - new_height) // 2
            result.paste(image, (x_offset, y_offset))
            
            return result
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return placeholder
            placeholder = Image.new('RGB', self.image_size, '#f0f0f0')
            return placeholder
    
    def create_full_comparison(self, max_pairs: int = None, output_path: str = "all_nearest_neighbors.jpg"):
        """Create one large image with all watch pairs."""
        print("🔍 Finding nearest neighbors for all watches...")
        
        # Find all valid watch pairs
        valid_pairs = []
        processed = set()
        
        for watch_id in self.embeddings.keys():
            if watch_id in processed:
                continue
            
            # Get image path for this watch
            image_path = self.get_watch_image_path(watch_id)
            if not image_path:
                continue
            
            # Find nearest neighbor
            neighbor_id = self.find_nearest_neighbor(watch_id)
            if not neighbor_id:
                continue
            
            # Get image path for neighbor
            neighbor_path = self.get_watch_image_path(neighbor_id)
            if not neighbor_path:
                continue
            
            valid_pairs.append((watch_id, image_path, neighbor_id, neighbor_path))
            processed.add(watch_id)
            processed.add(neighbor_id)  # Avoid duplicate pairs
            
            if max_pairs and len(valid_pairs) >= max_pairs:
                break
        
        print(f"✅ Found {len(valid_pairs)} valid watch pairs")
        
        if not valid_pairs:
            print("❌ No valid pairs found")
            return
        
        # Calculate grid layout
        pairs_per_row = 5  # 5 pairs per row = 10 images per row
        num_rows = (len(valid_pairs) + pairs_per_row - 1) // pairs_per_row
        
        # Calculate total image size
        pair_width = self.image_size[0] * 2  # Two images side by side
        total_width = pair_width * pairs_per_row
        total_height = self.image_size[1] * num_rows
        
        print(f"📐 Creating {total_width}x{total_height} image with {num_rows} rows")
        
        # Create the massive comparison image
        comparison = Image.new('RGB', (total_width, total_height), 'white')
        
        for i, (watch_id, image_path, neighbor_id, neighbor_path) in enumerate(valid_pairs):
            # Calculate position
            row = i // pairs_per_row
            col = i % pairs_per_row
            
            base_x = col * pair_width
            base_y = row * self.image_size[1]
            
            # Load and place original watch
            watch_img = self.load_and_resize_image(image_path)
            comparison.paste(watch_img, (base_x, base_y))
            
            # Load and place nearest neighbor
            neighbor_img = self.load_and_resize_image(neighbor_path)
            comparison.paste(neighbor_img, (base_x + self.image_size[0], base_y))
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(valid_pairs)} pairs...")
        
        # Save the massive comparison
        print(f"💾 Saving comparison image...")
        comparison.save(output_path, 'JPEG', quality=85)
        print(f"✅ Comparison saved to: {output_path}")
        print(f"📊 Image size: {comparison.size}")
        
        return comparison
    
    def create_sample_comparison(self, sample_size: int = 20, output_path: str = "sample_nearest_neighbors.jpg"):
        """Create a smaller sample comparison for testing."""
        print(f"🎯 Creating sample comparison with {sample_size} pairs")
        return self.create_full_comparison(max_pairs=sample_size, output_path=output_path)

def main():
    parser = argparse.ArgumentParser(description='Create nearest neighbor comparison of all watches')
    parser.add_argument('--sample', '-s', type=int, help='Create sample with N pairs instead of all')
    parser.add_argument('--output', '-o', default='all_nearest_neighbors.jpg', help='Output filename')
    parser.add_argument('--embeddings', '-e', default='data/precomputed_embeddings.pkl', help='Embeddings file path')
    parser.add_argument('--size', type=int, default=150, help='Individual image size (square)')
    
    args = parser.parse_args()
    
    # Initialize comparator
    image_size = (args.size, args.size)
    comparator = NearestNeighborComparison(image_size)
    
    # Load embeddings
    if not comparator.load_embeddings(args.embeddings):
        print("Failed to load embeddings. Exiting.")
        return
    
    # Create comparison
    if args.sample:
        comparator.create_sample_comparison(args.sample, args.output)
    else:
        comparator.create_full_comparison(output_path=args.output)

if __name__ == "__main__":
    main() 