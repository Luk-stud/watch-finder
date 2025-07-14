#!/usr/bin/env python3
"""
Multi VIT Model Comparison Tool

Creates multiple separate images, each showing:
- 1 main watch (random manufacturer)
- 5 nearest neighbors from each VIT model (each from different manufacturers)
- Labels at the top showing which model
"""

import os
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import argparse
import json
import random
from collections import defaultdict

class MultiVITComparison:
    def __init__(self, image_size: Tuple[int, int] = (150, 150)):
        self.image_size = image_size
        self.models = {}
        self.watch_data = {}
        
    def load_model_embeddings(self, models_dir: str = "balanced_embeddings"):
        """Load embeddings from different VIT models."""
        print("📊 Loading VIT model embeddings...")
        
        # Load metadata
        metadata_path = os.path.join(models_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.watch_data = {item['watch_id']: item for item in metadata}
            print(f"✅ Loaded metadata for {len(self.watch_data)} watches")
        
        # Load embeddings for each model
        model_files = {
            'CLIP ViT-B/32': 'clip_vit_b32_embeddings.pkl',
            'CLIP ViT-L/14': 'clip_vit_l14_embeddings.pkl',
            'DINO ViT': 'dino_vit_embeddings.pkl',
        }
        
        for model_name, filename in model_files.items():
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    embeddings = pickle.load(f)
                self.models[model_name] = embeddings
                print(f"✅ Loaded {model_name}: {len(embeddings)} embeddings")
            else:
                print(f"⚠️ Model file not found: {filepath}")
        
        if not self.models:
            print("❌ No model embeddings found")
            return False
        
        return True
    
    def load_dino_fallback(self, filepath: str = 'data/precomputed_embeddings.pkl'):
        """Fallback to load DINO embeddings if balanced embeddings not available."""
        print("📊 Loading DINO embeddings as fallback...")
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.watch_data = data['watch_data']
            self.models['DINO ViT'] = data['final_embeddings']
            print(f"✅ Loaded DINO fallback: {len(self.models['DINO ViT'])} embeddings")
            return True
        except Exception as e:
            print(f"❌ Could not load DINO fallback: {e}")
            return False
    
    def get_available_watches(self) -> List[str]:
        """Get watches that have embeddings in all models."""
        available = []
        for watch_id in self.watch_data.keys():
            if all(watch_id in model_embeddings for model_embeddings in self.models.values()):
                available.append(watch_id)
        return available
    
    def find_diverse_neighbors(self, model_embeddings: Dict, query_watch_id: str, n_neighbors: int = 5) -> List[str]:
        """Find nearest neighbors from different manufacturers."""
        if query_watch_id not in model_embeddings:
            return []
        
        query_embedding = model_embeddings[query_watch_id].reshape(1, -1)
        query_brand = self.watch_data[query_watch_id].get('brand', 'Unknown')
        
        # Calculate similarities for all watches
        similarities = []
        for watch_id, embedding in model_embeddings.items():
            if watch_id == query_watch_id:
                continue
            
            similarity = cosine_similarity(query_embedding, embedding.reshape(1, -1))[0][0]
            brand = self.watch_data[watch_id].get('brand', 'Unknown')
            similarities.append((watch_id, similarity, brand))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Select diverse neighbors (different manufacturers)
        selected = []
        used_brands = {query_brand}  # Don't include same brand as query
        
        for watch_id, similarity, brand in similarities:
            if brand not in used_brands:
                selected.append(watch_id)
                used_brands.add(brand)
                
                if len(selected) >= n_neighbors:
                    break
        
        return selected
    
    def get_watch_image_path(self, watch_id: str) -> str:
        """Get image path for a watch."""
        watch_info = self.watch_data.get(watch_id, {})
        image_path = watch_info.get('image_path')
        
        if image_path and os.path.exists(image_path):
            return image_path
        
        # Try to construct path
        brand = watch_info.get('brand', '')
        model = watch_info.get('model', '')
        
        if brand and model:
            brand_clean = brand.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
            model_clean = model.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
            
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
            placeholder = Image.new('RGB', self.image_size, '#f0f0f0')
            return placeholder
    
    def create_single_comparison(self, query_watch_id: str, output_path: str):
        """Create a single comparison image for one watch."""
        query_data = self.watch_data[query_watch_id]
        query_brand = query_data.get('brand', 'Unknown')
        query_model = query_data.get('model', 'Unknown')
        
        print(f"🎨 Creating comparison for: {query_brand} {query_model}")
        
        num_models = len(self.models)
        neighbors_per_model = 5
        
        # Layout: 1 main + 5 neighbors per model
        images_per_row = 1 + (neighbors_per_model * num_models)
        
        # Calculate dimensions
        header_height = 50  # Height for headers
        label_height = 30   # Height for model labels
        total_width = images_per_row * self.image_size[0]
        total_height = header_height + label_height + self.image_size[1]
        
        # Create the comparison image
        comparison = Image.new('RGB', (total_width, total_height), 'white')
        draw = ImageDraw.Draw(comparison)
        
        # Try to load fonts
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            try:
                title_font = ImageFont.truetype("arial.ttf", 20)
                label_font = ImageFont.truetype("arial.ttf", 14)
            except:
                title_font = ImageFont.load_default()
                label_font = ImageFont.load_default()
        
        # Add main title
        title = f"Query: {query_brand} {query_model}"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (total_width - title_width) // 2
        draw.text((title_x, 10), title, fill='red', font=title_font)
        
        # Add column headers
        current_x = self.image_size[0] // 2
        draw.text((current_x, header_height + 5), "QUERY", fill='black', font=label_font, anchor='mt')
        current_x += self.image_size[0]
        
        for model_name in self.models.keys():
            label_x = current_x + (neighbors_per_model * self.image_size[0]) // 2
            draw.text((label_x, header_height + 5), f"{model_name}", fill='blue', font=label_font, anchor='mt')
            current_x += neighbors_per_model * self.image_size[0]
        
        # Place main watch image
        image_y = header_height + label_height
        image_path = self.get_watch_image_path(query_watch_id)
        if image_path:
            main_img = self.load_and_resize_image(image_path)
            comparison.paste(main_img, (0, image_y))
        
        current_x = self.image_size[0]
        
        # Add neighbors for each model
        for model_name, model_embeddings in self.models.items():
            print(f"  Finding diverse neighbors for {model_name}...")
            neighbors = self.find_diverse_neighbors(model_embeddings, query_watch_id, neighbors_per_model)
            
            print(f"    Found {len(neighbors)} diverse neighbors:")
            for neighbor_id in neighbors:
                neighbor_data = self.watch_data[neighbor_id]
                print(f"      - {neighbor_data.get('brand', 'Unknown')} {neighbor_data.get('model', 'Unknown')}")
            
            # Place neighbor images
            for neighbor_id in neighbors:
                neighbor_path = self.get_watch_image_path(neighbor_id)
                if neighbor_path:
                    neighbor_img = self.load_and_resize_image(neighbor_path)
                    comparison.paste(neighbor_img, (current_x, image_y))
                
                current_x += self.image_size[0]
        
        # Save the comparison
        comparison.save(output_path, 'JPEG', quality=95)
        print(f"✅ Saved: {output_path}")
        
        return comparison
    
    def create_multiple_comparisons(self, num_images: int = 5, output_prefix: str = "vit_comparison"):
        """Create multiple comparison images."""
        print(f"🎯 Creating {num_images} VIT comparison images...")
        
        # Get all available watches
        available_watches = self.get_available_watches()
        
        if len(available_watches) < num_images:
            print(f"⚠️ Only {len(available_watches)} watches available, creating {len(available_watches)} images")
            num_images = len(available_watches)
        
        # Randomly select watches
        selected_watches = random.sample(available_watches, num_images)
        
        # Create comparison for each watch
        for i, watch_id in enumerate(selected_watches, 1):
            output_path = f"{output_prefix}_{i:02d}.jpg"
            self.create_single_comparison(watch_id, output_path)
        
        print(f"✅ Created {num_images} comparison images with prefix '{output_prefix}_XX.jpg'")

def main():
    parser = argparse.ArgumentParser(description='Create multiple VIT model comparison images')
    parser.add_argument('--models-dir', default='balanced_embeddings', help='Directory with model embeddings')
    parser.add_argument('--count', '-c', type=int, default=5, help='Number of comparison images to create')
    parser.add_argument('--prefix', '-p', default='vit_comparison', help='Output filename prefix')
    parser.add_argument('--size', type=int, default=150, help='Individual image size')
    
    args = parser.parse_args()
    
    # Initialize comparator
    image_size = (args.size, args.size)
    comparator = MultiVITComparison(image_size)
    
    # Load model embeddings
    if not comparator.load_model_embeddings(args.models_dir):
        print("Trying DINO fallback...")
        if not comparator.load_dino_fallback():
            print("❌ Could not load any model embeddings")
            return
    
    # Create multiple comparisons
    comparator.create_multiple_comparisons(args.count, args.prefix)

if __name__ == "__main__":
    main() 