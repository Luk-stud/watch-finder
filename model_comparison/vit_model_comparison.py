#!/usr/bin/env python3
"""
VIT Model Comparison Tool

Compares different VIT models on the same set of 10 example watches.
For each watch, shows 5 nearest neighbors from each model.
Layout: Main watch + 5 neighbors from Model 1 + 5 neighbors from Model 2 + etc.
All on one large image with model labels.
"""

import os
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import argparse
import json
from collections import defaultdict

class VITModelComparison:
    def __init__(self, image_size: Tuple[int, int] = (120, 120)):
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
    
    def select_diverse_watches(self, count: int = 10) -> List[str]:
        """Select diverse watches, one per manufacturer."""
        print(f"🎯 Selecting {count} diverse watches (one per manufacturer)...")
        
        # Group by manufacturer
        manufacturers = defaultdict(list)
        for watch_id, data in self.watch_data.items():
            # Check if this watch has embeddings in all models
            if all(watch_id in model_embeddings for model_embeddings in self.models.values()):
                brand = data.get('brand', 'Unknown')
                manufacturers[brand].append(watch_id)
        
        # Select one watch per manufacturer
        selected = []
        for brand, watches in manufacturers.items():
            if len(selected) >= count:
                break
            # Take the first watch from this manufacturer
            selected.append(watches[0])
        
        print(f"✅ Selected {len(selected)} watches from different manufacturers:")
        for watch_id in selected:
            data = self.watch_data[watch_id]
            print(f"  - {data.get('brand', 'Unknown')} {data.get('model', 'Unknown')}")
        
        return selected
    
    def find_nearest_neighbors(self, model_embeddings: Dict, query_watch_id: str, n_neighbors: int = 5) -> List[str]:
        """Find nearest neighbors for a query watch in a specific model."""
        if query_watch_id not in model_embeddings:
            return []
        
        query_embedding = model_embeddings[query_watch_id].reshape(1, -1)
        similarities = []
        
        for watch_id, embedding in model_embeddings.items():
            if watch_id == query_watch_id:
                continue
            
            similarity = cosine_similarity(query_embedding, embedding.reshape(1, -1))[0][0]
            similarities.append((watch_id, similarity))
        
        # Sort by similarity and take top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [watch_id for watch_id, _ in similarities[:n_neighbors]]
    
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
    
    def create_model_comparison(self, selected_watches: List[str], output_path: str = "vit_model_comparison.jpg"):
        """Create the comprehensive model comparison image."""
        print(f"🎨 Creating VIT model comparison for {len(selected_watches)} watches...")
        
        num_models = len(self.models)
        neighbors_per_model = 5
        
        # Layout: Main watch + 5 neighbors per model
        images_per_row = 1 + (neighbors_per_model * num_models)  # 1 main + 5*N neighbors
        
        # Calculate dimensions
        label_height = 40  # Height for model labels
        total_width = images_per_row * self.image_size[0]
        total_height = len(selected_watches) * (self.image_size[1] + label_height)
        
        print(f"📐 Creating {total_width}x{total_height} comparison image")
        print(f"   Layout: {len(selected_watches)} rows × {images_per_row} images per row")
        
        # Create the comparison image
        comparison = Image.new('RGB', (total_width, total_height), 'white')
        draw = ImageDraw.Draw(comparison)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
        
        for row, watch_id in enumerate(selected_watches):
            print(f"  Processing row {row + 1}/{len(selected_watches)}: {self.watch_data[watch_id].get('brand', 'Unknown')}")
            
            base_y = row * (self.image_size[1] + label_height)
            current_x = 0
            
            # Place main watch image
            image_path = self.get_watch_image_path(watch_id)
            if image_path:
                main_img = self.load_and_resize_image(image_path)
                comparison.paste(main_img, (current_x, base_y + label_height))
            
            current_x += self.image_size[0]
            
            # Add model sections
            for model_name, model_embeddings in self.models.items():
                # Add model label
                label_x = current_x + (neighbors_per_model * self.image_size[0]) // 2
                draw.text((label_x, base_y + 10), model_name, fill='black', font=font, anchor='mt')
                
                # Find and place neighbors
                neighbors = self.find_nearest_neighbors(model_embeddings, watch_id, neighbors_per_model)
                
                for i, neighbor_id in enumerate(neighbors):
                    neighbor_path = self.get_watch_image_path(neighbor_id)
                    if neighbor_path:
                        neighbor_img = self.load_and_resize_image(neighbor_path)
                        comparison.paste(neighbor_img, (current_x, base_y + label_height))
                    
                    current_x += self.image_size[0]
        
        # Add header labels
        header_y = 10
        current_x = self.image_size[0] // 2
        draw.text((current_x, header_y), "QUERY", fill='red', font=font, anchor='mt')
        current_x += self.image_size[0]
        
        for model_name in self.models.keys():
            label_x = current_x + (neighbors_per_model * self.image_size[0]) // 2
            draw.text((label_x, header_y), f"{model_name} (5 Nearest)", fill='blue', font=font, anchor='mt')
            current_x += neighbors_per_model * self.image_size[0]
        
        # Save the comparison
        comparison.save(output_path, 'JPEG', quality=90)
        print(f"✅ VIT model comparison saved to: {output_path}")
        print(f"📊 Image size: {comparison.size}")
        
        return comparison

def main():
    parser = argparse.ArgumentParser(description='Compare VIT models on watch nearest neighbors')
    parser.add_argument('--models-dir', default='balanced_embeddings', help='Directory with model embeddings')
    parser.add_argument('--count', '-c', type=int, default=10, help='Number of example watches')
    parser.add_argument('--output', '-o', default='vit_model_comparison.jpg', help='Output filename')
    parser.add_argument('--size', type=int, default=120, help='Individual image size')
    
    args = parser.parse_args()
    
    # Initialize comparator
    image_size = (args.size, args.size)
    comparator = VITModelComparison(image_size)
    
    # Load model embeddings
    if not comparator.load_model_embeddings(args.models_dir):
        print("Trying DINO fallback...")
        if not comparator.load_dino_fallback():
            print("❌ Could not load any model embeddings")
            return
    
    # Select diverse watches
    selected_watches = comparator.select_diverse_watches(args.count)
    
    if not selected_watches:
        print("❌ No suitable watches found")
        return
    
    # Create comparison
    comparator.create_model_comparison(selected_watches, args.output)

if __name__ == "__main__":
    main() 