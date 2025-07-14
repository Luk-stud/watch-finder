#!/usr/bin/env python3
"""
Interactive VIT Model Comparison Tool

Creates better-laid-out comparison images that display directly.
Layout: Query watch at top, then each model's neighbors in a grid below.
More intuitive and viewable format with proper spacing.
Includes user preference tracking for model performance.
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict, Counter

class InteractiveVITComparison:
    def __init__(self, image_size: Tuple[int, int] = (200, 200)):
        self.image_size = image_size
        self.models = {}
        self.watch_data = {}
        self.user_preferences = []  # Track user preferences
        
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
        
        return len(self.models) > 0
    
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
    
    def get_user_preference(self, query_watch_id: str, model_names: List[str]) -> str:
        """Get user preference for which model performed best."""
        query_data = self.watch_data[query_watch_id]
        query_brand = query_data.get('brand', 'Unknown')
        query_model = query_data.get('model', 'Unknown')
        
        print(f"\n🎯 Which VIT model found the best similar watches for {query_brand} {query_model}?")
        print("Options:")
        for i, model_name in enumerate(model_names, 1):
            print(f"  {i}. {model_name}")
        print(f"  {len(model_names) + 1}. Skip/No preference")
        
        while True:
            try:
                choice = input(f"\nEnter your choice (1-{len(model_names) + 1}): ").strip()
                
                if choice == '':
                    continue
                    
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(model_names):
                    selected_model = model_names[choice_num - 1]
                    print(f"✅ You selected: {selected_model}")
                    return selected_model
                elif choice_num == len(model_names) + 1:
                    print("⏭️ Skipped")
                    return "skip"
                else:
                    print(f"❌ Please enter a number between 1 and {len(model_names) + 1}")
                    
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                return "exit"
    
    def create_improved_comparison(self, query_watch_id: str):
        """Create an improved comparison with better layout using matplotlib."""
        query_data = self.watch_data[query_watch_id]
        query_brand = query_data.get('brand', 'Unknown')
        query_model = query_data.get('model', 'Unknown')
        
        print(f"🎨 Creating improved comparison for: {query_brand} {query_model}")
        
        num_models = len(self.models)
        neighbors_per_model = 5
        
        # Create a matplotlib figure with better layout
        # Layout: Query at top center, then models in rows below
        fig = plt.figure(figsize=(16, 4 + 3 * num_models))
        fig.suptitle(f'VIT Model Comparison: {query_brand} {query_model}', fontsize=20, fontweight='bold')
        
        # Create grid layout
        gs = fig.add_gridspec(num_models + 1, neighbors_per_model, 
                             height_ratios=[1.2] + [1] * num_models,
                             hspace=0.3, wspace=0.1)
        
        # Add query watch at top center
        query_ax = fig.add_subplot(gs[0, neighbors_per_model//2 - 1:neighbors_per_model//2 + 2])
        query_path = self.get_watch_image_path(query_watch_id)
        if query_path:
            query_img = self.load_and_resize_image(query_path)
            query_ax.imshow(query_img)
        query_ax.set_title(f'QUERY\n{query_brand} {query_model}', fontsize=14, fontweight='bold', color='red')
        query_ax.axis('off')
        
        # Add model comparisons
        for model_idx, (model_name, model_embeddings) in enumerate(self.models.items()):
            print(f"  Finding diverse neighbors for {model_name}...")
            neighbors = self.find_diverse_neighbors(model_embeddings, query_watch_id, neighbors_per_model)
            
            # Add model label
            row = model_idx + 1
            fig.text(0.02, 1 - (row + 0.5) / (num_models + 1), model_name, 
                    fontsize=16, fontweight='bold', color='blue', rotation=90, 
                    verticalalignment='center')
            
            # Add neighbor images
            for neighbor_idx, neighbor_id in enumerate(neighbors):
                neighbor_data = self.watch_data[neighbor_id]
                neighbor_brand = neighbor_data.get('brand', 'Unknown')
                neighbor_model = neighbor_data.get('model', 'Unknown')
                
                ax = fig.add_subplot(gs[row, neighbor_idx])
                neighbor_path = self.get_watch_image_path(neighbor_id)
                if neighbor_path:
                    neighbor_img = self.load_and_resize_image(neighbor_path)
                    ax.imshow(neighbor_img)
                
                # Add brand name below image
                ax.set_title(f'{neighbor_brand}', fontsize=10, pad=5)
                ax.axis('off')
                
                print(f"    {neighbor_idx + 1}. {neighbor_brand} {neighbor_model}")
        
        # Add instructions
        fig.text(0.5, 0.02, 'Each row shows 5 most similar watches from different manufacturers for each VIT model', 
                ha='center', fontsize=12, style='italic')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def show_final_results(self):
        """Show final preference results."""
        if not self.user_preferences:
            print("\n📊 No preferences recorded.")
            return
        
        print("\n" + "="*60)
        print("📊 FINAL VIT MODEL PERFORMANCE RESULTS")
        print("="*60)
        
        # Count preferences
        preference_counts = Counter(self.user_preferences)
        total_votes = len([p for p in self.user_preferences if p != "skip"])
        
        print(f"Total comparisons: {len(self.user_preferences)}")
        print(f"Total votes cast: {total_votes}")
        print(f"Skipped: {preference_counts.get('skip', 0)}")
        print()
        
        if total_votes > 0:
            # Sort by votes
            sorted_models = sorted(
                [(model, count) for model, count in preference_counts.items() if model != "skip"],
                key=lambda x: x[1], reverse=True
            )
            
            print("🏆 MODEL RANKING:")
            for i, (model, count) in enumerate(sorted_models, 1):
                percentage = (count / total_votes) * 100
                print(f"  {i}. {model}: {count} votes ({percentage:.1f}%)")
            
            if sorted_models:
                winner = sorted_models[0]
                print(f"\n🥇 WINNER: {winner[0]} with {winner[1]} votes!")
        
        print("\n📋 Detailed results:")
        for i, preference in enumerate(self.user_preferences, 1):
            status = "⏭️" if preference == "skip" else "✅"
            print(f"  {i:2d}. {status} {preference}")
    
    def run_interactive_comparison(self, num_comparisons: int = 20):
        """Run multiple interactive comparisons with user preference tracking."""
        print(f"🎯 Creating {num_comparisons} interactive VIT comparisons...")
        print("💡 For each comparison, you'll be asked which model found the best similar watches!")
        
        # Get all available watches
        available_watches = self.get_available_watches()
        
        if len(available_watches) < num_comparisons:
            print(f"⚠️ Only {len(available_watches)} watches available")
            num_comparisons = len(available_watches)
        
        # Randomly select watches
        selected_watches = random.sample(available_watches, num_comparisons)
        model_names = list(self.models.keys())
        
        # Create comparison for each watch
        for i, watch_id in enumerate(selected_watches, 1):
            print(f"\n{'='*60}")
            print(f"COMPARISON {i}/{num_comparisons}")
            print('='*60)
            
            # Show the comparison
            self.create_improved_comparison(watch_id)
            
            # Get user preference
            preference = self.get_user_preference(watch_id, model_names)
            
            if preference == "exit":
                print("👋 Exiting early...")
                break
            
            self.user_preferences.append(preference)
            
            # Show running tally every 5 comparisons
            if i % 5 == 0 and i < num_comparisons:
                print(f"\n📊 Progress: {i}/{num_comparisons} completed")
                current_counts = Counter(self.user_preferences)
                for model in model_names:
                    count = current_counts.get(model, 0)
                    print(f"  {model}: {count} votes")
        
        # Show final results
        self.show_final_results()

def main():
    parser = argparse.ArgumentParser(description='Interactive VIT model comparison with user preference tracking')
    parser.add_argument('--models-dir', default='balanced_embeddings', help='Directory with model embeddings')
    parser.add_argument('--count', '-c', type=int, default=20, help='Number of comparisons to show')
    parser.add_argument('--size', type=int, default=200, help='Individual image size')
    
    args = parser.parse_args()
    
    # Initialize comparator
    image_size = (args.size, args.size)
    comparator = InteractiveVITComparison(image_size)
    
    # Load model embeddings
    if not comparator.load_model_embeddings(args.models_dir):
        print("Trying DINO fallback...")
        if not comparator.load_dino_fallback():
            print("❌ Could not load any model embeddings")
            return
    
    # Run interactive comparisons
    comparator.run_interactive_comparison(args.count)

if __name__ == "__main__":
    main() 