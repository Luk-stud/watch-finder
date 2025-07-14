#!/usr/bin/env python3
"""
Visual Embedding Comparison Tool

Shows actual watch images alongside their embeddings to understand
what the vision encoder is learning and why certain matches occur.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import requests
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class VisualEmbeddingComparator:
    def __init__(self):
        self.embeddings = {}
        self.watch_data = {}
        self.image_cache = {}
        
    def load_embeddings(self, filepath: str = 'data/precomputed_embeddings.pkl'):
        """Load embeddings and watch data."""
        print("📊 Loading embeddings...")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.watch_data = data['watch_data']
        self.embeddings = data['final_embeddings']
        print(f"✅ Loaded {len(self.embeddings)} embeddings")
    
    def get_watch_image(self, watch_id: str, size: Tuple[int, int] = (200, 200)) -> Image.Image:
        """Get watch image from local path or cache."""
        if watch_id in self.image_cache:
            return self.image_cache[watch_id]
        
        try:
            watch_data = self.watch_data[watch_id]
            image_path = watch_data.get('image_path')
            
            if not image_path:
                print(f"⚠️ No image path for {watch_id}")
                # Create placeholder
                placeholder = Image.new('RGB', size, color='lightgray')
                self.image_cache[watch_id] = placeholder
                return placeholder
            
            # Check if file exists
            if not os.path.exists(image_path):
                print(f"⚠️ Image file not found: {image_path}")
                placeholder = Image.new('RGB', size, color='lightgray')
                self.image_cache[watch_id] = placeholder
                return placeholder
            
            # Load local image
            image = Image.open(image_path).convert('RGB')
            image = image.resize(size, Image.Resampling.LANCZOS)
            
            self.image_cache[watch_id] = image
            return image
            
        except Exception as e:
            print(f"⚠️ Failed to load image for {watch_id}: {e}")
            placeholder = Image.new('RGB', size, color='lightgray')
            self.image_cache[watch_id] = placeholder
            return placeholder
    
    def find_nearest_neighbors(self, query_watch_id: str, n_neighbors: int = 5, exclude_same_brand: bool = True) -> List[Tuple]:
        """Find nearest neighbors for a query watch."""
        if query_watch_id not in self.embeddings:
            return []
        
        query_embedding = self.embeddings[query_watch_id]
        query_data = self.watch_data[query_watch_id]
        query_brand = query_data.get('brand', '').lower()
        
        similarities = []
        for watch_id, embedding in self.embeddings.items():
            if watch_id == query_watch_id:
                continue
            
            # Skip same brand if requested
            if exclude_same_brand:
                neighbor_data = self.watch_data[watch_id]
                neighbor_brand = neighbor_data.get('brand', '').lower()
                if query_brand == neighbor_brand:
                    continue
            
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append((watch_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_neighbors]
    
    def visualize_query_and_neighbors(self, query_watch_id: str, n_neighbors: int = 5, 
                                    exclude_same_brand: bool = True, save_path: str = None):
        """Create a visual comparison of query watch and its nearest neighbors."""
        if query_watch_id not in self.watch_data:
            print(f"❌ Watch {query_watch_id} not found")
            return
        
        # Get query watch info
        query_data = self.watch_data[query_watch_id]
        query_brand = query_data.get('brand', 'Unknown')
        query_model = query_data.get('model', 'Unknown')
        query_type = query_data.get('specs', {}).get('watch_type', 'Unknown')
        
        print(f"🔍 Query: {query_brand} {query_model} ({query_type})")
        print(f"   Image path: {query_data.get('image_path', 'No path')}")
        
        # Find neighbors
        neighbors = self.find_nearest_neighbors(query_watch_id, n_neighbors, exclude_same_brand)
        
        if not neighbors:
            print("❌ No neighbors found")
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, n_neighbors + 1, figsize=(4 * (n_neighbors + 1), 8))
        fig.suptitle(f'Visual Embedding Comparison: {query_brand} {query_model}', fontsize=16)
        
        # Query watch
        query_img = self.get_watch_image(query_watch_id)
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title(f'Query\n{query_brand} {query_model}\n({query_type})', 
                           fontsize=10, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Add similarity info for query
        axes[1, 0].text(0.5, 0.5, f'Query Watch\n{query_type}', 
                       ha='center', va='center', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Neighbors
        for i, (neighbor_id, similarity) in enumerate(neighbors):
            neighbor_data = self.watch_data[neighbor_id]
            neighbor_brand = neighbor_data.get('brand', 'Unknown')
            neighbor_model = neighbor_data.get('model', 'Unknown')
            neighbor_type = neighbor_data.get('specs', {}).get('watch_type', 'Unknown')
            
            # Image
            neighbor_img = self.get_watch_image(neighbor_id)
            axes[0, i + 1].imshow(neighbor_img)
            axes[0, i + 1].set_title(f'Match #{i+1}\n{neighbor_brand} {neighbor_model}\n({neighbor_type})\nSimilarity: {similarity:.3f}', 
                                   fontsize=10)
            axes[0, i + 1].axis('off')
            
            # Color code by type match
            color = 'green' if neighbor_type.lower() == query_type.lower() else 'red'
            axes[1, i + 1].text(0.5, 0.5, f'{neighbor_type}\nSimilarity: {similarity:.3f}', 
                               ha='center', va='center', fontsize=12, color=color)
            axes[1, i + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved visualization to {save_path}")
        
        plt.show()
        
        # Print detailed analysis
        print(f"\n📊 Analysis:")
        print(f"  Query type: {query_type}")
        type_matches = sum(1 for _, (n_id, _) in enumerate(neighbors) 
                          if self.watch_data[n_id].get('specs', {}).get('watch_type', '').lower() == query_type.lower())
        print(f"  Type matches: {type_matches}/{len(neighbors)}")
        print(f"  Average similarity: {np.mean([s for _, s in neighbors]):.3f}")
        
        return neighbors
    
    def compare_type_clustering(self, watch_types: List[str] = ['Dress', 'Diver', 'Chronograph']):
        """Compare how different watch types cluster visually."""
        print("🎯 Type Clustering Comparison")
        print("="*50)
        
        fig, axes = plt.subplots(len(watch_types), 6, figsize=(24, 4 * len(watch_types)))
        
        for i, watch_type in enumerate(watch_types):
            print(f"\n📊 Analyzing {watch_type} watches...")
            
            # Find watches of this type
            type_watches = []
            for watch_id, data in self.watch_data.items():
                if watch_id in self.embeddings:
                    specs = data.get('specs', {})
                    if specs.get('watch_type', '').lower() == watch_type.lower():
                        type_watches.append(watch_id)
                        if len(type_watches) >= 5:
                            break
            
            if not type_watches:
                print(f"  ⚠️ No {watch_type} watches found")
                continue
            
            # Show examples of this type
            for j, watch_id in enumerate(type_watches):
                watch_data = self.watch_data[watch_id]
                brand = watch_data.get('brand', 'Unknown')
                model = watch_data.get('model', 'Unknown')
                
                img = self.get_watch_image(watch_id)
                axes[i, j].imshow(img)
                axes[i, j].set_title(f'{brand}\n{model}', fontsize=10)
                axes[i, j].axis('off')
            
            # Find nearest neighbors for first watch of this type
            if type_watches:
                query_watch = type_watches[0]
                neighbors = self.find_nearest_neighbors(query_watch, n_neighbors=1, exclude_same_brand=True)
                
                if neighbors:
                    neighbor_id, similarity = neighbors[0]
                    neighbor_data = self.watch_data[neighbor_id]
                    neighbor_brand = neighbor_data.get('brand', 'Unknown')
                    neighbor_model = neighbor_data.get('model', 'Unknown')
                    neighbor_type = neighbor_data.get('specs', {}).get('watch_type', 'Unknown')
                    
                    img = self.get_watch_image(neighbor_id)
                    axes[i, 5].imshow(img)
                    axes[i, 5].set_title(f'Nearest Neighbor\n{neighbor_brand} {neighbor_model}\n({neighbor_type})\nSimilarity: {similarity:.3f}', 
                                       fontsize=10, color='red' if neighbor_type != watch_type else 'green')
                    axes[i, 5].axis('off')
                    
                    print(f"  Query: {self.watch_data[query_watch].get('brand')} {self.watch_data[query_watch].get('model')}")
                    print(f"  Nearest: {neighbor_brand} {neighbor_model} ({neighbor_type}) - {similarity:.3f}")
        
        plt.tight_layout()
        plt.savefig('type_clustering_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Saved type clustering comparison to type_clustering_comparison.png")
        plt.show()
    
    def analyze_problematic_matches(self):
        """Analyze specific problematic matches to understand the issue."""
        print("🔍 Analyzing Problematic Matches")
        print("="*50)
        
        # Find dress watches that match with dive watches
        problematic_cases = []
        
        for watch_id, data in self.watch_data.items():
            if watch_id not in self.embeddings:
                continue
                
            specs = data.get('specs', {})
            if specs.get('watch_type', '').lower() == 'dress':
                neighbors = self.find_nearest_neighbors(watch_id, n_neighbors=3, exclude_same_brand=True)
                
                for neighbor_id, similarity in neighbors:
                    neighbor_data = self.watch_data[neighbor_id]
                    neighbor_type = neighbor_data.get('specs', {}).get('watch_type', 'Unknown')
                    
                    if neighbor_type.lower() in ['diver', 'chronograph']:
                        problematic_cases.append({
                            'dress_watch': watch_id,
                            'dress_brand': data.get('brand'),
                            'dress_model': data.get('model'),
                            'neighbor_watch': neighbor_id,
                            'neighbor_brand': neighbor_data.get('brand'),
                            'neighbor_model': neighbor_data.get('model'),
                            'neighbor_type': neighbor_type,
                            'similarity': similarity
                        })
                        break
        
        print(f"📊 Found {len(problematic_cases)} problematic dress watch matches")
        
        # Visualize top problematic cases
        if problematic_cases:
            # Sort by similarity (most problematic first)
            problematic_cases.sort(key=lambda x: x['similarity'], reverse=True)
            
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            fig.suptitle('Problematic Dress Watch Matches', fontsize=16)
            
            for i, case in enumerate(problematic_cases[:6]):
                row = i // 2
                col = (i % 2) * 2
                
                # Dress watch
                dress_img = self.get_watch_image(case['dress_watch'])
                axes[row, col].imshow(dress_img)
                axes[row, col].set_title(f"Dress Watch\n{case['dress_brand']} {case['dress_model']}", 
                                       fontsize=10, color='blue')
                axes[row, col].axis('off')
                
                # Problematic neighbor
                neighbor_img = self.get_watch_image(case['neighbor_watch'])
                axes[row, col + 1].imshow(neighbor_img)
                axes[row, col + 1].set_title(f"Nearest Neighbor\n{case['neighbor_brand']} {case['neighbor_model']}\n({case['neighbor_type']})\nSimilarity: {case['similarity']:.3f}", 
                                           fontsize=10, color='red')
                axes[row, col + 1].axis('off')
            
            plt.tight_layout()
            plt.savefig('problematic_matches.png', dpi=300, bbox_inches='tight')
            print("✅ Saved problematic matches to problematic_matches.png")
            plt.show()

def main():
    """Main function to run visual comparisons."""
    print("🎨 Visual Embedding Comparison Tool")
    print("="*50)
    
    # Initialize comparator
    comparator = VisualEmbeddingComparator()
    comparator.load_embeddings()
    
    # Find some interesting test cases
    test_cases = []
    
    # Find dress watches
    for watch_id, data in comparator.watch_data.items():
        if watch_id in comparator.embeddings:
            specs = data.get('specs', {})
            if specs.get('watch_type', '').lower() == 'dress':
                test_cases.append(watch_id)
                if len(test_cases) >= 3:
                    break
    
    # Find dive watches
    for watch_id, data in comparator.watch_data.items():
        if watch_id in comparator.embeddings:
            specs = data.get('specs', {})
            if specs.get('watch_type', '').lower() == 'diver':
                test_cases.append(watch_id)
                if len(test_cases) >= 6:
                    break
    
    print(f"🎯 Testing {len(test_cases)} watch samples...")
    
    # Visualize each test case
    for i, watch_id in enumerate(test_cases):
        print(f"\n" + "="*60)
        print(f"Test Case {i+1}/{len(test_cases)}")
        print("="*60)
        
        comparator.visualize_query_and_neighbors(
            watch_id, 
            n_neighbors=5, 
            exclude_same_brand=True,
            save_path=f'visual_comparison_{i+1}.png'
        )
    
    # Compare type clustering
    print(f"\n" + "="*60)
    print("TYPE CLUSTERING COMPARISON")
    print("="*60)
    comparator.compare_type_clustering()
    
    # Analyze problematic matches
    print(f"\n" + "="*60)
    print("PROBLEMATIC MATCHES ANALYSIS")
    print("="*60)
    comparator.analyze_problematic_matches()
    
    print(f"\n✅ Visual comparison complete! Check the generated PNG files.")

if __name__ == "__main__":
    main() 