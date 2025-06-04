#!/usr/bin/env python3
"""
Brand Distance Analyzer
=======================

Analyze how far apart different watch brands are in the embedding space:
- Calculate inter-brand and intra-brand distances
- Measure brand cohesion vs separation
- Identify most distinctive vs similar brands
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import defaultdict

class BrandDistanceAnalyzer:
    def __init__(self):
        self.df = None
        self.embeddings = None
        self.metadata = None
        self.significant_brands = None
        self.load_data()
        
    def load_data(self):
        """Load watch data and embeddings"""
        print("Loading watch data...")
        
        with open('watch_text_embeddings.pkl', 'rb') as f:
            self.embeddings = pickle.load(f)
        with open('watch_text_metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"✅ Loaded {len(self.metadata)} watches with {self.embeddings.shape[1]}D embeddings")
        
        # Create dataframe
        self.df = pd.DataFrame(self.metadata)
        self.df['watch_id'] = range(len(self.df))
        
        # Filter to brands with sufficient watches for analysis
        brand_counts = self.df['brand'].value_counts()
        self.significant_brands = brand_counts[brand_counts >= 10].index.tolist()
        
        print(f"📊 Found {len(self.significant_brands)} brands with 10+ watches for analysis")
        
    def compute_brand_centroids(self) -> Dict[str, np.ndarray]:
        """Compute the centroid embedding for each brand"""
        print("🎯 Computing brand centroids...")
        
        brand_centroids = {}
        for brand in self.significant_brands:
            brand_watches = self.df[self.df['brand'] == brand]
            brand_embeddings = self.embeddings[brand_watches['watch_id'].values]
            
            # Compute centroid (mean embedding)
            centroid = np.mean(brand_embeddings, axis=0)
            brand_centroids[brand] = centroid
            
        print(f"✅ Computed centroids for {len(brand_centroids)} brands")
        return brand_centroids
    
    def analyze_intra_brand_cohesion(self) -> Dict[str, Dict[str, float]]:
        """Analyze how cohesive each brand is internally"""
        print("📏 Analyzing intra-brand cohesion...")
        
        brand_cohesion = {}
        
        for brand in self.significant_brands:
            brand_watches = self.df[self.df['brand'] == brand]
            brand_embeddings = self.embeddings[brand_watches['watch_id'].values]
            
            if len(brand_embeddings) < 2:
                continue
                
            # Calculate pairwise distances within brand
            cosine_sim_matrix = cosine_similarity(brand_embeddings)
            euclidean_dist_matrix = euclidean_distances(brand_embeddings)
            
            # Extract upper triangle (excluding diagonal)
            n = len(brand_embeddings)
            upper_tri_indices = np.triu_indices(n, k=1)
            
            cosine_similarities = cosine_sim_matrix[upper_tri_indices]
            euclidean_distances_vals = euclidean_dist_matrix[upper_tri_indices]
            
            brand_cohesion[brand] = {
                'avg_cosine_similarity': np.mean(cosine_similarities),
                'std_cosine_similarity': np.std(cosine_similarities),
                'avg_euclidean_distance': np.mean(euclidean_distances_vals),
                'std_euclidean_distance': np.std(euclidean_distances_vals),
                'num_watches': len(brand_embeddings),
                'cohesion_score': np.mean(cosine_similarities)  # Higher = more cohesive
            }
        
        return brand_cohesion
    
    def analyze_inter_brand_distances(self, brand_centroids: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Analyze distances between different brands"""
        print("🔍 Analyzing inter-brand distances...")
        
        brands = list(brand_centroids.keys())
        n_brands = len(brands)
        
        # Create distance matrices
        cosine_distances = np.zeros((n_brands, n_brands))
        euclidean_distances_matrix = np.zeros((n_brands, n_brands))
        
        for i, brand_a in enumerate(brands):
            for j, brand_b in enumerate(brands):
                if i != j:
                    # Cosine distance (1 - cosine similarity)
                    cos_sim = cosine_similarity([brand_centroids[brand_a]], [brand_centroids[brand_b]])[0, 0]
                    cosine_distances[i, j] = 1 - cos_sim
                    
                    # Euclidean distance
                    eucl_dist = np.linalg.norm(brand_centroids[brand_a] - brand_centroids[brand_b])
                    euclidean_distances_matrix[i, j] = eucl_dist
        
        # Create DataFrame for easier analysis
        distance_df = pd.DataFrame(cosine_distances, index=brands, columns=brands)
        
        return distance_df, euclidean_distances_matrix
    
    def find_most_similar_and_different_brands(self, distance_df: pd.DataFrame) -> Dict:
        """Find most similar and most different brand pairs"""
        print("🎯 Finding most similar and different brand pairs...")
        
        # Get upper triangle (exclude diagonal and duplicates)
        brands = distance_df.index.tolist()
        similarities = []
        
        for i in range(len(brands)):
            for j in range(i + 1, len(brands)):
                brand_a, brand_b = brands[i], brands[j]
                distance = distance_df.loc[brand_a, brand_b]
                similarities.append({
                    'brand_a': brand_a,
                    'brand_b': brand_b,
                    'cosine_distance': distance,
                    'cosine_similarity': 1 - distance
                })
        
        similarities_df = pd.DataFrame(similarities)
        
        # Most similar (lowest distance)
        most_similar = similarities_df.nsmallest(10, 'cosine_distance')
        
        # Most different (highest distance)
        most_different = similarities_df.nlargest(10, 'cosine_distance')
        
        return {
            'most_similar': most_similar,
            'most_different': most_different,
            'all_similarities': similarities_df
        }
    
    def analyze_brand_separation_statistics(self, distance_df: pd.DataFrame, brand_cohesion: Dict) -> Dict:
        """Compute overall statistics about brand separation"""
        print("📊 Computing brand separation statistics...")
        
        # Get all inter-brand distances (upper triangle)
        brands = distance_df.index.tolist()
        inter_brand_distances = []
        
        for i in range(len(brands)):
            for j in range(i + 1, len(brands)):
                inter_brand_distances.append(distance_df.iloc[i, j])
        
        inter_brand_distances = np.array(inter_brand_distances)
        
        # Get all intra-brand similarities
        intra_brand_similarities = [
            data['avg_cosine_similarity'] for data in brand_cohesion.values()
        ]
        
        # Convert to distances for comparison
        intra_brand_distances = [1 - sim for sim in intra_brand_similarities]
        
        stats = {
            'inter_brand_stats': {
                'mean_distance': np.mean(inter_brand_distances),
                'std_distance': np.std(inter_brand_distances),
                'min_distance': np.min(inter_brand_distances),
                'max_distance': np.max(inter_brand_distances),
                'median_distance': np.median(inter_brand_distances)
            },
            'intra_brand_stats': {
                'mean_distance': np.mean(intra_brand_distances),
                'std_distance': np.std(intra_brand_distances),
                'min_distance': np.min(intra_brand_distances),
                'max_distance': np.max(intra_brand_distances),
                'median_distance': np.median(intra_brand_distances)
            },
            'separation_ratio': np.mean(inter_brand_distances) / np.mean(intra_brand_distances)
        }
        
        return stats
    
    def create_brand_heatmap(self, distance_df: pd.DataFrame, top_n: int = 15):
        """Create a heatmap of brand distances"""
        print(f"🎨 Creating brand distance heatmap for top {top_n} brands...")
        
        # Select top N brands by number of watches
        brand_counts = self.df['brand'].value_counts()
        top_brands = [b for b in brand_counts.head(top_n).index if b in distance_df.index]
        
        # Subset the distance matrix
        subset_df = distance_df.loc[top_brands, top_brands]
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(subset_df, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='viridis_r',  # Reverse so dark = similar
                   square=True,
                   cbar_kws={'label': 'Cosine Distance (0=identical, 1=opposite)'})
        
        plt.title(f'Brand Distance Matrix - Top {len(top_brands)} Brands\n(Darker = More Similar)')
        plt.xlabel('Brand')
        plt.ylabel('Brand')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('brand_distance_heatmap.png', dpi=150, bbox_inches='tight')
        print("✅ Heatmap saved as 'brand_distance_heatmap.png'")
        
        return subset_df
    
    def print_analysis_summary(self, stats: Dict, similarities: Dict, brand_cohesion: Dict):
        """Print a comprehensive summary of the analysis"""
        print("\n" + "="*60)
        print("📊 BRAND SEPARATION ANALYSIS SUMMARY")
        print("="*60)
        
        # Overall separation
        inter_stats = stats['inter_brand_stats']
        intra_stats = stats['intra_brand_stats']
        
        print(f"\n🎯 OVERALL BRAND SEPARATION:")
        print(f"   Average distance BETWEEN brands: {inter_stats['mean_distance']:.4f}")
        print(f"   Average distance WITHIN brands:  {intra_stats['mean_distance']:.4f}")
        print(f"   Separation ratio: {stats['separation_ratio']:.2f}x")
        
        if stats['separation_ratio'] > 2.0:
            separation_quality = "EXCELLENT - Brands are very distinct"
        elif stats['separation_ratio'] > 1.5:
            separation_quality = "GOOD - Brands are reasonably separated"
        elif stats['separation_ratio'] > 1.2:
            separation_quality = "MODERATE - Some brand overlap"
        else:
            separation_quality = "POOR - Brands are not well separated"
            
        print(f"   Quality: {separation_quality}")
        
        # Range analysis
        print(f"\n📏 DISTANCE RANGES:")
        print(f"   Inter-brand distances: {inter_stats['min_distance']:.4f} to {inter_stats['max_distance']:.4f}")
        print(f"   Intra-brand distances: {intra_stats['min_distance']:.4f} to {intra_stats['max_distance']:.4f}")
        
        # Most similar brands
        print(f"\n🤝 MOST SIMILAR BRAND PAIRS:")
        for idx, row in similarities['most_similar'].head(5).iterrows():
            print(f"   {row['brand_a']} ↔ {row['brand_b']}: {row['cosine_similarity']:.4f} similarity")
        
        # Most different brands
        print(f"\n⚡ MOST DIFFERENT BRAND PAIRS:")
        for idx, row in similarities['most_different'].head(5).iterrows():
            print(f"   {row['brand_a']} ↔ {row['brand_b']}: {row['cosine_distance']:.4f} distance")
        
        # Most cohesive brands
        print(f"\n🎯 MOST COHESIVE BRANDS (internal consistency):")
        cohesive_brands = sorted(brand_cohesion.items(), 
                               key=lambda x: x[1]['cohesion_score'], reverse=True)
        for brand, data in cohesive_brands[:5]:
            print(f"   {brand}: {data['cohesion_score']:.4f} avg similarity ({data['num_watches']} watches)")
        
        # Least cohesive brands
        print(f"\n🌪️  LEAST COHESIVE BRANDS (most diverse internally):")
        for brand, data in cohesive_brands[-5:]:
            print(f"   {brand}: {data['cohesion_score']:.4f} avg similarity ({data['num_watches']} watches)")

def main():
    """Run complete brand distance analysis"""
    print("🔍 BRAND DISTANCE ANALYSIS")
    print("=" * 50)
    
    analyzer = BrandDistanceAnalyzer()
    
    # Compute brand centroids
    brand_centroids = analyzer.compute_brand_centroids()
    
    # Analyze intra-brand cohesion
    brand_cohesion = analyzer.analyze_intra_brand_cohesion()
    
    # Analyze inter-brand distances
    distance_df, euclidean_matrix = analyzer.analyze_inter_brand_distances(brand_centroids)
    
    # Find similarities and differences
    similarities = analyzer.find_most_similar_and_different_brands(distance_df)
    
    # Compute overall statistics
    stats = analyzer.analyze_brand_separation_statistics(distance_df, brand_cohesion)
    
    # Create visualization
    subset_df = analyzer.create_brand_heatmap(distance_df, top_n=15)
    
    # Print comprehensive summary
    analyzer.print_analysis_summary(stats, similarities, brand_cohesion)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📊 Check 'brand_distance_heatmap.png' for visual representation")
    
    return {
        'brand_centroids': brand_centroids,
        'brand_cohesion': brand_cohesion,
        'distance_matrix': distance_df,
        'similarities': similarities,
        'stats': stats
    }

if __name__ == "__main__":
    main() 