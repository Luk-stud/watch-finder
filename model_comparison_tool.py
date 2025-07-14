#!/usr/bin/env python3
"""
Model Comparison Tool for Watch Style Analysis

Compares different vision models to see which one best groups watches by aesthetic style:
- DINO ViT (current)
- CLIP ViT-B/32 
- CLIP ViT-L/14

Analyzes how well each model clusters watches by style categories like:
- Minimalist
- Art Deco
- Retro/Vintage
- Modern
- Sporty
- Luxury
"""

import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ModelComparisonTool:
    def __init__(self, data_dir='balanced_embeddings'):
        self.models = {}
        self.watch_data = {}
        self.data_dir = data_dir
        self.style_categories = {
            'minimalist': ['minimalist', 'simple', 'clean', 'minimal'],
            'art_deco': ['art deco', 'deco', 'geometric', 'decorative'],
            'retro': ['retro', 'vintage', 'classic', 'heritage'],
            'modern': ['modern', 'contemporary', 'sleek'],
            'sporty': ['sport', 'diver', 'racing', 'chronograph'],
            'luxury': ['luxury', 'premium', 'elegant', 'sophisticated']
        }
        
    def load_metadata(self):
        """Load metadata from balanced_embeddings/metadata.json."""
        with open(os.path.join(self.data_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        self.watch_data = {item['watch_id']: item for item in metadata}
        print(f"✅ Loaded metadata for {len(self.watch_data)} watches")

    def load_embeddings(self):
        """Load all model embeddings from balanced_embeddings/"""
        model_files = {
            'clip_vit_b32': 'clip_vit_b32_embeddings.pkl',
            'clip_vit_l14': 'clip_vit_l14_embeddings.pkl',
            'dino_vit': 'dino_vit_embeddings.pkl',
        }
        for model_name, fname in model_files.items():
            path = os.path.join(self.data_dir, fname)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self.models[model_name] = {
                        'embeddings': pickle.load(f),
                        'name': model_name
                    }
                print(f"✅ Loaded {model_name} embeddings: {len(self.models[model_name]['embeddings'])}")
            else:
                print(f"⚠️ Embeddings file not found: {path}")

    def get_watch_image(self, watch_id: str, size: Tuple[int, int] = (100, 100)) -> Image.Image:
        try:
            watch_data = self.watch_data[watch_id]
            image_path = watch_data.get('image_path')
            if not image_path or not os.path.exists(image_path):
                return Image.new('RGB', size, color='lightgray')
            image = Image.open(image_path).convert('RGB')
            image = image.resize(size, Image.Resampling.LANCZOS)
            return image
        except Exception:
            return Image.new('RGB', size, color='lightgray')

    def analyze_style_clustering(self, model_name: str, n_samples: int = 50):
        if model_name not in self.models:
            print(f"❌ Model {model_name} not found")
            return None
        model = self.models[model_name]
        embeddings = model['embeddings']
        print(f"\n🎨 Analyzing {model_name} style clustering...")
        # Get sample of watches that exist in both embeddings and watch_data
        common_watches = [wid for wid in list(embeddings.keys()) if wid in self.watch_data][:n_samples]
        if len(common_watches) < 10:
            print(f"⚠️ Not enough common watches ({len(common_watches)})")
            return None
        watch_embeddings = [embeddings[watch_id] for watch_id in common_watches]
        similarity_matrix = cosine_similarity(watch_embeddings)
        n_clusters = min(6, len(common_watches) // 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(watch_embeddings)
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_watches = [common_watches[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            watch_types = []
            brands = []
            for watch_id in cluster_watches:
                watch_data = self.watch_data[watch_id]
                watch_type = watch_data.get('watch_type', 'Unknown')
                brand = watch_data.get('brand', 'Unknown')
                watch_types.append(watch_type)
                brands.append(brand)
            cluster_analysis[cluster_id] = {
                'watches': cluster_watches,
                'watch_types': watch_types,
                'brands': brands,
                'size': len(cluster_watches)
            }
        return {
            'model_name': model_name,
            'similarity_matrix': similarity_matrix,
            'cluster_labels': cluster_labels,
            'cluster_analysis': cluster_analysis,
            'watch_ids': common_watches
        }

    def visualize_style_clustering(self, analysis_results: Dict[str, Any], save_path: str = None):
        if not analysis_results:
            return
        model_name = analysis_results['model_name']
        similarity_matrix = analysis_results['similarity_matrix']
        cluster_labels = analysis_results['cluster_labels']
        cluster_analysis = analysis_results['cluster_analysis']
        watch_ids = analysis_results['watch_ids']
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Style Clustering Analysis: {model_name}', fontsize=16)
        sns.heatmap(similarity_matrix, ax=axes[0, 0], cmap='viridis', cbar_kws={'label': 'Cosine Similarity'})
        axes[0, 0].set_title('Similarity Matrix')
        axes[0, 0].set_xlabel('Watch Index')
        axes[0, 0].set_ylabel('Watch Index')
        cluster_sizes = [analysis['size'] for analysis in cluster_analysis.values()]
        cluster_ids = list(cluster_analysis.keys())
        axes[0, 1].bar(cluster_ids, cluster_sizes)
        axes[0, 1].set_title('Cluster Sizes')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Number of Watches')
        watch_type_counts = {}
        for cluster_id, analysis in cluster_analysis.items():
            for watch_type in analysis['watch_types']:
                if watch_type not in watch_type_counts:
                    watch_type_counts[watch_type] = {}
                if cluster_id not in watch_type_counts[watch_type]:
                    watch_type_counts[watch_type][cluster_id] = 0
                watch_type_counts[watch_type][cluster_id] += 1
        watch_types = list(watch_type_counts.keys())
        x = np.arange(len(cluster_ids))
        width = 0.8 / len(watch_types)
        for i, watch_type in enumerate(watch_types):
            counts = [watch_type_counts[watch_type].get(cluster_id, 0) for cluster_id in cluster_ids]
            axes[0, 2].bar(x + i * width, counts, width, label=watch_type, alpha=0.8)
        axes[0, 2].set_title('Watch Types per Cluster')
        axes[0, 2].set_xlabel('Cluster ID')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].legend()
        axes[0, 2].set_xticks(x + width * (len(watch_types) - 1) / 2)
        axes[0, 2].set_xticklabels(cluster_ids)
        for i, cluster_id in enumerate(cluster_ids[:3]):
            cluster_watches = cluster_analysis[cluster_id]['watches']
            sample_watches = cluster_watches[:4]
            for j, watch_id in enumerate(sample_watches):
                img = self.get_watch_image(watch_id, size=(80, 80))
                axes[1, i].imshow(img)
                axes[1, i].set_title(f'Cluster {cluster_id} Sample')
                axes[1, i].axis('off')
                watch_data = self.watch_data[watch_id]
                brand = watch_data.get('brand', 'Unknown')
                watch_type = watch_data.get('watch_type', 'Unknown')
                axes[1, i].text(0.5, -0.1, f'{brand}\n{watch_type}', 
                               ha='center', va='top', transform=axes[1, i].transAxes, 
                               fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved clustering analysis to {save_path}")
        plt.show()
        print(f"\n📊 {model_name} Clustering Analysis:")
        print("="*50)
        for cluster_id, analysis in cluster_analysis.items():
            print(f"\nCluster {cluster_id} ({analysis['size']} watches):")
            from collections import Counter
            type_counts = Counter(analysis['watch_types'])
            print(f"  Watch types: {dict(type_counts.most_common(3))}")
            brand_counts = Counter(analysis['brands'])
            print(f"  Top brands: {dict(brand_counts.most_common(3))}")
            sample_watches = analysis['watches'][:3]
            for watch_id in sample_watches:
                watch_data = self.watch_data[watch_id]
                print(f"    - {watch_data.get('brand')} {watch_data.get('model')} ({watch_data.get('watch_type', 'Unknown')})")

    def compare_models_side_by_side(self, n_samples: int = 40):
        print("🔍 Comparing Models Side by Side")
        print("="*60)
        results = {}
        for model_name in self.models.keys():
            results[model_name] = self.analyze_style_clustering(model_name, n_samples)
        if len(results) > 1:
            fig, axes = plt.subplots(len(results), 2, figsize=(12, 4 * len(results)))
            if len(results) == 1:
                axes = axes.reshape(1, -1)
            for i, (model_name, analysis) in enumerate(results.items()):
                if not analysis:
                    continue
                sns.heatmap(analysis['similarity_matrix'], ax=axes[i, 0], cmap='viridis')
                axes[i, 0].set_title(f'{analysis["model_name"]} - Similarity Matrix')
                cluster_analysis = analysis['cluster_analysis']
                cluster_sizes = [cluster_analysis[cid]['size'] for cid in sorted(cluster_analysis.keys())]
                cluster_ids = sorted(cluster_analysis.keys())
                axes[i, 1].bar(cluster_ids, cluster_sizes)
                axes[i, 1].set_title(f'{analysis["model_name"]} - Cluster Sizes')
                axes[i, 1].set_xlabel('Cluster ID')
                axes[i, 1].set_ylabel('Count')
            plt.tight_layout()
            plt.savefig('model_comparison_side_by_side.png', dpi=300, bbox_inches='tight')
            print("✅ Saved side-by-side comparison to model_comparison_side_by_side.png")
            plt.show()
        return results

    def recommend_best_model(self, results: Dict[str, Any]):
        print("\n🏆 Model Recommendation Analysis")
        print("="*50)
        if not results:
            print("❌ No results to analyze")
            return
        model_scores = {}
        for model_name, analysis in results.items():
            if not analysis:
                continue
            cluster_analysis = analysis['cluster_analysis']
            total_watches = sum(cluster['size'] for cluster in cluster_analysis.values())
            avg_cluster_size = total_watches / len(cluster_analysis)
            type_separation_score = 0
            for cluster in cluster_analysis.values():
                type_counts = {}
                for watch_type in cluster['watch_types']:
                    type_counts[watch_type] = type_counts.get(watch_type, 0) + 1
                max_type_count = max(type_counts.values()) if type_counts else 0
                cluster_dominance = max_type_count / cluster['size']
                type_separation_score += cluster_dominance
            type_separation_score /= len(cluster_analysis)
            brand_diversity_score = 0
            for cluster in cluster_analysis.values():
                unique_brands = len(set(cluster['brands']))
                brand_diversity_score += unique_brands / cluster['size']
            brand_diversity_score /= len(cluster_analysis)
            overall_score = (type_separation_score * 0.6 + 
                           (1 - brand_diversity_score) * 0.4)
            model_scores[model_name] = {
                'overall_score': overall_score,
                'type_separation': type_separation_score,
                'brand_diversity': brand_diversity_score,
                'avg_cluster_size': avg_cluster_size,
                'num_clusters': len(cluster_analysis)
            }
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1]['overall_score'], reverse=True)
        print("📊 Model Rankings:")
        for i, (model_name, scores) in enumerate(ranked_models):
            print(f"\n{i+1}. {model_name}")
            print(f"   Overall Score: {scores['overall_score']:.3f}")
            print(f"   Type Separation: {scores['type_separation']:.3f}")
            print(f"   Brand Diversity: {scores['brand_diversity']:.3f}")
            print(f"   Avg Cluster Size: {scores['avg_cluster_size']:.1f}")
            print(f"   Number of Clusters: {scores['num_clusters']}")
        if ranked_models:
            best_model = ranked_models[0][0]
            print(f"\n🏆 RECOMMENDATION: {best_model}")
            print(f"This model shows the best balance of style-based clustering and watch type separation.")
            return best_model
        return None

def main():
    print("🔬 Model Comparison Tool for Watch Style Analysis")
    print("="*60)
    tool = ModelComparisonTool()
    tool.load_metadata()
    tool.load_embeddings()
    print(f"\n📊 Loaded {len(tool.models)} models:")
    for model_name, model_info in tool.models.items():
        print(f"  - {model_name}: {len(model_info['embeddings'])} embeddings")
    if len(tool.models) < 2:
        print("\n⚠️ Need at least 2 models for comparison.")
        return
    results = tool.compare_models_side_by_side(n_samples=40)
    for model_name in tool.models.keys():
        if model_name in results and results[model_name]:
            tool.visualize_style_clustering(
                results[model_name], 
                save_path=f'{model_name}_style_clustering.png'
            )
    best_model = tool.recommend_best_model(results)
    print(f"\n✅ Model comparison complete!")
    print(f"Check the generated PNG files for detailed visualizations.")

if __name__ == "__main__":
    main() 