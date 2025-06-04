#!/usr/bin/env python3
"""
Multi-Expert LinUCB Strategy Analysis
====================================

Analyze different approaches to expert initialization, exploration, and specialization
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple, Set
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class ExpertStrategyAnalyzer:
    def __init__(self):
        self.load_data()
        
    def load_data(self):
        """Load the watch embeddings for analysis"""
        print("Loading watch data...")
        with open('watch_text_embeddings.pkl', 'rb') as f:
            self.embeddings = pickle.load(f)
        with open('watch_text_metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"✅ Loaded {len(self.metadata)} watches with {self.embeddings.shape[1]}D embeddings")
        
        # Extract brands for analysis
        self.brands = [watch.get('brand', 'Unknown') for watch in self.metadata]
        self.unique_brands = list(set(self.brands))
        print(f"📊 Found {len(self.unique_brands)} unique brands")

    def analyze_current_strategy(self):
        """Analyze the current dynamic expert strategy"""
        print("\n🔍 CURRENT STRATEGY ANALYSIS")
        print("=" * 60)
        
        current_config = {
            'initial_experts': 2,
            'max_experts': 4,
            'min_expert_size': 3,
            'similarity_threshold': 0.65,
            'total_watches': len(self.metadata)
        }
        
        print("📋 Current Configuration:")
        for key, value in current_config.items():
            print(f"   {key}: {value}")
        
        # Calculate initial assignment
        initially_assigned = current_config['initial_experts'] * current_config['min_expert_size']
        unassigned_ratio = (current_config['total_watches'] - initially_assigned) / current_config['total_watches']
        
        print(f"\n⚖️  Initial Distribution:")
        print(f"   Initially assigned: {initially_assigned} watches ({(1-unassigned_ratio)*100:.1f}%)")
        print(f"   Initially unassigned: {current_config['total_watches'] - initially_assigned} watches ({unassigned_ratio*100:.1f}%)")
        
        print(f"\n🎯 Exploration Strategy:")
        print(f"   ✅ Expert specialization: Each expert focuses on similar watches")
        print(f"   ✅ Dynamic creation: New experts for diverse preferences")
        print(f"   ✅ Unassigned exploration: Large pool for discovery")
        print(f"   ✅ UCB within experts: Balances exploitation vs exploration")
        
        return current_config

    def analyze_clustering_approaches(self):
        """Analyze different clustering approaches for expert initialization"""
        print("\n🧮 CLUSTERING APPROACH ANALYSIS")
        print("=" * 60)
        
        # Test different numbers of initial clusters
        cluster_options = [2, 3, 4, 5, 8, 10]
        results = {}
        
        # Standardize embeddings for clustering
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.embeddings)
        
        for n_clusters in cluster_options:
            print(f"\n📊 Testing {n_clusters} initial clusters...")
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_scaled)
            
            # Analyze cluster characteristics
            cluster_sizes = [np.sum(cluster_labels == i) for i in range(n_clusters)]
            cluster_brands = {}
            
            for i in range(n_clusters):
                cluster_indices = np.where(cluster_labels == i)[0]
                cluster_brand_list = [self.brands[idx] for idx in cluster_indices]
                unique_cluster_brands = list(set(cluster_brand_list))
                cluster_brands[i] = {
                    'top_brands': sorted([(brand, cluster_brand_list.count(brand)) 
                                        for brand in unique_cluster_brands], 
                                       key=lambda x: x[1], reverse=True)[:5],
                    'brand_diversity': len(unique_cluster_brands)
                }
            
            # Calculate metrics
            size_balance = np.std(cluster_sizes) / np.mean(cluster_sizes)  # Lower is more balanced
            avg_brand_diversity = np.mean([cluster_brands[i]['brand_diversity'] for i in range(n_clusters)])
            
            results[n_clusters] = {
                'cluster_sizes': cluster_sizes,
                'size_balance': size_balance,
                'brand_diversity': avg_brand_diversity,
                'cluster_brands': cluster_brands
            }
            
            print(f"   Cluster sizes: {cluster_sizes}")
            print(f"   Size balance (std/mean): {size_balance:.3f}")
            print(f"   Avg brand diversity: {avg_brand_diversity:.1f}")
            
            # Show top brands per cluster
            for i in range(min(3, n_clusters)):  # Show first 3 clusters
                top_brands = results[n_clusters]['cluster_brands'][i]['top_brands'][:3]
                print(f"   Cluster {i} top brands: {[f'{b}({c})' for b, c in top_brands]}")
        
        return results

    def analyze_exploration_strategies(self):
        """Analyze different exploration strategies"""
        print("\n🔍 EXPLORATION STRATEGY OPTIONS")
        print("=" * 60)
        
        strategies = {
            "Current: Dynamic + Unassigned": {
                'description': 'Start small, create experts dynamically, large unassigned pool',
                'pros': [
                    'Fast initial recommendations',
                    'Adapts to user preferences', 
                    'Good exploration through unassigned pool',
                    'Memory efficient'
                ],
                'cons': [
                    'May miss natural clusters initially',
                    'Cold start problem for new experts',
                    'Potentially slower convergence'
                ],
                'best_for': 'Diverse user bases, speed priority'
            },
            
            "Pre-clustered Experts": {
                'description': 'Start with 4-5 experts based on K-means clustering',
                'pros': [
                    'Immediate specialization',
                    'Natural cluster boundaries',
                    'Faster convergence to user preferences',
                    'Good for known watch categories'
                ],
                'cons': [
                    'Higher initial memory usage',
                    'May force artificial boundaries',
                    'Less adaptive to unexpected preferences'
                ],
                'best_for': 'Well-understood watch market, quality priority'
            },
            
            "Hybrid: Clustered + Dynamic": {
                'description': 'Start with 2-3 pre-clustered experts + dynamic creation',
                'pros': [
                    'Balance of speed and specialization',
                    'Some immediate structure',
                    'Still adaptive',
                    'Medium memory usage'
                ],
                'cons': [
                    'More complex to implement',
                    'Requires tuning multiple parameters'
                ],
                'best_for': 'Balanced performance and adaptability'
            },
            
            "Brand-based Experts": {
                'description': 'Create experts based on luxury tiers or brand categories',
                'pros': [
                    'Intuitive segmentation',
                    'Matches user mental models',
                    'Clear specialization',
                    'Easy to interpret'
                ],
                'cons': [
                    'May miss cross-brand similarities',
                    'Brand bias',
                    'Fixed structure'
                ],
                'best_for': 'Brand-conscious users, interpretability'
            },
            
            "Epsilon-Greedy Multi-Expert": {
                'description': 'Use epsilon-greedy for expert selection + UCB within experts',
                'pros': [
                    'Simple exploration control',
                    'Predictable behavior',
                    'Good for A/B testing'
                ],
                'cons': [
                    'Less sophisticated than UCB',
                    'Uniform random exploration'
                ],
                'best_for': 'Simple, predictable exploration'
            }
        }
        
        for name, strategy in strategies.items():
            print(f"\n🎯 {name}")
            print(f"   Description: {strategy['description']}")
            print(f"   ✅ Pros:")
            for pro in strategy['pros']:
                print(f"      • {pro}")
            print(f"   ❌ Cons:")
            for con in strategy['cons']:
                print(f"      • {con}")
            print(f"   🎯 Best for: {strategy['best_for']}")
        
        return strategies

    def recommend_optimal_strategy(self, clustering_results):
        """Recommend the optimal strategy based on analysis"""
        print("\n💡 STRATEGY RECOMMENDATIONS")
        print("=" * 60)
        
        # Analyze clustering results to find sweet spot
        best_clustering = None
        best_score = -float('inf')
        
        for n_clusters, results in clustering_results.items():
            # Score based on balance and diversity
            balance_score = 1 / (1 + results['size_balance'])  # Higher is better
            diversity_score = results['brand_diversity'] / 20  # Normalize
            
            # Prefer moderate number of clusters (3-5)
            cluster_penalty = abs(n_clusters - 4) * 0.1
            
            total_score = balance_score + diversity_score - cluster_penalty
            
            if total_score > best_score:
                best_score = total_score
                best_clustering = n_clusters
        
        print(f"🎯 RECOMMENDED APPROACH: Hybrid Strategy")
        print(f"   Optimal initial clusters: {best_clustering}")
        print(f"   Strategy: Pre-clustered + Dynamic")
        
        print(f"\n📋 Recommended Configuration:")
        recommended_config = {
            'initial_experts': best_clustering,
            'max_experts': max(6, best_clustering + 2),
            'pre_cluster_method': 'kmeans',
            'dynamic_creation': True,
            'similarity_threshold': 0.7,
            'min_expert_size': 50,  # Larger initial experts
            'unassigned_ratio': 0.3,  # Keep 30% unassigned for exploration
            'exploration_strategy': 'ucb_with_epsilon_boost'
        }
        
        for key, value in recommended_config.items():
            print(f"   {key}: {value}")
        
        print(f"\n🚀 Benefits of this approach:")
        print(f"   ✅ Immediate specialization from clustering")
        print(f"   ✅ Adaptive to unexpected preferences")
        print(f"   ✅ Balanced exploration vs exploitation")
        print(f"   ✅ Good scalability")
        
        return recommended_config

    def analyze_alpha_exploration_tradeoffs(self):
        """Analyze different alpha values for exploration vs exploitation"""
        print("\n⚖️  ALPHA PARAMETER ANALYSIS")
        print("=" * 60)
        
        alpha_values = [0.5, 1.0, 1.5, 2.0, 2.5]
        
        print("Alpha controls exploration vs exploitation balance:")
        print("   • Lower alpha → More exploitation (confident recommendations)")
        print("   • Higher alpha → More exploration (try diverse options)")
        
        for alpha in alpha_values:
            exploration_level = min(100, alpha * 50)  # Rough mapping
            exploitation_level = 100 - exploration_level
            
            print(f"\n🎯 Alpha = {alpha}")
            print(f"   Exploration: {exploration_level:.0f}%")
            print(f"   Exploitation: {exploitation_level:.0f}%")
            
            if alpha <= 0.8:
                print("   📊 Behavior: Mostly exploit known preferences")
                print("   🎯 Good for: Established users, consistent preferences")
            elif alpha <= 1.5:
                print("   📊 Behavior: Balanced exploration and exploitation")
                print("   🎯 Good for: General use, moderate discovery")
            else:
                print("   📊 Behavior: Heavy exploration, try diverse options")
                print("   🎯 Good for: New users, diverse preferences")
        
        print(f"\n💡 Current alpha = 1.2 is well-balanced for general use")

def main():
    """Run complete strategy analysis"""
    print("🎯 MULTI-EXPERT LINUCB STRATEGY ANALYSIS")
    print("=" * 80)
    
    analyzer = ExpertStrategyAnalyzer()
    
    # Analyze current approach
    current_config = analyzer.analyze_current_strategy()
    
    # Analyze clustering options
    clustering_results = analyzer.analyze_clustering_approaches()
    
    # Analyze exploration strategies
    strategies = analyzer.analyze_exploration_strategies()
    
    # Alpha analysis
    analyzer.analyze_alpha_exploration_tradeoffs()
    
    # Final recommendations
    recommended_config = analyzer.recommend_optimal_strategy(clustering_results)
    
    print(f"\n🎉 Analysis complete!")
    print(f"Current approach is good for speed, but hybrid approach may be better for quality.")
    
    return {
        'current_config': current_config,
        'clustering_results': clustering_results,
        'strategies': strategies,
        'recommended_config': recommended_config
    }

if __name__ == "__main__":
    main() 