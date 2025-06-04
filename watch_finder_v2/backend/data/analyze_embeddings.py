#!/usr/bin/env python3
"""
Analyze text embeddings to determine optimal dimensionality reduction
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def load_embeddings():
    """Load the text embeddings"""
    print("Loading embeddings...")
    with open('watch_text_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    print(f"✅ Loaded embeddings shape: {embeddings.shape}")
    return embeddings

def analyze_density_and_sparsity(embeddings):
    """Analyze the density and sparsity of embeddings"""
    print("\n🔍 EMBEDDING DENSITY ANALYSIS")
    print("=" * 50)
    
    # Basic statistics
    mean_vals = np.mean(embeddings, axis=0)
    std_vals = np.std(embeddings, axis=0)
    
    print(f"📊 Basic Statistics:")
    print(f"   Mean range: [{np.min(mean_vals):.4f}, {np.max(mean_vals):.4f}]")
    print(f"   Std range: [{np.min(std_vals):.4f}, {np.max(std_vals):.4f}]")
    print(f"   Overall mean: {np.mean(embeddings):.4f}")
    print(f"   Overall std: {np.std(embeddings):.4f}")
    
    # Check for near-zero dimensions
    near_zero_threshold = 0.001
    near_zero_dims = np.sum(np.abs(mean_vals) < near_zero_threshold)
    low_variance_dims = np.sum(std_vals < 0.01)
    
    print(f"\n🎯 Information Density:")
    print(f"   Near-zero dimensions (|mean| < {near_zero_threshold}): {near_zero_dims}/1536 ({near_zero_dims/1536*100:.1f}%)")
    print(f"   Low-variance dimensions (std < 0.01): {low_variance_dims}/1536 ({low_variance_dims/1536*100:.1f}%)")
    
    # Most/least informative dimensions
    variance_sorted = np.argsort(std_vals)[::-1]
    print(f"\n📈 Most informative dimensions (highest variance):")
    for i in range(5):
        dim = variance_sorted[i]
        print(f"   Dim {dim}: std={std_vals[dim]:.4f}, mean={mean_vals[dim]:.4f}")
    
    print(f"\n📉 Least informative dimensions (lowest variance):")
    for i in range(5):
        dim = variance_sorted[-(i+1)]
        print(f"   Dim {dim}: std={std_vals[dim]:.4f}, mean={mean_vals[dim]:.4f}")
    
    return mean_vals, std_vals, variance_sorted

def analyze_pca_information_retention(embeddings):
    """Analyze how much information is retained at different PCA levels"""
    print("\n🎯 PCA INFORMATION RETENTION ANALYSIS")
    print("=" * 50)
    
    # Standardize the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Perform PCA
    print("Computing PCA...")
    pca = PCA()
    pca.fit(embeddings_scaled)
    
    # Analyze explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find dimensions needed for different variance thresholds
    thresholds = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(f"\n📊 Variance Retention vs Dimensions:")
    for threshold in thresholds:
        dims_needed = np.argmax(cumulative_variance >= threshold) + 1
        print(f"   {threshold*100:2.0f}% variance: {dims_needed:3d} dimensions ({dims_needed/1536*100:4.1f}% of original)")
    
    # Current reduction analysis
    current_dims = [20, 50, 100, 200, 300]
    print(f"\n🔍 Information Loss at Different Reductions:")
    for dims in current_dims:
        if dims <= len(cumulative_variance):
            variance_retained = cumulative_variance[dims-1]
            variance_lost = 1 - variance_retained
            print(f"   {dims:3d}D: retains {variance_retained*100:5.1f}%, loses {variance_lost*100:5.1f}%")
    
    return pca, explained_variance_ratio, cumulative_variance

def recommend_optimal_dimensions(cumulative_variance):
    """Recommend optimal dimensions based on analysis"""
    print("\n💡 RECOMMENDATIONS")
    print("=" * 50)
    
    # Different use cases
    recommendations = {
        "Speed-focused (current)": 20,
        "Balanced performance": np.argmax(cumulative_variance >= 0.80) + 1,
        "High quality": np.argmax(cumulative_variance >= 0.90) + 1,
        "Maximum quality": np.argmax(cumulative_variance >= 0.95) + 1
    }
    
    print("🎯 Recommended dimensions for different priorities:")
    for use_case, dims in recommendations.items():
        if dims <= len(cumulative_variance):
            variance_retained = cumulative_variance[dims-1] * 100
            print(f"   {use_case:20s}: {dims:3d}D (retains {variance_retained:5.1f}% of information)")
    
    # Performance vs quality analysis
    print(f"\n⚖️  Trade-off Analysis:")
    print(f"   Current 20D: {cumulative_variance[19]*100:5.1f}% info, FAST")
    
    balanced_dims = recommendations["Balanced performance"]
    if balanced_dims <= len(cumulative_variance):
        print(f"   Recommended {balanced_dims}D: {cumulative_variance[balanced_dims-1]*100:5.1f}% info, still fast")
    
    return recommendations

def visualize_analysis(explained_variance_ratio, cumulative_variance):
    """Create visualizations of the analysis"""
    print("\n📊 Creating visualizations...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scree plot - individual component variance
    ax1.plot(range(1, min(101, len(explained_variance_ratio)+1)), 
             explained_variance_ratio[:100], 'b-', alpha=0.7)
    ax1.set_title('Scree Plot - Individual Component Variance')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative variance plot
    dims_to_show = min(300, len(cumulative_variance))
    ax2.plot(range(1, dims_to_show+1), cumulative_variance[:dims_to_show], 'g-', linewidth=2)
    ax2.axhline(y=0.8, color='orange', linestyle='--', label='80% variance')
    ax2.axhline(y=0.9, color='red', linestyle='--', label='90% variance')
    ax2.axvline(x=20, color='purple', linestyle=':', label='Current (20D)')
    ax2.set_title('Cumulative Variance Explained')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Zoom on first 100 components
    ax3.plot(range(1, min(101, len(cumulative_variance)+1)), 
             cumulative_variance[:100], 'r-', linewidth=2)
    ax3.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7)
    ax3.axhline(y=0.9, color='red', linestyle='--', alpha=0.7)
    ax3.axvline(x=20, color='purple', linestyle=':', alpha=0.7)
    ax3.set_title('Cumulative Variance - First 100 Components')
    ax3.set_xlabel('Number of Components')
    ax3.set_ylabel('Cumulative Variance Ratio')
    ax3.grid(True, alpha=0.3)
    
    # 4. Information loss vs dimensions
    dims_range = range(10, min(201, len(cumulative_variance)+1), 10)
    info_loss = [1 - cumulative_variance[d-1] for d in dims_range]
    ax4.plot(dims_range, info_loss, 'orange', marker='o', linewidth=2)
    ax4.axvline(x=20, color='purple', linestyle=':', label='Current (20D)')
    ax4.set_title('Information Loss vs Dimensions')
    ax4.set_xlabel('Number of Dimensions')
    ax4.set_ylabel('Information Loss (1 - Variance Retained)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('embedding_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved as 'embedding_analysis.png'")
    
    return fig

def main():
    """Main analysis function"""
    print("🔍 EMBEDDING DIMENSIONALITY ANALYSIS")
    print("=" * 60)
    
    # Load data
    embeddings = load_embeddings()
    
    # Analyze density and sparsity
    mean_vals, std_vals, variance_sorted = analyze_density_and_sparsity(embeddings)
    
    # PCA analysis
    pca, explained_variance_ratio, cumulative_variance = analyze_pca_information_retention(embeddings)
    
    # Recommendations
    recommendations = recommend_optimal_dimensions(cumulative_variance)
    
    # Visualizations
    try:
        fig = visualize_analysis(explained_variance_ratio, cumulative_variance)
        print("\n📊 Check 'embedding_analysis.png' for detailed visualizations")
    except ImportError:
        print("\n⚠️  Matplotlib not available for visualizations")
    
    print(f"\n🎉 Analysis complete!")
    print(f"Current setting (20D) retains {cumulative_variance[19]*100:.1f}% of information")
    
    return recommendations

if __name__ == "__main__":
    main() 