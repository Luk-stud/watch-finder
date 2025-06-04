#!/usr/bin/env python3
"""
Interactive Embedding Visualizer
================================

Visualize watch embeddings using interactive plots with:
- t-SNE and UMAP dimensionality reduction
- Color coding by brand, price, cluster
- Interactive hover information
- 2D and 3D views
- Dynamic filtering and exploration
"""

import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import colorsys
import argparse
from typing import Dict, List, Tuple, Optional

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    print("⚠️  UMAP not available. Install with: pip install umap-learn")
    UMAP_AVAILABLE = False

class EmbeddingVisualizer:
    def __init__(self):
        self.load_data()
        self.df = None
        
    def load_data(self):
        """Load watch data and embeddings"""
        print("Loading watch data...")
        
        with open('watch_text_embeddings.pkl', 'rb') as f:
            self.embeddings = pickle.load(f)
        with open('watch_text_metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"✅ Loaded {len(self.metadata)} watches with {self.embeddings.shape[1]}D embeddings")
        
    def prepare_dataframe(self):
        """Prepare pandas DataFrame with watch information"""
        print("📊 Preparing data for visualization...")
        
        # Extract key information
        data = []
        for i, watch in enumerate(self.metadata):
            # Extract price as numeric value
            price = self._extract_price(watch.get('price', ''))
            
            # Create price category
            if price == 0:
                price_category = 'Unknown'
            elif price < 500:
                price_category = 'Budget (<$500)'
            elif price < 2000:
                price_category = 'Mid-range ($500-$2K)'
            elif price < 5000:
                price_category = 'Premium ($2K-$5K)'
            else:
                price_category = 'Luxury ($5K+)'
            
            data.append({
                'watch_id': i,
                'brand': watch.get('brand', 'Unknown'),
                'model': watch.get('model', 'Unknown'),
                'price': price,
                'price_category': price_category,
                'description': watch.get('description', '')[:100] + '...' if watch.get('description', '') else '',
                'source': watch.get('source', 'Unknown'),
                'hover_text': f"{watch.get('brand', 'Unknown')} {watch.get('model', 'Unknown')}<br>" +
                             f"Price: ${price:,.0f}<br>" +
                             f"Source: {watch.get('source', 'Unknown')}"
            })
        
        self.df = pd.DataFrame(data)
        print(f"✅ DataFrame prepared with {len(self.df)} watches")
        
    def _extract_price(self, price_str: str) -> float:
        """Extract numeric price from string"""
        if not price_str:
            return 0.0
        
        # Remove currency symbols and commas
        price_clean = str(price_str).replace('$', '').replace(',', '').replace('€', '').replace('£', '')
        
        try:
            return float(price_clean)
        except ValueError:
            return 0.0
    
    def add_clusters(self, n_clusters: int = 8):
        """Add K-means cluster information"""
        print(f"🎯 Computing {n_clusters} clusters...")
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.embeddings)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_scaled)
        
        self.df['cluster'] = cluster_labels
        self.df['cluster_name'] = [f'Cluster {i}' for i in cluster_labels]
        
        # Analyze clusters
        print("📈 Cluster analysis:")
        for i in range(n_clusters):
            cluster_df = self.df[self.df['cluster'] == i]
            top_brands = cluster_df['brand'].value_counts().head(3)
            avg_price = cluster_df[cluster_df['price'] > 0]['price'].mean()
            
            print(f"   Cluster {i}: {len(cluster_df)} watches, avg price: ${avg_price:,.0f}")
            print(f"      Top brands: {list(top_brands.index)}")
        
    def compute_tsne(self, perplexity: int = 30, n_components: int = 2) -> np.ndarray:
        """Compute t-SNE embedding"""
        print(f"🧮 Computing t-SNE ({n_components}D, perplexity={perplexity})...")
        
        # Standardize first
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.embeddings)
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            n_iter=1000,
            verbose=1
        )
        
        return tsne.fit_transform(embeddings_scaled)
    
    def compute_umap(self, n_neighbors: int = 15, min_dist: float = 0.1, n_components: int = 2) -> np.ndarray:
        """Compute UMAP embedding"""
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        
        print(f"🗺️  Computing UMAP ({n_components}D, neighbors={n_neighbors}, min_dist={min_dist})...")
        
        # Standardize first
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.embeddings)
        
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=42,
            verbose=True
        )
        
        return reducer.fit_transform(embeddings_scaled)
    
    def create_2d_plot(self, coords: np.ndarray, color_by: str = 'brand', title: str = "Watch Embeddings") -> go.Figure:
        """Create interactive 2D scatter plot"""
        print(f"📊 Creating 2D plot colored by {color_by}...")
        
        # Add coordinates to dataframe
        plot_df = self.df.copy()
        plot_df['x'] = coords[:, 0]
        plot_df['y'] = coords[:, 1]
        
        # Create color mapping
        if color_by == 'brand':
            # Use top brands for color, rest as 'Other'
            top_brands = plot_df['brand'].value_counts().head(10).index
            plot_df['color_var'] = plot_df['brand'].apply(lambda x: x if x in top_brands else 'Other')
            color_discrete_map = None
        elif color_by == 'cluster':
            plot_df['color_var'] = plot_df['cluster_name']
            color_discrete_map = None
        elif color_by == 'price_category':
            plot_df['color_var'] = plot_df['price_category']
            color_discrete_map = {
                'Budget (<$500)': '#2E8B57',
                'Mid-range ($500-$2K)': '#4169E1', 
                'Premium ($2K-$5K)': '#FF8C00',
                'Luxury ($5K+)': '#DC143C',
                'Unknown': '#808080'
            }
        else:
            plot_df['color_var'] = plot_df[color_by]
            color_discrete_map = None
        
        # Create plot
        fig = px.scatter(
            plot_df,
            x='x', y='y',
            color='color_var',
            hover_name='brand',
            hover_data={
                'model': True,
                'price': ':$,.0f',
                'x': False,
                'y': False,
                'color_var': False
            },
            title=title,
            color_discrete_map=color_discrete_map,
            width=1000, height=800
        )
        
        # Customize layout
        fig.update_layout(
            title_font_size=16,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            legend_title=color_by.replace('_', ' ').title(),
            hovermode='closest'
        )
        
        return fig
    
    def create_3d_plot(self, coords: np.ndarray, color_by: str = 'brand', title: str = "Watch Embeddings 3D") -> go.Figure:
        """Create interactive 3D scatter plot"""
        print(f"📊 Creating 3D plot colored by {color_by}...")
        
        # Add coordinates to dataframe
        plot_df = self.df.copy()
        plot_df['x'] = coords[:, 0]
        plot_df['y'] = coords[:, 1]
        plot_df['z'] = coords[:, 2]
        
        # Create color mapping (similar to 2D)
        if color_by == 'brand':
            top_brands = plot_df['brand'].value_counts().head(10).index
            plot_df['color_var'] = plot_df['brand'].apply(lambda x: x if x in top_brands else 'Other')
        elif color_by == 'cluster':
            plot_df['color_var'] = plot_df['cluster_name']
        elif color_by == 'price_category':
            plot_df['color_var'] = plot_df['price_category']
        else:
            plot_df['color_var'] = plot_df[color_by]
        
        # Create 3D scatter
        fig = px.scatter_3d(
            plot_df,
            x='x', y='y', z='z',
            color='color_var',
            hover_name='brand',
            hover_data={
                'model': True,
                'price': ':$,.0f',
                'x': False,
                'y': False,
                'z': False,
                'color_var': False
            },
            title=title,
            width=1000, height=800
        )
        
        # Customize layout
        fig.update_layout(
            title_font_size=16,
            scene=dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                zaxis_title="Dimension 3"
            ),
            legend_title=color_by.replace('_', ' ').title()
        )
        
        return fig
    
    def create_comparison_dashboard(self):
        """Create a dashboard comparing different reduction methods"""
        print("🚀 Creating comparison dashboard...")
        
        # Compute both t-SNE and UMAP (if available)
        tsne_2d = self.compute_tsne(n_components=2)
        
        if UMAP_AVAILABLE:
            umap_2d = self.compute_umap(n_components=2)
        
        # Create subplots
        if UMAP_AVAILABLE:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['t-SNE (Brand)', 't-SNE (Price)', 'UMAP (Brand)', 'UMAP (Price)'],
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
        else:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['t-SNE (Brand)', 't-SNE (Price)'],
                specs=[[{"type": "scatter"}, {"type": "scatter"}]]
            )
        
        # Add t-SNE plots
        self._add_scatter_to_subplot(fig, tsne_2d, 'brand', 1, 1)
        self._add_scatter_to_subplot(fig, tsne_2d, 'price_category', 1, 2)
        
        # Add UMAP plots if available
        if UMAP_AVAILABLE:
            self._add_scatter_to_subplot(fig, umap_2d, 'brand', 2, 1)
            self._add_scatter_to_subplot(fig, umap_2d, 'price_category', 2, 2)
        
        fig.update_layout(
            height=800 if UMAP_AVAILABLE else 400,
            title_text="Watch Embedding Visualization Comparison",
            showlegend=False
        )
        
        return fig
    
    def _add_scatter_to_subplot(self, fig, coords: np.ndarray, color_by: str, row: int, col: int):
        """Add scatter plot to subplot"""
        plot_df = self.df.copy()
        plot_df['x'] = coords[:, 0]
        plot_df['y'] = coords[:, 1]
        
        if color_by == 'brand':
            top_brands = plot_df['brand'].value_counts().head(8).index
            plot_df['color_var'] = plot_df['brand'].apply(lambda x: x if x in top_brands else 'Other')
        else:
            plot_df['color_var'] = plot_df[color_by]
        
        # Add traces for each category
        for category in plot_df['color_var'].unique():
            category_data = plot_df[plot_df['color_var'] == category]
            
            fig.add_trace(
                go.Scatter(
                    x=category_data['x'],
                    y=category_data['y'],
                    mode='markers',
                    name=category,
                    text=category_data['hover_text'],
                    hovertemplate='%{text}<extra></extra>',
                    marker=dict(size=4, opacity=0.7),
                    showlegend=False
                ),
                row=row, col=col
            )
    
    def save_plot(self, fig: go.Figure, filename: str):
        """Save plot as interactive HTML"""
        filepath = f"{filename}.html"
        fig.write_html(filepath)
        print(f"✅ Plot saved as {filepath}")
        print(f"🌐 Open in browser: file://{filepath}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Visualize watch embeddings interactively')
    parser.add_argument('--method', choices=['tsne', 'umap', 'both'], default='both',
                       help='Dimensionality reduction method')
    parser.add_argument('--dimensions', choices=[2, 3], type=int, default=2,
                       help='Number of dimensions for visualization')
    parser.add_argument('--color', choices=['brand', 'price_category', 'cluster'], default='brand',
                       help='Variable to color points by')
    parser.add_argument('--clusters', type=int, default=8,
                       help='Number of clusters for K-means')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--output', default='watch_embeddings_viz',
                       help='Output filename (without extension)')
    
    args = parser.parse_args()
    
    print("🎨 INTERACTIVE WATCH EMBEDDING VISUALIZER")
    print("=" * 50)
    
    # Initialize visualizer
    viz = EmbeddingVisualizer()
    viz.prepare_dataframe()
    viz.add_clusters(n_clusters=args.clusters)
    
    # Create visualizations based on method
    if args.method in ['tsne', 'both']:
        print(f"\n📊 Creating t-SNE visualization...")
        tsne_coords = viz.compute_tsne(perplexity=args.perplexity, n_components=args.dimensions)
        
        if args.dimensions == 2:
            tsne_fig = viz.create_2d_plot(tsne_coords, color_by=args.color, title="t-SNE: Watch Embeddings")
        else:
            tsne_fig = viz.create_3d_plot(tsne_coords, color_by=args.color, title="t-SNE: Watch Embeddings 3D")
        
        viz.save_plot(tsne_fig, f"{args.output}_tsne_{args.dimensions}d")
    
    if args.method in ['umap', 'both'] and UMAP_AVAILABLE:
        print(f"\n🗺️  Creating UMAP visualization...")
        umap_coords = viz.compute_umap(n_components=args.dimensions)
        
        if args.dimensions == 2:
            umap_fig = viz.create_2d_plot(umap_coords, color_by=args.color, title="UMAP: Watch Embeddings")
        else:
            umap_fig = viz.create_3d_plot(umap_coords, color_by=args.color, title="UMAP: Watch Embeddings 3D")
        
        viz.save_plot(umap_fig, f"{args.output}_umap_{args.dimensions}d")
    
    # Create comparison dashboard
    if args.method == 'both' and args.dimensions == 2:
        print(f"\n📈 Creating comparison dashboard...")
        comparison_fig = viz.create_comparison_dashboard()
        viz.save_plot(comparison_fig, f"{args.output}_comparison")
    
    print(f"\n🎉 Visualization complete!")
    if not UMAP_AVAILABLE:
        print(f"💡 Install UMAP for additional visualization options: pip install umap-learn")

if __name__ == "__main__":
    main() 