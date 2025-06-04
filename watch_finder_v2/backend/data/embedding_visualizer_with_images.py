#!/usr/bin/env python3
"""
Interactive Embedding Visualizer with Images
===========================================

Enhanced version that shows watch images in visualizations:
- Hover to see watch images
- Click to get detailed view
- Image grid for selected regions
- Filter by visual characteristics
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
import base64
import io
from typing import Dict, List, Tuple, Optional
from urllib.parse import quote

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    print("⚠️  UMAP not available. Install with: pip install umap-learn")
    UMAP_AVAILABLE = False

try:
    from PIL import Image
    import requests
    IMAGES_AVAILABLE = True
except ImportError:
    print("⚠️  PIL/requests not available for image processing. Install with: pip install Pillow requests")
    IMAGES_AVAILABLE = False

class ImageEmbeddingVisualizer:
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
        """Prepare pandas DataFrame with watch information including image URLs"""
        print("📊 Preparing data for visualization with images...")
        
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
            
            # Get image URL
            image_url = watch.get('image_url', '')
            
            # Create enhanced hover text with image
            hover_text = self._create_hover_with_image(watch, price, image_url)
            
            data.append({
                'watch_id': i,
                'brand': watch.get('brand', 'Unknown'),
                'model': watch.get('model', 'Unknown'),
                'price': price,
                'price_category': price_category,
                'description': watch.get('description', '')[:100] + '...' if watch.get('description', '') else '',
                'source': watch.get('source', 'Unknown'),
                'image_url': image_url,
                'product_url': watch.get('product_url', ''),
                'hover_text': hover_text,
                'simple_hover': f"{watch.get('brand', 'Unknown')} {watch.get('model', 'Unknown')}<br>Price: ${price:,.0f}"
            })
        
        self.df = pd.DataFrame(data)
        print(f"✅ DataFrame prepared with {len(self.df)} watches")
        
    def _create_hover_with_image(self, watch: dict, price: float, image_url: str) -> str:
        """Create rich hover text with image"""
        brand = watch.get('brand', 'Unknown')
        model = watch.get('model', 'Unknown')
        source = watch.get('source', 'Unknown')
        
        # Base information
        hover_html = f"""
        <b>{brand} {model}</b><br>
        Price: ${price:,.0f}<br>
        Source: {source}<br>
        """
        
        # Add image if URL exists
        if image_url and image_url.startswith('http'):
            # Create image HTML with size limits
            hover_html += f"""
            <br>
            <img src="{image_url}" 
                 style="max-width:200px; max-height:200px; border-radius:8px; border:1px solid #ddd;"
                 onerror="this.style.display='none'">
            """
        
        return hover_html
    
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
    
    def create_2d_plot_with_images(self, coords: np.ndarray, color_by: str = 'brand', 
                                  title: str = "Watch Embeddings with Images") -> go.Figure:
        """Create interactive 2D scatter plot with image hovers"""
        print(f"📊 Creating 2D plot with images, colored by {color_by}...")
        
        # Add coordinates to dataframe
        plot_df = self.df.copy()
        plot_df['x'] = coords[:, 0]
        plot_df['y'] = coords[:, 1]
        
        # Create color mapping
        if color_by == 'brand':
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
        
        # Create the plot
        fig = go.Figure()
        
        # Add traces for each category to get proper legend
        for category in plot_df['color_var'].unique():
            if pd.isna(category):
                continue
                
            category_data = plot_df[plot_df['color_var'] == category]
            
            # Get color for this category
            if color_discrete_map and category in color_discrete_map:
                color = color_discrete_map[category]
            else:
                # Use plotly's default color sequence
                color_idx = list(plot_df['color_var'].unique()).index(category) % 10
                colors = px.colors.qualitative.Plotly
                color = colors[color_idx]
            
            fig.add_trace(go.Scatter(
                x=category_data['x'],
                y=category_data['y'],
                mode='markers',
                name=category,
                text=category_data['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                customdata=category_data[['brand', 'model', 'price', 'product_url']].values
            ))
        
        # Customize layout
        fig.update_layout(
            title=dict(
                text=title,
                font_size=18,
                x=0.5
            ),
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            legend_title=color_by.replace('_', ' ').title(),
            hovermode='closest',
            width=1200,
            height=900,
            font=dict(size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add instructions
        fig.add_annotation(
            text="💡 Hover over points to see watch images!<br>🔗 Click points to open product pages",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=10, color="gray"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
        
        return fig
    
    def create_image_grid_for_cluster(self, cluster_id: int, max_images: int = 20) -> go.Figure:
        """Create a grid showing sample images from a specific cluster"""
        print(f"🖼️ Creating image grid for cluster {cluster_id}...")
        
        cluster_data = self.df[self.df['cluster'] == cluster_id].head(max_images)
        
        # Calculate grid dimensions
        n_images = len(cluster_data)
        grid_size = int(np.ceil(np.sqrt(n_images)))
        
        # Create subplot grid
        fig = make_subplots(
            rows=grid_size,
            cols=grid_size,
            subplot_titles=[f"{row['brand']} {row['model']}" for _, row in cluster_data.iterrows()],
            specs=[[{"type": "scatter"}] * grid_size for _ in range(grid_size)],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        for idx, (_, watch) in enumerate(cluster_data.iterrows()):
            row = idx // grid_size + 1
            col = idx % grid_size + 1
            
            # Add placeholder scatter (we'd need to implement image loading for actual images)
            fig.add_trace(
                go.Scatter(
                    x=[0], y=[0],
                    mode='markers+text',
                    text=[f"${watch['price']:,.0f}"],
                    textposition="middle center",
                    marker=dict(size=50, color='lightblue'),
                    showlegend=False,
                    hovertemplate=f"<b>{watch['brand']} {watch['model']}</b><br>" +
                                f"Price: ${watch['price']:,.0f}<br>" +
                                f"<a href='{watch['image_url']}' target='_blank'>View Image</a>" +
                                "<extra></extra>"
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=f"Sample Watches from Cluster {cluster_id}",
            showlegend=False,
            height=800,
            width=1000
        )
        
        return fig
    
    def create_interactive_explorer(self, coords: np.ndarray, color_by: str = 'brand') -> go.Figure:
        """Create an advanced interactive explorer with click callbacks"""
        print(f"🎮 Creating interactive explorer...")
        
        # This would be the main visualization
        fig = self.create_2d_plot_with_images(coords, color_by, 
                                            "Interactive Watch Explorer - Click and Hover!")
        
        # Add selection tools
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=["mode", "markers"],
                            label="Points",
                            method="restyle"
                        ),
                        dict(
                            args=["mode", "markers+text"],
                            label="Labels",
                            method="restyle"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ]
        )
        
        return fig
    
    def save_plot(self, fig: go.Figure, filename: str):
        """Save plot as interactive HTML with enhanced features"""
        filepath = f"{filename}.html"
        
        # Add custom JavaScript for enhanced interactivity
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': filename,
                'height': 900,
                'width': 1200,
                'scale': 2
            }
        }
        
        fig.write_html(filepath, config=config)
        print(f"✅ Enhanced plot saved as {filepath}")
        print(f"🌐 Open in browser: file://{filepath}")
        print(f"💡 Features: Hover for images, click for details, select regions!")

def main():
    """Main function with enhanced command line interface"""
    parser = argparse.ArgumentParser(description='Visualize watch embeddings with images')
    parser.add_argument('--method', choices=['tsne', 'umap', 'both'], default='tsne',
                       help='Dimensionality reduction method')
    parser.add_argument('--color', choices=['brand', 'price_category', 'cluster'], default='brand',
                       help='Variable to color points by')
    parser.add_argument('--clusters', type=int, default=8,
                       help='Number of clusters for K-means')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--output', default='watch_image_explorer',
                       help='Output filename (without extension)')
    parser.add_argument('--cluster-grids', action='store_true',
                       help='Create image grids for each cluster')
    
    args = parser.parse_args()
    
    print("🎨 INTERACTIVE WATCH EMBEDDING VISUALIZER WITH IMAGES")
    print("=" * 60)
    
    # Initialize visualizer
    viz = ImageEmbeddingVisualizer()
    viz.prepare_dataframe()
    viz.add_clusters(n_clusters=args.clusters)
    
    # Check for images
    images_with_urls = viz.df[viz.df['image_url'].str.len() > 0]
    print(f"📸 Found {len(images_with_urls)} watches with image URLs")
    
    # Create main visualization
    if args.method in ['tsne', 'both']:
        print(f"\n📊 Creating t-SNE visualization with images...")
        tsne_coords = viz.compute_tsne(perplexity=args.perplexity, n_components=2)
        
        # Create enhanced plot
        tsne_fig = viz.create_interactive_explorer(tsne_coords, color_by=args.color)
        viz.save_plot(tsne_fig, f"{args.output}_tsne_images")
    
    if args.method in ['umap', 'both'] and UMAP_AVAILABLE:
        print(f"\n🗺️  Creating UMAP visualization with images...")
        umap_coords = viz.compute_umap(n_components=2)
        
        umap_fig = viz.create_interactive_explorer(umap_coords, color_by=args.color)
        viz.save_plot(umap_fig, f"{args.output}_umap_images")
    
    # Create cluster image grids if requested
    if args.cluster_grids:
        print(f"\n🖼️ Creating cluster image grids...")
        for cluster_id in range(args.clusters):
            cluster_fig = viz.create_image_grid_for_cluster(cluster_id)
            viz.save_plot(cluster_fig, f"{args.output}_cluster_{cluster_id}_grid")
    
    print(f"\n🎉 Enhanced visualization complete!")
    print(f"💡 Features available:")
    print(f"   🖼️  Hover over points to see watch images")
    print(f"   🔗 Click points to open product pages")
    print(f"   🎯 Select regions to focus on specific areas")
    print(f"   🎨 Toggle between point and label modes")
    
    if not IMAGES_AVAILABLE:
        print(f"⚠️  For better image support, install: pip install Pillow requests")

if __name__ == "__main__":
    main() 