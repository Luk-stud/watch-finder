import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def load_watch_data(data_path):
    """Load watch data from JSON file."""
    with open(data_path, 'r') as f:
        return json.load(f)

def create_watch_description(watch):
    """Create a text description of the watch for embedding."""
    description = f"{watch['brand']} {watch['model_name']} - "
    description += f"Price: {watch['price']}. "
    if watch.get('description'):
        description += watch['description']
    return description

def generate_embeddings(data_path, output_path):
    """Generate embeddings for watch data."""
    print("Loading watch data...")
    watch_data = load_watch_data(data_path)
    
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating descriptions...")
    descriptions = [create_watch_description(watch) for watch in watch_data]
    
    print("Generating embeddings...")
    embeddings = model.encode(descriptions, show_progress_bar=True)
    
    print("Saving embeddings...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'watch_data': watch_data
        }, f)
    
    print(f"Generated embeddings for {len(watch_data)} watches")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved to: {output_path}")

if __name__ == '__main__':
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths
    data_path = os.path.join(project_root, 'data', 'raw', 'watch_data.json')
    output_path = os.path.join(project_root, 'embeddings', 'watch_embeddings.pkl')
    
    # Generate embeddings
    generate_embeddings(data_path, output_path) 