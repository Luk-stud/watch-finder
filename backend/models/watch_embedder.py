import json
import numpy as np
import torch
from typing import List, Dict, Any
import pickle
import os
import requests
from PIL import Image
from io import BytesIO
import time

# Try to import CLIP - will need to be installed
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not available. Please install it with: pip install git+https://github.com/openai/CLIP.git")

class WatchEmbedder:
    def __init__(self, model_path=None, device=None):
        """
        Initialize the watch embedder with a CLIP model for image-based embeddings.
        
        Args:
            model_path: Path to your tuned CLIP model, or None for default CLIP
            device: torch device to use
        """
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP is not available. Please install it first.")
            
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path and os.path.exists(model_path):
            # Load your custom tuned model
            print(f"Loading custom CLIP model from {model_path}")
            try:
                # First load the base CLIP model to get the architecture and preprocessing
                base_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
                
                # Load the custom model state dict
                custom_state_dict = torch.load(model_path, map_location=self.device)
                
                # If the saved model is a complete model, use it directly
                if hasattr(custom_state_dict, 'encode_image'):
                    self.model = custom_state_dict
                # If it's a state dict, load it into the base model
                elif isinstance(custom_state_dict, dict):
                    base_model.load_state_dict(custom_state_dict)
                    self.model = base_model
                else:
                    # Fallback: assume it's a model object
                    self.model = custom_state_dict
                
                # Ensure model is in evaluation mode and on correct device
                self.model.eval()
                self.model = self.model.to(self.device)
                
                print(f"Successfully loaded custom CLIP model on {self.device}")
                
            except Exception as e:
                print(f"Error loading custom model {model_path}: {e}")
                print("Falling back to default CLIP model...")
                self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        else:
            # Load default CLIP model
            print("Loading default CLIP ViT-B/32 model")
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        self.model_path = model_path
        self.embeddings = None
        self.watch_data = None
        
    def download_image(self, url: str, timeout: int = 10) -> Image.Image:
        """
        Download image from URL and return PIL Image.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
            return None
    
    def generate_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Generate CLIP embedding for a single image.
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                # Normalize the features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            return image_features.cpu().numpy().squeeze()
        except Exception as e:
            print(f"Error generating embedding for image: {e}")
            return None

    def generate_embeddings(self, watch_data_path: str) -> np.ndarray:
        """
        Generate CLIP image embeddings for all watches in the dataset.
        """
        # Load watch data
        with open(watch_data_path, 'r') as f:
            self.watch_data = json.load(f)
            
        # Add indices to watch data
        for i, watch in enumerate(self.watch_data):
            watch['index'] = i
            
        embeddings_list = []
        successful_indices = []
        
        print(f"Generating image embeddings for {len(self.watch_data)} watches...")
        
        for i, watch in enumerate(self.watch_data):
            if i % 50 == 0:
                print(f"Processing watch {i+1}/{len(self.watch_data)}")
                
            image_url = watch.get('image_url')
            if not image_url:
                print(f"No image URL for watch {i}")
                continue
                
            # Download and process image
            image = self.download_image(image_url)
            if image is None:
                print(f"Failed to download image for watch {i}: {image_url}")
                continue
                
            # Generate embedding
            embedding = self.generate_image_embedding(image)
            if embedding is None:
                print(f"Failed to generate embedding for watch {i}")
                continue
                
            embeddings_list.append(embedding)
            successful_indices.append(i)
            
            # Small delay to be respectful to the server
            time.sleep(0.1)
        
        # Filter watch data to only include watches with successful embeddings
        self.watch_data = [self.watch_data[i] for i in successful_indices]
        
        # Update indices
        for i, watch in enumerate(self.watch_data):
            watch['index'] = i
            
        # Convert to numpy array
        self.embeddings = np.array(embeddings_list)
        
        print(f"Successfully generated embeddings for {len(self.watch_data)} watches")
        print(f"Embedding shape: {self.embeddings.shape}")
        
        return self.embeddings
    
    def save_embeddings(self, embeddings_path: str):
        """
        Save embeddings and associated data to disk.
        """
        data_to_save = {
            'embeddings': self.embeddings,
            'watch_data': self.watch_data,
            'model_path': self.model_path,
            'embedding_type': 'clip_image'
        }
        
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        with open(embeddings_path, 'wb') as f:
            pickle.dump(data_to_save, f)
            
        print(f"CLIP image embeddings saved to {embeddings_path}")
    
    def load_embeddings(self, embeddings_path: str):
        """
        Load embeddings and associated data from disk.
        """
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            
        self.embeddings = data['embeddings']
        self.watch_data = data['watch_data']
        
        print(f"CLIP image embeddings loaded from {embeddings_path}")
        print(f"Loaded {len(self.watch_data)} watches with embeddings shape: {self.embeddings.shape}")
        return self.embeddings, self.watch_data

if __name__ == "__main__":
    # Example usage
    embedder = WatchEmbedder()
    
    # Generate embeddings
    data_path = "../data/raw/watch_data.json"
    embeddings = embedder.generate_embeddings(data_path)
    
    # Save embeddings
    embeddings_path = "../embeddings/watch_image_embeddings.pkl"
    embedder.save_embeddings(embeddings_path)
    
    print(f"Generated CLIP image embeddings with shape: {embeddings.shape}") 