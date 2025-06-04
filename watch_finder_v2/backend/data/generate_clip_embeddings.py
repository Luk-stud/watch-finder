"""
Generate CLIP Embeddings for Watch Images
=========================================

This script loads the existing watch metadata and generates CLIP embeddings
from the watch images for visual-based recommendations.
"""

import os
import pickle
import numpy as np
import requests
from PIL import Image
import torch
import clip
from tqdm import tqdm
import time
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_existing_data():
    """Load existing watch metadata."""
    try:
        with open('watch_text_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        logger.info(f"✅ Loaded {len(metadata)} watch metadata entries")
        return metadata
    except Exception as e:
        logger.error(f"❌ Failed to load metadata: {e}")
        return []

def download_and_process_image(image_url, max_retries=3):
    """Download and process image for CLIP."""
    for attempt in range(max_retries):
        try:
            # Add headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(image_url, headers=headers, timeout=10)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                return image
            else:
                logger.warning(f"HTTP {response.status_code} for {image_url}")
                
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {image_url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
    
    return None

def generate_clip_embeddings(metadata, device, model, preprocess):
    """Generate CLIP embeddings for all watch images."""
    clip_embeddings = []
    successful_indices = []
    
    logger.info(f"🚀 Starting CLIP embedding generation for {len(metadata)} watches...")
    
    for idx, watch_data in enumerate(tqdm(metadata, desc="Processing watches")):
        try:
            # Get image URL
            image_url = watch_data.get('image_url')
            if not image_url:
                logger.warning(f"No image URL for watch {idx}")
                # Use zero embedding for missing images
                clip_embeddings.append(np.zeros(512))  # CLIP ViT-B/32 is 512 dimensions
                continue
            
            # Download and process image
            image = download_and_process_image(image_url)
            if image is None:
                logger.warning(f"Failed to download image for watch {idx}: {image_url}")
                # Use zero embedding for failed downloads
                clip_embeddings.append(np.zeros(512))
                continue
            
            # Preprocess image for CLIP
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            # Generate CLIP embedding
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                # Normalize the features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            # Convert to numpy and store
            embedding = image_features.cpu().numpy().flatten()
            clip_embeddings.append(embedding)
            successful_indices.append(idx)
            
            # Small delay to be respectful to servers
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error processing watch {idx}: {e}")
            # Use zero embedding for errors
            clip_embeddings.append(np.zeros(512))
    
    logger.info(f"✅ Successfully generated {len(successful_indices)} CLIP embeddings")
    return np.array(clip_embeddings)

def main():
    """Main function to generate and save CLIP embeddings."""
    logger.info("🎯 Starting CLIP embedding generation...")
    
    # Load existing metadata
    metadata = load_existing_data()
    if not metadata:
        logger.error("No metadata found, exiting...")
        return
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🔧 Using device: {device}")
    
    # Load CLIP model
    logger.info("📥 Loading CLIP model...")
    try:
        model, preprocess = clip.load("ViT-B/32", device=device)
        logger.info("✅ CLIP model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load CLIP model: {e}")
        logger.info("💡 Install CLIP with: pip install git+https://github.com/openai/CLIP.git")
        return
    
    # Generate embeddings
    clip_embeddings = generate_clip_embeddings(metadata, device, model, preprocess)
    
    # Save embeddings
    try:
        output_file = 'watch_clip_embeddings.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(clip_embeddings, f)
        logger.info(f"✅ Saved CLIP embeddings to {output_file}")
        logger.info(f"📊 Shape: {clip_embeddings.shape}")
        
        # Save a summary
        summary = {
            'total_watches': len(metadata),
            'embedding_shape': clip_embeddings.shape,
            'successful_embeddings': np.count_nonzero(np.any(clip_embeddings != 0, axis=1)),
            'failed_embeddings': np.count_nonzero(np.all(clip_embeddings == 0, axis=1))
        }
        
        with open('clip_embedding_summary.pkl', 'wb') as f:
            pickle.dump(summary, f)
        
        logger.info(f"📈 Summary: {summary}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save embeddings: {e}")

if __name__ == "__main__":
    main() 