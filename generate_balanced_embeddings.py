#!/usr/bin/env python3
"""
Generate Balanced Embeddings for Model Comparison

Creates a balanced dataset with 10 watches per manufacturer from the latest scrape,
then generates embeddings for multiple models (DINO, CLIP ViT-B/32, CLIP ViT-L/14)
for fair comparison.
"""

import os
import json
import pickle
import numpy as np
from collections import defaultdict
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import timm
from timm.models.vision_transformer import VisionTransformer
import warnings
warnings.filterwarnings('ignore')

class BalancedEmbeddingGenerator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.models = {}
        self.processors = {}
        
    def load_models(self):
        """Load all models for embedding generation."""
        print("🔄 Loading models...")
        
        # Load CLIP ViT-B/32
        try:
            print("  Loading CLIP ViT-B/32...")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.models['clip_vit_b32'] = clip_model.to(self.device)
            self.processors['clip_vit_b32'] = clip_processor
            print("  ✅ CLIP ViT-B/32 loaded")
        except Exception as e:
            print(f"  ❌ Failed to load CLIP ViT-B/32: {e}")
        
        # Load CLIP ViT-L/14
        try:
            print("  Loading CLIP ViT-L/14...")
            clip_l_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            clip_l_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.models['clip_vit_l14'] = clip_l_model.to(self.device)
            self.processors['clip_vit_l14'] = clip_l_processor
            print("  ✅ CLIP ViT-L/14 loaded")
        except Exception as e:
            print(f"  ❌ Failed to load CLIP ViT-L/14: {e}")
        
        # Load DINO ViT
        try:
            print("  Loading DINO ViT...")
            dino_model = timm.create_model('vit_base_patch16_224_dino', pretrained=True, num_classes=0)
            dino_model = dino_model.to(self.device)
            self.models['dino_vit'] = dino_model
            print("  ✅ DINO ViT loaded")
        except Exception as e:
            print(f"  ❌ Failed to load DINO ViT: {e}")
        
        print(f"✅ Loaded {len(self.models)} models")
    
    def load_scrape_data(self, filepath: str = 'production_scrape_20250601_175426/data/final_scrape.json'):
        """Load the final scrape data."""
        print(f"📊 Loading scrape data from {filepath}...")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {len(data)} watches")
        return data
    
    def create_balanced_subset(self, data, watches_per_brand: int = 10):
        """Create a balanced subset with specified number of watches per brand."""
        print(f"⚖️ Creating balanced subset ({watches_per_brand} watches per brand)...")
        
        # Group watches by brand
        brand_groups = defaultdict(list)
        for watch in data:
            brand = watch.get('specs', {}).get('brand', 'Unknown')
            if brand != 'Unknown':
                brand_groups[brand].append(watch)
        
        # Select watches per brand
        balanced_data = []
        for brand, watches in brand_groups.items():
            # Take up to watches_per_brand watches from each brand
            selected_watches = watches[:watches_per_brand]
            balanced_data.extend(selected_watches)
            print(f"  {brand}: {len(selected_watches)} watches")
        
        print(f"✅ Created balanced subset with {len(balanced_data)} watches from {len(brand_groups)} brands")
        return balanced_data
    
    def get_image_path(self, watch_data):
        """Get the image path for a watch."""
        # Check if image exists in the scrape images directory
        brand = watch_data.get('specs', {}).get('brand', 'Unknown')
        model = watch_data.get('specs', {}).get('model', 'Unknown')
        
        # Create filename similar to what we've seen
        filename = f"{brand}_{model}_main.jpg"
        image_path = f"production_scrape_20250601_175426/images/{filename}"
        
        if os.path.exists(image_path):
            return image_path
        
        # Try alternative naming patterns
        alt_paths = [
            f"production_scrape_20250601_175426/images/{brand}_{model.replace(' ', '_')}_main.jpg",
            f"production_scrape_20250601_175426/images/{brand}_{model.replace(' ', '_').replace('(', '').replace(')', '')}_main.jpg",
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                return alt_path
        
        return None
    
    def load_and_preprocess_image(self, image_path, target_size: tuple = (224, 224)):
        """Load and preprocess image for model input."""
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            print(f"⚠️ Failed to load image {image_path}: {e}")
            return None
    
    def generate_clip_embeddings(self, image, model_name: str):
        """Generate CLIP embeddings for an image."""
        if model_name not in self.models or model_name not in self.processors:
            return None
        
        try:
            processor = self.processors[model_name]
            model = self.models[model_name]
            
            # Process image
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                embedding = F.normalize(image_features, p=2, dim=1)
            
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"⚠️ Failed to generate {model_name} embedding: {e}")
            return None
    
    def generate_dino_embeddings(self, image):
        """Generate DINO embeddings for an image."""
        if 'dino_vit' not in self.models:
            return None
        
        try:
            model = self.models['dino_vit']
            
            # Preprocess for DINO
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Process image
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                embedding = model(image_tensor)
            
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"⚠️ Failed to generate DINO embedding: {e}")
            return None
    
    def generate_all_embeddings(self, balanced_data):
        """Generate embeddings for all models and all watches."""
        print("🎨 Generating embeddings for all models...")
        
        results = {
            'metadata': [],
            'embeddings': {
                'clip_vit_b32': {},
                'clip_vit_l14': {},
                'dino_vit': {}
            }
        }
        
        successful_watches = 0
        
        for i, watch in enumerate(balanced_data):
            if i % 50 == 0:
                print(f"  Processing watch {i+1}/{len(balanced_data)}...")
            
            # Get image path
            image_path = self.get_image_path(watch)
            if not image_path:
                continue
            
            # Load image
            image = self.load_and_preprocess_image(image_path)
            if image is None:
                continue
            
            # Generate embeddings for each model
            watch_id = f"watch_{i}"
            all_embeddings_generated = True
            
            # CLIP ViT-B/32
            clip_b32_emb = self.generate_clip_embeddings(image, 'clip_vit_b32')
            if clip_b32_emb is not None:
                results['embeddings']['clip_vit_b32'][watch_id] = clip_b32_emb
            else:
                all_embeddings_generated = False
            
            # CLIP ViT-L/14
            clip_l14_emb = self.generate_clip_embeddings(image, 'clip_vit_l14')
            if clip_l14_emb is not None:
                results['embeddings']['clip_vit_l14'][watch_id] = clip_l14_emb
            else:
                all_embeddings_generated = False
            
            # DINO ViT
            dino_emb = self.generate_dino_embeddings(image)
            if dino_emb is not None:
                results['embeddings']['dino_vit'][watch_id] = dino_emb
            else:
                all_embeddings_generated = False
            
            # Only include watches where all embeddings were generated successfully
            if all_embeddings_generated:
                results['metadata'].append({
                    'watch_id': watch_id,
                    'image_path': image_path,
                    'brand': watch.get('specs', {}).get('brand', 'Unknown'),
                    'model': watch.get('specs', {}).get('model', 'Unknown'),
                    'watch_type': watch.get('specs', {}).get('watch_type', 'Unknown'),
                    'price': watch.get('specs', {}).get('price_usd', 'Unknown'),
                    'original_data': watch
                })
                successful_watches += 1
        
        print(f"✅ Generated embeddings for {successful_watches} watches")
        return results
    
    def save_results(self, results, output_dir: str = 'balanced_embeddings'):
        """Save the results to disk."""
        print(f"💾 Saving results to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(results['metadata'], f, indent=2)
        
        # Save embeddings for each model
        for model_name, embeddings in results['embeddings'].items():
            if embeddings:
                with open(f"{output_dir}/{model_name}_embeddings.pkl", 'wb') as f:
                    pickle.dump(embeddings, f)
                print(f"  ✅ Saved {model_name}: {len(embeddings)} embeddings")
        
        # Save summary
        summary = {
            'total_watches': len(results['metadata']),
            'models': list(results['embeddings'].keys()),
            'embedding_dimensions': {
                model_name: len(next(iter(embeddings.values()))) if embeddings else 0
                for model_name, embeddings in results['embeddings'].items()
            }
        }
        
        with open(f"{output_dir}/summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ All results saved to {output_dir}/")
        return summary

def main():
    """Main function to generate balanced embeddings."""
    print("🎯 Balanced Embedding Generator for Model Comparison")
    print("="*60)
    
    # Initialize generator
    generator = BalancedEmbeddingGenerator()
    
    # Load models
    generator.load_models()
    
    if not generator.models:
        print("❌ No models loaded. Exiting.")
        return
    
    # Load scrape data
    data = generator.load_scrape_data()
    
    # Create balanced subset
    balanced_data = generator.create_balanced_subset(data, watches_per_brand=10)
    
    # Generate embeddings
    results = generator.generate_all_embeddings(balanced_data)
    
    # Save results
    summary = generator.save_results(results)
    
    print(f"\n🏆 Generation Complete!")
    print(f"📊 Summary:")
    print(f"  Total watches: {summary['total_watches']}")
    print(f"  Models: {', '.join(summary['models'])}")
    print(f"  Embedding dimensions:")
    for model, dims in summary['embedding_dimensions'].items():
        print(f"    {model}: {dims}D")
    
    print(f"\n✅ Ready for model comparison!")

if __name__ == "__main__":
    main() 