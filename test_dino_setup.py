#!/usr/bin/env python3
"""
Test DINO Setup
==============

Simple test script to verify DINO model loading and basic functionality.
"""

import os
import sys
import torch
import logging
from PIL import Image
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dino_import():
    """Test if DINO can be imported."""
    try:
        import torch.hub
        logger.info("✅ PyTorch hub available")
        
        # Try to load DINO model
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        logger.info("✅ DINO model loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ DINO import failed: {e}")
        return False

def test_image_processing():
    """Test image processing pipeline."""
    try:
        from torchvision import transforms
        
        # Create test transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("✅ Image transforms created")
        return True
        
    except Exception as e:
        logger.error(f"❌ Image processing test failed: {e}")
        return False

def test_data_paths():
    """Test if required data paths exist."""
    csv_path = "production_scrape_20250601_175426/data/final_scrape.csv"
    images_dir = "production_scrape_20250601_175426/images"
    
    if os.path.exists(csv_path):
        logger.info(f"✅ CSV file found: {csv_path}")
    else:
        logger.error(f"❌ CSV file not found: {csv_path}")
        return False
    
    if os.path.exists(images_dir):
        image_count = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        logger.info(f"✅ Images directory found: {images_dir} ({image_count} images)")
    else:
        logger.error(f"❌ Images directory not found: {images_dir}")
        return False
    
    return True

def test_sample_embedding():
    """Test generating a sample embedding."""
    try:
        import torch.hub
        from torchvision import transforms
        from PIL import Image
        
        # Load model
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        model.eval()
        
        # Setup transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Find a sample image
        images_dir = "production_scrape_20250601_175426/images"
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        
        if not image_files:
            logger.error("❌ No sample images found")
            return False
        
        sample_image = os.path.join(images_dir, image_files[0])
        logger.info(f"📸 Using sample image: {sample_image}")
        
        # Load and process image
        image = Image.open(sample_image).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Generate embedding
        with torch.no_grad():
            features = model.forward_features(image_tensor)
            embedding = features['x_norm_clstoken'].cpu().numpy().flatten()
        
        logger.info(f"✅ Sample embedding generated: {embedding.shape}")
        logger.info(f"📊 Embedding stats - Min: {embedding.min():.4f}, Max: {embedding.max():.4f}, Mean: {embedding.mean():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Sample embedding test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Testing DINO Setup")
    logger.info("=" * 30)
    
    tests = [
        ("DINO Import", test_dino_import),
        ("Image Processing", test_image_processing),
        ("Data Paths", test_data_paths),
        ("Sample Embedding", test_sample_embedding)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running: {test_name}")
        try:
            if test_func():
                logger.info(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! DINO setup is ready.")
        logger.info("🚀 You can now run: python generate_dino_embeddings.py")
    else:
        logger.error("❌ Some tests failed. Please fix the issues before proceeding.")
        logger.info("💡 Check the error messages above for details.")

if __name__ == "__main__":
    main() 