#!/usr/bin/env python3
"""
Simple Watch Image Comparison Tool

Creates side-by-side comparisons of watch images without any text or information.
Just pure visual comparison on the same image.
"""

import os
import sys
from PIL import Image, ImageDraw
import argparse
from typing import List, Tuple

class ImageComparator:
    def __init__(self, output_size: Tuple[int, int] = (1200, 600)):
        self.output_size = output_size
        
    def load_and_resize_image(self, image_path: str, target_size: Tuple[int, int]) -> Image.Image:
        """Load and resize an image to fit the target size while maintaining aspect ratio."""
        try:
            image = Image.open(image_path)
            image = image.convert('RGB')
            
            # Calculate aspect ratio preserving resize
            img_ratio = image.width / image.height
            target_ratio = target_size[0] / target_size[1]
            
            if img_ratio > target_ratio:
                # Image is wider, fit to width
                new_width = target_size[0]
                new_height = int(new_width / img_ratio)
            else:
                # Image is taller, fit to height
                new_height = target_size[1]
                new_width = int(new_height * img_ratio)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create a centered image on white background
            result = Image.new('RGB', target_size, 'white')
            x_offset = (target_size[0] - new_width) // 2
            y_offset = (target_size[1] - new_height) // 2
            result.paste(image, (x_offset, y_offset))
            
            return result
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a placeholder
            placeholder = Image.new('RGB', target_size, '#f0f0f0')
            return placeholder
    
    def create_comparison(self, image_paths: List[str], output_path: str = "comparison.jpg"):
        """Create a side-by-side comparison of the given images."""
        if len(image_paths) < 2:
            print("Need at least 2 images to compare")
            return
        
        if len(image_paths) > 4:
            print("Maximum 4 images supported for comparison")
            image_paths = image_paths[:4]
        
        num_images = len(image_paths)
        
        # Calculate grid layout
        if num_images == 2:
            cols, rows = 2, 1
        elif num_images == 3:
            cols, rows = 3, 1
        elif num_images == 4:
            cols, rows = 2, 2
        
        # Calculate individual image size
        img_width = self.output_size[0] // cols
        img_height = self.output_size[1] // rows
        individual_size = (img_width, img_height)
        
        # Create output image
        comparison = Image.new('RGB', self.output_size, 'white')
        
        # Load and place each image
        for i, image_path in enumerate(image_paths):
            # Calculate position
            col = i % cols
            row = i // cols
            x = col * img_width
            y = row * img_height
            
            # Load and resize image
            img = self.load_and_resize_image(image_path, individual_size)
            
            # Paste into comparison
            comparison.paste(img, (x, y))
        
        # Save the comparison
        comparison.save(output_path, 'JPEG', quality=95)
        print(f"✅ Comparison saved to: {output_path}")
        
        return comparison
    
    def create_grid_comparison(self, image_paths: List[str], cols: int = 3, 
                             output_path: str = "grid_comparison.jpg"):
        """Create a grid comparison with multiple images."""
        if not image_paths:
            print("No images provided")
            return
        
        num_images = len(image_paths)
        rows = (num_images + cols - 1) // cols  # Ceiling division
        
        # Calculate individual image size
        img_width = self.output_size[0] // cols
        img_height = self.output_size[1] // rows
        individual_size = (img_width, img_height)
        
        # Adjust output size to fit grid
        grid_width = cols * img_width
        grid_height = rows * img_height
        
        # Create output image
        comparison = Image.new('RGB', (grid_width, grid_height), 'white')
        
        # Load and place each image
        for i, image_path in enumerate(image_paths):
            # Calculate position
            col = i % cols
            row = i // cols
            x = col * img_width
            y = row * img_height
            
            # Load and resize image
            img = self.load_and_resize_image(image_path, individual_size)
            
            # Paste into comparison
            comparison.paste(img, (x, y))
        
        # Save the comparison
        comparison.save(output_path, 'JPEG', quality=95)
        print(f"✅ Grid comparison saved to: {output_path}")
        
        return comparison

def get_sample_images(image_dir: str, count: int = 4) -> List[str]:
    """Get sample images from the directory."""
    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} not found")
        return []
    
    image_files = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.join(image_dir, filename))
    
    return image_files[:count]

def main():
    parser = argparse.ArgumentParser(description='Create side-by-side watch image comparisons')
    parser.add_argument('images', nargs='*', help='Image paths to compare')
    parser.add_argument('--output', '-o', default='comparison.jpg', help='Output filename')
    parser.add_argument('--grid', '-g', action='store_true', help='Create grid layout')
    parser.add_argument('--cols', '-c', type=int, default=3, help='Number of columns for grid')
    parser.add_argument('--sample', '-s', action='store_true', help='Use sample images from production_scrape')
    parser.add_argument('--count', type=int, default=4, help='Number of sample images to use')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = ImageComparator()
    
    # Get images to compare
    if args.sample:
        # Use sample images from the scrape directory
        image_dir = "production_scrape_20250601_175426/images"
        image_paths = get_sample_images(image_dir, args.count)
        if not image_paths:
            print("No sample images found")
            return
        print(f"Using {len(image_paths)} sample images")
    elif args.images:
        image_paths = args.images
    else:
        print("Please provide image paths or use --sample flag")
        return
    
    # Create comparison
    if args.grid:
        comparator.create_grid_comparison(image_paths, args.cols, args.output)
    else:
        comparator.create_comparison(image_paths, args.output)

if __name__ == "__main__":
    main() 