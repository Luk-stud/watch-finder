#!/usr/bin/env python3
"""
Style-Focused Watch Embedding Generation
=======================================

This script generates brand-agnostic watch descriptions focusing purely on design
characteristics and creates embeddings for style-based recommendations.
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import time
from dotenv import load_dotenv
import requests
import base64
from io import BytesIO

try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI library not found. Please install it with: pip install openai>=1.0.0")
    sys.exit(1)

# Load environment variables
load_dotenv()

class StyleFocusedEmbeddingGenerator:
    """Generate style-focused embeddings for watches without brand bias."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the embedding generator."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.embedding_model = "text-embedding-3-small"
        self.description_model = "gpt-4o-mini"
        
        # Rate limiting for GPT-4o-mini (lower token limits)
        self.description_delay = 2.0  # Increased delay for vision calls
        self.embedding_delay = 0.2   # Slightly increased for embeddings
        
    def generate_style_description(self, brand: str, model: str, watch_specs: Dict = None) -> str:
        """
        Generate a brand-agnostic description focusing purely on style and design.
        
        Args:
            brand: Watch brand (used for research but not included in output)
            model: Watch model (used for research but not included in output)
            watch_specs: Additional specifications if available
            
        Returns:
            Brand-agnostic style description
        """
        # Include specs in the prompt if available
        specs_info = ""
        if watch_specs:
            relevant_specs = []
            for field in ['case_material', 'dial_color', 'movement', 'watch_type', 'case_size']:
                if field in watch_specs and watch_specs[field]:
                    relevant_specs.append(f"{field}: {watch_specs[field]}")
            if relevant_specs:
                specs_info = f"\n\nAvailable specifications: {', '.join(relevant_specs)}"
        
        prompt = f"""Research the {brand} {model} watch and create a brand-agnostic description focusing ONLY on design characteristics.{specs_info}

Create a description that captures:
- Design style (sporty, elegant, rugged, minimalist, vintage, modern, etc.)
- Aesthetic personality (bold, refined, casual, professional, adventure-ready, etc.)
- Visual character (clean lines, complex details, geometric, organic, industrial, etc.)
- Target lifestyle (office wear, sports, diving, aviation, casual daily wear, etc.)
- Design elements (dial layout, case shape, bezel style, crown design, etc.)

CRITICAL RULES:
- DO NOT mention the brand name anywhere in the description
- DO NOT mention the specific model name
- Focus on what the design LOOKS like and FEELS like
- Describe it as if explaining the style to someone who can't see the brand
- Use descriptive adjectives about the design language
- 2-3 sentences maximum

Example good description: "A sporty chronograph with a bold, racing-inspired aesthetic featuring a clean three-subdial layout and tachymeter bezel. The design exudes athletic confidence with modern proportions and legible markers suited for active lifestyles."

Example bad description: "The Rolex Daytona features the brand's signature design..." (mentions brand)
"""

        try:
            response = self.client.chat.completions.create(
                model=self.description_model,
                messages=[
                    {"role": "system", "content": "You are an expert watch designer focused on design language and aesthetics. Create brand-agnostic descriptions that focus purely on visual style and design characteristics."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            description = response.choices[0].message.content.strip()
            
            # Double-check that brand name isn't mentioned
            if brand.lower() in description.lower():
                # Retry with more explicit instruction
                retry_prompt = f"""The previous description mentioned the brand name. Please rewrite this description for a watch with these characteristics, but DO NOT include any brand names:

{description}

Rewrite focusing only on design style, aesthetic, and visual characteristics."""
                
                retry_response = self.client.chat.completions.create(
                    model=self.description_model,
                    messages=[
                        {"role": "system", "content": "You are creating brand-agnostic watch descriptions. Never mention brand names."},
                        {"role": "user", "content": retry_prompt}
                    ],
                    max_tokens=200,
                    temperature=0.5
                )
                description = retry_response.choices[0].message.content.strip()
            
            return description
            
        except Exception as e:
            print(f"Error generating description for {brand} {model}: {e}")
            return "A timepiece with distinctive design characteristics and unique aesthetic personality suited for its intended lifestyle and wearing occasions."
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text."""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def process_final_scrape_with_images(self, final_scrape_path: str, image_base_path: str = None, max_watches: int = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process final scrape data to generate style-focused descriptions from images.
        
        Args:
            final_scrape_path: Path to final scrape JSON file
            image_base_path: Base path where watch images are stored
            max_watches: Maximum number of watches to process
            
        Returns:
            Tuple of (embedding_matrix, enhanced_watch_data)
        """
        # Load final scrape data
        try:
            with open(final_scrape_path, 'r') as f:
                watch_data = json.load(f)
            print(f"✅ Loaded {len(watch_data)} watches from final scrape")
        except Exception as e:
            print(f"❌ Error loading final scrape data: {e}")
            return np.array([]), []
        
        # Set default image path if not provided
        if image_base_path is None:
            image_base_path = "../production_scrape_20250601_175426/images"
        
        # Limit if specified
        if max_watches is not None:
            watch_data = watch_data[:max_watches]
            print(f"🎯 Processing first {len(watch_data)} watches")
        
        enhanced_watches = []
        embeddings = []
        skipped_no_image = 0
        
        print(f"🚀 Generating style-focused descriptions from images...")
        
        # Get all available image files for faster lookup
        available_images = {}
        try:
            for filename in os.listdir(image_base_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    # Create a normalized key for matching
                    key = filename.lower().replace('_', ' ').replace('-', ' ')
                    available_images[key] = filename
        except Exception as e:
            print(f"Error reading image directory: {e}")
            return np.array([]), []
        
        for i, watch in enumerate(tqdm(watch_data, desc="Processing watches")):
            try:
                # Extract info from specs
                specs = watch.get('specs', {})
                brand = specs.get('brand', 'Unknown')
                model = specs.get('model', 'Unknown Model')
                main_image_url = watch.get('main_image', '')
                
                if brand == 'Unknown' or model == 'Unknown Model':
                    print(f"Skipping watch {i}: missing brand or model")
                    continue
                
                # Try to find the corresponding local image
                image_path = None
                
                # Create search patterns based on brand and model
                brand_clean = brand.replace(' ', '_').replace('-', '_')
                model_clean = model.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                
                # Try different combinations
                search_patterns = [
                    f"{brand_clean}_{model_clean}_main.jpg",
                    f"{brand}_{model}_main.jpg".replace(' ', '_'),
                    f"{brand} {model} main.jpg".lower(),
                ]
                
                for pattern in search_patterns:
                    normalized_pattern = pattern.lower().replace('_', ' ').replace('-', ' ')
                    if normalized_pattern in available_images:
                        image_path = os.path.join(image_base_path, available_images[normalized_pattern])
                        break
                
                # If still not found, try partial matching
                if not image_path:
                    brand_lower = brand.lower()
                    model_words = model.lower().split()[:3]  # First few words of model
                    
                    for img_key, img_filename in available_images.items():
                        if brand_lower in img_key:
                            # Check if some model words match
                            matches = sum(1 for word in model_words if word in img_key and len(word) > 2)
                            if matches >= 2:  # At least 2 significant words match
                                image_path = os.path.join(image_base_path, img_filename)
                                break
                
                if not image_path:
                    print(f"Skipping {brand} {model}: no matching image found")
                    skipped_no_image += 1
                    continue
                
                # Extract specifications that might help with description
                watch_specs = {}
                for field in ['case_material', 'dial_color', 'movement', 'watch_type', 'case_size', 'diameter_mm']:
                    if field in specs and specs[field] and specs[field] != '-':
                        watch_specs[field] = specs[field]
                
                # Generate style-focused description from image
                description = self.generate_style_description_from_image(image_path, watch_specs)
                time.sleep(self.description_delay)
                
                # Generate embedding
                embedding = self.generate_embedding(description)
                time.sleep(self.embedding_delay)
                
                if embedding is None:
                    print(f"Failed to generate embedding for {brand} {model}")
                    continue
                
                # Store enhanced watch data
                enhanced_watch = {
                    'brand': brand,
                    'model': model,
                    'specs': specs,
                    'main_image_url': main_image_url,
                    'visual_style_description': description,
                    'style_embedding': embedding,
                    'image_path': image_path,
                    'index': i
                }
                
                enhanced_watches.append(enhanced_watch)
                embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    print(f"✅ Processed {i + 1} watches ({skipped_no_image} skipped - no image)")
                
            except Exception as e:
                print(f"Error processing watch {i} ({brand} {model}): {e}")
                continue
        
        # Convert to numpy array
        embedding_matrix = np.array(embeddings) if embeddings else np.array([])
        
        print(f"✅ Successfully processed {len(enhanced_watches)} watches")
        print(f"⚠️  Skipped {skipped_no_image} watches due to missing images")
        return embedding_matrix, enhanced_watches
    
    def process_final_scrape_with_image_urls(self, final_scrape_path: str, max_watches: int = None, brand_balanced: bool = False) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process final scrape data to generate style-focused descriptions from image URLs.
        
        Args:
            final_scrape_path: Path to final scrape JSON file
            max_watches: Maximum number of watches to process
            brand_balanced: Whether to use brand-balanced sampling
            
        Returns:
            Tuple of (embedding_matrix, enhanced_watch_data)
        """
        # Load final scrape data
        try:
            with open(final_scrape_path, 'r') as f:
                watch_data = json.load(f)
            print(f"✅ Loaded {len(watch_data)} watches from final scrape")
        except Exception as e:
            print(f"❌ Error loading final scrape data: {e}")
            return np.array([]), []
        
        # Apply sampling strategy
        if max_watches is not None:
            if brand_balanced:
                print(f"🎯 Using brand-balanced sampling for {max_watches} watches")
                watch_data = self.get_brand_balanced_sample(watch_data, max_watches)
            else:
                watch_data = watch_data[:max_watches]
                print(f"🎯 Processing first {len(watch_data)} watches")
        
        enhanced_watches = []
        embeddings = []
        skipped_no_image = 0
        
        print(f"🚀 Generating style-focused descriptions from image URLs...")
        
        for i, watch in enumerate(tqdm(watch_data, desc="Processing watches")):
            try:
                # Extract info from specs
                specs = watch.get('specs', {})
                brand = specs.get('brand', 'Unknown')
                model = specs.get('model', 'Unknown Model')
                main_image_url = watch.get('main_image', '')
                
                if brand == 'Unknown' or model == 'Unknown Model':
                    print(f"Skipping watch {i}: missing brand or model")
                    continue
                
                if not main_image_url:
                    print(f"Skipping {brand} {model}: no image URL")
                    skipped_no_image += 1
                    continue
                
                # Extract specifications that might help with description
                watch_specs = {}
                for field in ['case_material', 'dial_color', 'movement', 'watch_type', 'case_size', 'diameter_mm']:
                    if field in specs and specs[field] and specs[field] != '-':
                        watch_specs[field] = specs[field]
                
                # Generate style-focused description from image URL
                description = self.generate_style_description_from_url(main_image_url, watch_specs)
                time.sleep(self.description_delay)
                
                # Generate embedding
                embedding = self.generate_embedding(description)
                time.sleep(self.embedding_delay)
                
                if embedding is None:
                    print(f"Failed to generate embedding for {brand} {model}")
                    continue
                
                # Store enhanced watch data
                enhanced_watch = {
                    'brand': brand,
                    'model': model,
                    'specs': specs,
                    'main_image_url': main_image_url,
                    'visual_style_description': description,
                    'style_embedding': embedding,
                    'index': i
                }
                
                enhanced_watches.append(enhanced_watch)
                embeddings.append(embedding)
                
                if (i + 1) % 25 == 0:
                    print(f"✅ Processed {i + 1} watches ({skipped_no_image} skipped - no image)")
                
            except Exception as e:
                print(f"Error processing watch {i} ({brand} {model}): {e}")
                continue
        
        # Convert to numpy array
        embedding_matrix = np.array(embeddings) if embeddings else np.array([])
        
        print(f"✅ Successfully processed {len(enhanced_watches)} watches")
        print(f"⚠️  Skipped {skipped_no_image} watches due to missing images")
        return embedding_matrix, enhanced_watches
    
    def save_results(self, embedding_matrix: np.ndarray, enhanced_watches: List[Dict], output_dir: str = "style_focused_output"):
        """Save the results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save metadata
        metadata_file = output_path / "watch_style_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(enhanced_watches, f)
        
        # Save embeddings
        embeddings_file = output_path / "watch_style_embeddings.pkl"
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embedding_matrix, f)
        
        # Save sample descriptions for review
        sample_file = output_path / "sample_visual_descriptions.txt"
        with open(sample_file, 'w') as f:
            f.write("Visual Style-Focused Watch Descriptions (Sample)\n")
            f.write("Generated from image analysis - brand-agnostic\n")
            f.write("=" * 60 + "\n\n")
            
            for i, watch in enumerate(enhanced_watches[:20]):  # First 20 samples
                f.write(f"{i+1}. {watch.get('brand', 'Unknown')} {watch.get('model', 'Unknown')}\n")
                f.write(f"   Image: {watch.get('image_path', 'N/A')}\n")
                f.write(f"   Original text: {watch.get('original_description', 'N/A')}\n")
                f.write(f"   Visual analysis: {watch.get('visual_style_description', 'N/A')}\n\n")
        
        print(f"✅ Results saved to {output_dir}/")
        print(f"   - Metadata: {metadata_file}")
        print(f"   - Embeddings: {embeddings_file}")
        print(f"   - Sample descriptions: {sample_file}")

    def generate_style_description_from_image(self, image_path: str, watch_specs: Dict = None) -> str:
        """
        Generate a brand-agnostic description from watch image using vision model.
        
        Args:
            image_path: Path to the watch image
            watch_specs: Additional specifications if available
            
        Returns:
            Brand-agnostic style description based on visual analysis
        """
        # Include specs in the prompt if available
        specs_info = ""
        if watch_specs:
            relevant_specs = []
            for field in ['case_material', 'dial_color', 'movement', 'watch_type', 'case_size']:
                if field in watch_specs and watch_specs[field]:
                    relevant_specs.append(f"{field}: {watch_specs[field]}")
            if relevant_specs:
                specs_info = f"\n\nAdditional specifications: {', '.join(relevant_specs)}"
        
        prompt = f"""Analyze this watch image and create a brand-agnostic description focusing ONLY on the visual design characteristics you can observe.{specs_info}

Describe what you see in terms of:
- Design style (sporty, elegant, rugged, minimalist, vintage, modern, etc.)
- Aesthetic personality (bold, refined, casual, professional, adventure-ready, etc.)
- Visual elements (dial layout, case shape, bezel type, hands style, markers, etc.)
- Overall mood and character (clean, complex, industrial, organic, geometric, etc.)

CRITICAL RULES:
- Describe ONLY what you can visually observe in the image
- DO NOT guess or mention any brand names
- DO NOT mention specific model names or series
- Focus on design language and aesthetic characteristics
- Use descriptive adjectives about the visual style
- 2-3 sentences maximum
- Ignore any text/logos visible on the watch face

Example good description: "A robust diving timepiece with a professional aesthetic featuring a unidirectional rotating bezel, luminous markers, and clean dial layout. The design emphasizes functionality with bold hands and high-contrast elements optimized for underwater readability."
"""

        try:
            # Read and encode the image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",  # Use cost-effective vision model
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert watch designer analyzing timepiece designs. Create brand-agnostic descriptions focusing purely on visual design characteristics without mentioning brands or model names."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            description = response.choices[0].message.content.strip()
            return description
            
        except Exception as e:
            print(f"Error generating description from image {image_path}: {e}")
            return "A timepiece with distinctive design characteristics and unique aesthetic personality."

    def generate_style_description_from_url(self, image_url: str, watch_specs: Dict = None) -> str:
        """
        Generate a brand-agnostic description from watch image URL using vision model.
        
        Args:
            image_url: URL of the watch image
            watch_specs: Additional specifications if available
            
        Returns:
            Brand-agnostic style description based on visual analysis
        """
        # Include specs in the prompt if available
        specs_info = ""
        if watch_specs:
            relevant_specs = []
            for field in ['case_material', 'dial_color', 'movement', 'watch_type', 'case_size', 'diameter_mm']:
                if field in watch_specs and watch_specs[field]:
                    relevant_specs.append(f"{field}: {watch_specs[field]}")
            if relevant_specs:
                specs_info = f"\n\nAdditional specifications: {', '.join(relevant_specs)}"
        
        prompt = f"""Analyze this watch image and create a brand-agnostic description focusing ONLY on the visual design characteristics you can observe.{specs_info}

Describe what you see in terms of:
- Design style (sporty, elegant, rugged, minimalist, vintage, modern, etc.)
- Aesthetic personality (bold, refined, casual, professional, adventure-ready, etc.)
- Visual elements (dial layout, case shape, bezel type, hands style, markers, etc.)
- Overall mood and character (clean, complex, industrial, organic, geometric, etc.)
- Target lifestyle impression (office wear, sports, diving, aviation, casual, etc.)

CRITICAL RULES:
- Describe ONLY what you can visually observe in the image
- DO NOT guess or mention any brand names
- DO NOT mention specific model names or series
- Focus on design language and aesthetic characteristics
- Use descriptive adjectives about the visual style
- 2-3 sentences maximum
- Ignore any text/logos visible on the watch face

Example good description: "A robust diving timepiece with a professional aesthetic featuring a unidirectional rotating bezel, luminous markers, and clean dial layout. The design emphasizes functionality with bold hands and high-contrast elements optimized for underwater readability."
"""

        try:
            # Download and encode the image from URL
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Convert to base64
            base64_image = base64.b64encode(response.content).decode('utf-8')
            
            # Get image type for proper MIME type
            content_type = response.headers.get('content-type', 'image/jpeg')
            
            # Retry logic for rate limits
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",  # Use cost-effective vision model
                        messages=[
                            {
                                "role": "system", 
                                "content": "You are an expert watch designer analyzing timepiece designs. Create brand-agnostic descriptions focusing purely on visual design characteristics without mentioning brands or model names."
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{content_type};base64,{base64_image}",
                                            "detail": "high"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=200,
                        temperature=0.7
                    )
                    
                    description = response.choices[0].message.content.strip()
                    return description
                    
                except Exception as api_error:
                    if "rate_limit" in str(api_error).lower() and attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                        print(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise api_error
            
        except requests.RequestException as e:
            print(f"Error downloading image from {image_url}: {e}")
            return "A timepiece with distinctive design characteristics and unique aesthetic personality."
        except Exception as e:
            print(f"Error generating description from image URL {image_url}: {e}")
            return "A timepiece with distinctive design characteristics and unique aesthetic personality."

    def get_brand_balanced_sample(self, watch_data: List[Dict], target_count: int = 500) -> List[Dict]:
        """
        Get a brand-balanced sample of watches for more representative coverage.
        
        Args:
            watch_data: List of all watch data
            target_count: Target number of watches to sample
            
        Returns:
            Brand-balanced sample of watches
        """
        from collections import defaultdict
        import random
        
        # Group watches by brand
        brand_groups = defaultdict(list)
        for watch in watch_data:
            specs = watch.get('specs', {})
            brand = specs.get('brand', 'Unknown')
            if brand != 'Unknown':
                brand_groups[brand].append(watch)
        
        print(f"📊 Found {len(brand_groups)} brands in dataset")
        
        # Print brand distribution
        brand_counts = {brand: len(watches) for brand, watches in brand_groups.items()}
        sorted_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
        print("🏷️  Top 10 brands by count:")
        for brand, count in sorted_brands[:10]:
            print(f"   {brand}: {count} watches")
        
        # Calculate samples per brand
        total_brands = len(brand_groups)
        base_per_brand = target_count // total_brands
        remainder = target_count % total_brands
        
        print(f"🎯 Targeting {target_count} watches across {total_brands} brands")
        print(f"📏 Base allocation: {base_per_brand} per brand, {remainder} extra for largest brands")
        
        # Sample from each brand
        balanced_sample = []
        brands_with_extra = sorted_brands[:remainder]  # Give extra to largest brands
        
        for brand, watches in brand_groups.items():
            # Determine how many to sample from this brand
            samples_needed = base_per_brand
            if any(brand == b[0] for b in brands_with_extra):
                samples_needed += 1
            
            # Sample (with replacement if needed)
            available_count = len(watches)
            if available_count >= samples_needed:
                sampled = random.sample(watches, samples_needed)
            else:
                # If not enough watches, take all and pad with repeats
                sampled = watches + random.choices(watches, k=samples_needed - available_count)
            
            balanced_sample.extend(sampled)
            print(f"   {brand}: {len(sampled)} watches (from {available_count} available)")
        
        # Shuffle the final sample
        random.shuffle(balanced_sample)
        
        print(f"✅ Created brand-balanced sample of {len(balanced_sample)} watches")
        return balanced_sample

def main():
    """Main function to run the style-focused embedding generation."""
    generator = StyleFocusedEmbeddingGenerator()
    
    # Path to final scrape data
    input_path = "data/processed/watches_processed.json"
    
    # Generate style-focused embeddings from image URLs
    embedding_matrix, enhanced_watches = generator.process_final_scrape_with_image_urls(
        input_path
    )
    
    # Save results
    generator.save_results(embedding_matrix, enhanced_watches, "visual_style_output")
    
    print("\n🎉 Visual style embedding generation complete!")

if __name__ == "__main__":
    main() 