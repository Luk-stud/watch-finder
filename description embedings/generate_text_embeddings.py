#!/usr/bin/env python3
"""
Text-based Watch Embedding Generation

This script generates watch descriptions using OpenAI GPT and creates embeddings
using OpenAI's text-embedding-3-small model for semantic search.
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

try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI library not found. Please install it with: pip install openai>=1.0.0")
    sys.exit(1)

# Load environment variables
load_dotenv()

class WatchTextEmbeddingGenerator:
    """Generate text embeddings for watches using OpenAI's APIs."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: OpenAI API key. If None, will read from environment.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.embedding_model = "text-embedding-3-small"
        self.description_model = "gpt-4"  # Upgraded to GPT-4 for better quality
        
        # Rate limiting
        self.description_delay = 1.0  # seconds between description requests
        self.embedding_delay = 0.1    # seconds between embedding requests
        
    def build_color_context(self, specs: Dict) -> str:
        """Build a comprehensive color context from watch specifications."""
        context_elements = []
        
        # Extract color-related specifications
        dial_color = specs.get('dial_color', '')
        case_material = specs.get('case_material', '')
        bezel_material = specs.get('bezel_insert_material', '')
        strap_material = specs.get('strap_1_material', '')
        
        # Add dial color if available
        if dial_color and dial_color != '-':
            context_elements.append(f"with its {dial_color} dial")
        
        # Add case material if it has a distinctive color
        if case_material and any(color in case_material.lower() for color in 
            ['black', 'blue', 'gold', 'rose', 'silver', 'bronze', 'copper', 'platinum', 'titanium']):
            context_elements.append(f"featuring a {case_material} case")
        
        # Add bezel information if available
        if bezel_material and bezel_material != '-':
            context_elements.append(f"complemented by a {bezel_material} bezel")
        
        # Add strap/bracelet information if available
        if strap_material and strap_material != '-':
            context_elements.append(f"finished with a {strap_material} strap")
        
        # Combine all elements
        if context_elements:
            return ", " + ", ".join(context_elements)
        return ""
        
    def generate_watch_description(self, brand: str, model: str, specs: Dict = None) -> str:
        """
        Generate an enhanced watch description using the structured multi-aspect prompt.
        
        Args:
            brand: Watch brand name
            model: Watch model name
            specs: Optional watch specifications
            
        Returns:
            Generated description text
        """
        # Build color context
        color_context = self.build_color_context(specs) if specs else ""
        
        # Enhanced system message
        system_message = """You are an expert watch designer and critic with deep knowledge of:
- Horological aesthetics and design language
- Watch materials, finishes, and their visual impact
- How different design elements interact
- The relationship between watches and lifestyle
- Color theory and material combinations in watchmaking

Focus on creating descriptions that are both precise and evocative, connecting specific details to their broader impact on the watch's character."""

        # Structured multi-aspect prompt
        prompt = f"""Describe the {brand} {model} watch{color_context} by covering these key aspects:

AESTHETICS & IDENTITY:
- Overall design philosophy and visual character
- How the watch makes a statement about its wearer
- The lifestyle and occasions it's best suited for

VISUAL ELEMENTS:
- Dial design: color scheme, texture, and layout
- Case and bezel: materials, finish, and proportions
- Hands and markers: style and relationship to the dial
- Strap/bracelet: material, texture, and integration with the case

EMOTIONAL & PRACTICAL APPEAL:
- The feelings and emotions the watch evokes
- Key functional elements that enhance its appeal
- How different elements work together to create harmony

Keep the description natural and flowing (2-3 sentences per aspect), emphasizing how each element contributes to the overall experience."""

        try:
            # Set temperature based on available information
            temperature = 0.5 if specs else 0.7
            
            response = self.client.chat.completions.create(
                model=self.description_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,  # Increased for more detailed descriptions
                temperature=temperature
            )
            
            description = response.choices[0].message.content.strip()
            return description
            
        except Exception as e:
            print(f"Error generating description for {brand} {model}: {e}")
            # Fallback description incorporating available color context
            fallback = f"A distinctive timepiece from {brand}, the {model}"
            if color_context:
                fallback += f"{color_context}"
            fallback += " represents the brand's signature design approach with its unique aesthetic character."
            return fallback
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for given text using OpenAI's embedding model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def process_watch_data(self, watch_data: List[Dict], output_dir: str = "output", max_watches: int = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process watch data to generate descriptions and embeddings.
        
        Args:
            watch_data: List of watch dictionaries with brand and model info
            output_dir: Directory to save intermediate and final results
            max_watches: Maximum number of watches to process (None for all)
            
        Returns:
            Tuple of (embedding_matrix, enhanced_watch_data)
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load existing progress if available
        existing_watches, existing_embeddings = self._load_existing_progress(output_path)
        existing_count = len(existing_watches)
        
        if existing_count > 0:
            print(f"📋 Found {existing_count} existing processed watches - will resume from there")
        
        # Limit watches if specified
        if max_watches is not None:
            if existing_count >= max_watches:
                print(f"✅ Already have {existing_count} watches (target: {max_watches})")
                embedding_matrix = np.array(existing_embeddings) if existing_embeddings else np.array([])
                return embedding_matrix, existing_watches
            
            remaining_needed = max_watches - existing_count
            watches_to_process = watch_data[existing_count:existing_count + remaining_needed]
            print(f"🎯 Processing {len(watches_to_process)} more watches (have {existing_count}, target {max_watches})")
        else:
            watches_to_process = watch_data[existing_count:]
            print(f"📊 Processing {len(watches_to_process)} remaining watches out of {len(watch_data)} total")
        
        enhanced_watches = existing_watches.copy()
        embeddings = existing_embeddings.copy()
        
        if len(watches_to_process) == 0:
            print("✅ All watches already processed!")
            embedding_matrix = np.array(embeddings) if embeddings else np.array([])
            return embedding_matrix, enhanced_watches
        
        print(f"🚀 Starting processing from watch #{existing_count + 1}...")
        
        for i, watch in enumerate(tqdm(watches_to_process, desc="Generating descriptions and embeddings")):
            actual_index = existing_count + i
            try:
                # Extract brand and model
                brand = watch.get('brand', 'Unknown')
                model = watch.get('model', 'Unknown Model')
                
                # Skip if essential info is missing
                if brand == 'Unknown' or model == 'Unknown Model':
                    print(f"Skipping watch {actual_index}: missing brand or model")
                    continue
                
                # Generate description
                print(f"Generating description for {brand} {model}...")
                description = self.generate_watch_description(brand, model, watch.get('specs', {}))
                time.sleep(self.description_delay)  # Rate limiting
                
                # Generate embedding
                print(f"Generating embedding for {brand} {model}...")
                embedding = self.generate_embedding(description)
                time.sleep(self.embedding_delay)  # Rate limiting
                
                if embedding is None:
                    print(f"Failed to generate embedding for {brand} {model}")
                    continue
                
                # Store enhanced watch data
                enhanced_watch = watch.copy()
                enhanced_watch['ai_description'] = description
                enhanced_watch['text_embedding'] = embedding
                
                enhanced_watches.append(enhanced_watch)
                embeddings.append(embedding)
                
                # Save intermediate results every 10 watches
                if (actual_index + 1) % 10 == 0:
                    self._save_intermediate_results(enhanced_watches, embeddings, output_path, actual_index + 1)
                
            except Exception as e:
                print(f"Error processing watch {actual_index} ({brand} {model}): {e}")
                continue
        
        # Convert embeddings to numpy array
        embedding_matrix = np.array(embeddings) if embeddings else np.array([])
        
        print(f"Successfully processed {len(enhanced_watches)} watches total")
        return embedding_matrix, enhanced_watches
    
    def _load_existing_progress(self, output_path: Path) -> Tuple[List[Dict], List[List[float]]]:
        """Load existing progress from intermediate or final files."""
        try:
            # Check for final files first
            metadata_file = output_path / "watch_metadata.pkl"
            embeddings_file = output_path / "watch_embeddings.pkl"
            
            if metadata_file.exists() and embeddings_file.exists():
                print(f"📁 Loading existing final results...")
                with open(metadata_file, 'rb') as f:
                    watches = pickle.load(f)
                with open(embeddings_file, 'rb') as f:
                    embeddings_matrix = pickle.load(f)
                
                # Convert numpy array back to list of lists
                if isinstance(embeddings_matrix, np.ndarray):
                    embeddings = embeddings_matrix.tolist()
                else:
                    embeddings = embeddings_matrix
                
                return watches, embeddings
            
            # Check for most recent intermediate file
            intermediate_files = list(output_path.glob("enhanced_watches_intermediate_*.pkl"))
            if intermediate_files:
                # Get the most recent intermediate file
                latest_file = max(intermediate_files, key=lambda x: int(x.stem.split('_')[-1]))
                count = int(latest_file.stem.split('_')[-1])
                
                print(f"📁 Loading intermediate results from {latest_file.name}...")
                
                # Load watches
                with open(latest_file, 'rb') as f:
                    watches = pickle.load(f)
                
                # Load corresponding embeddings
                embeddings_file = output_path / f"embeddings_intermediate_{count}.npy"
                if embeddings_file.exists():
                    embeddings_matrix = np.load(embeddings_file)
                    embeddings = embeddings_matrix.tolist()
                else:
                    # Extract embeddings from watches
                    embeddings = [w.get('text_embedding', []) for w in watches if 'text_embedding' in w]
                
                return watches, embeddings
            
        except Exception as e:
            print(f"⚠️  Error loading existing progress: {e}")
        
        return [], []
    
    def _save_intermediate_results(self, watches: List[Dict], embeddings: List[List[float]], 
                                 output_path: Path, count: int):
        """Save intermediate results to avoid losing progress."""
        try:
            # Save enhanced watch data
            with open(output_path / f"enhanced_watches_intermediate_{count}.pkl", 'wb') as f:
                pickle.dump(watches, f)
            
            # Save embeddings
            embedding_array = np.array(embeddings)
            np.save(output_path / f"embeddings_intermediate_{count}.npy", embedding_array)
            
            print(f"Saved intermediate results for {count} watches")
            
        except Exception as e:
            print(f"Error saving intermediate results: {e}")
    
    def save_results(self, embedding_matrix: np.ndarray, enhanced_watches: List[Dict], 
                    output_dir: str = "output"):
        """
        Save final results to files in the same format as the image-based embedding system.
        
        Args:
            embedding_matrix: Numpy array of embeddings
            enhanced_watches: List of enhanced watch dictionaries
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            # Add index field to each watch (required by backend)
            for i, watch in enumerate(enhanced_watches):
                watch['index'] = i
            
            # Remove text_embedding from individual watches (we save the matrix separately)
            clean_watches = []
            for watch in enhanced_watches:
                clean_watch = {k: v for k, v in watch.items() if k != 'text_embedding'}
                clean_watches.append(clean_watch)
            
            # Save in SAME FORMAT as image-based system
            # 1. Save embeddings as numpy array in pickle file (same as image system)
            embeddings_file = output_path / "watch_embeddings.pkl"
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embedding_matrix, f)
            print(f"Saved embeddings to {embeddings_file}")
            
            # 2. Save watch metadata separately (same as image system)
            metadata_file = output_path / "watch_metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(clean_watches, f)
            print(f"Saved watch metadata to {metadata_file}")
            
            # Additional files for reference and debugging
            # Save as JSON for human readability
            json_file = output_path / "enhanced_watch_metadata.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(clean_watches, f, indent=2, ensure_ascii=False)
            print(f"Saved readable metadata to {json_file}")
            
            # Save summary statistics
            stats = {
                'total_watches': len(enhanced_watches),
                'embedding_dimension': embedding_matrix.shape[1] if embedding_matrix.size > 0 else 0,
                'embedding_model': self.embedding_model,
                'description_model': self.description_model,
                'embedding_type': 'openai_text',
                'compatible_with_backend': True
            }
            
            stats_file = output_path / "generation_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Saved generation stats to {stats_file}")
            
            print(f"\n🎯 Backend-Compatible Files Generated:")
            print(f"📊 Embeddings: {embeddings_file} (format: numpy array in pickle)")
            print(f"📋 Metadata: {metadata_file} (format: list of dicts with 'index' field)")
            print(f"🔄 Ready to replace existing backend files!")
            
        except Exception as e:
            print(f"Error saving results: {e}")

    def save_backend_ready_files(self, embedding_matrix: np.ndarray, enhanced_watches: List[Dict]):
        """
        Save files directly to backend directory in production format.
        
        Args:
            embedding_matrix: Numpy array of embeddings
            enhanced_watches: List of enhanced watch dictionaries
        """
        backend_data_dir = Path("../backend/data")
        
        try:
            # Ensure backend data directory exists
            backend_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Add index field to each watch (required by backend)
            for i, watch in enumerate(enhanced_watches):
                watch['index'] = i
            
            # Remove text_embedding from individual watches
            clean_watches = []
            for watch in enhanced_watches:
                clean_watch = {k: v for k, v in watch.items() if k != 'text_embedding'}
                clean_watches.append(clean_watch)
            
            # Save embeddings in exact backend format
            embeddings_file = backend_data_dir / "watch_embeddings.pkl"
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embedding_matrix, f)
            print(f"✅ Saved backend embeddings to {embeddings_file}")
            
            # Save metadata in exact backend format  
            metadata_file = backend_data_dir / "watch_metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(clean_watches, f)
            print(f"✅ Saved backend metadata to {metadata_file}")
            
            print(f"\n🎉 Backend files updated!")
            print(f"📊 {len(clean_watches)} watches with {embedding_matrix.shape[1]}D text embeddings")
            print(f"🚀 Ready for deployment!")
            
        except Exception as e:
            print(f"❌ Error saving backend files: {e}")


def clean_value(value):
    """Convert pandas NaN values to None for proper JSON serialization"""
    if pd.isna(value):
        return None
    return value

def load_production_scrape_data(scrape_path: str) -> List[Dict]:
    """
    Load watch data from the production scrape CSV.
    
    Args:
        scrape_path: Path to the production scrape directory
        
    Returns:
        List of watch dictionaries
    """
    csv_path = os.path.join(scrape_path, "data", "final_scrape.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Final scrape CSV not found: {csv_path}")
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Convert to the format expected by the text embedding generator
    watch_data = []
    
    for _, row in df.iterrows():
        # Build comprehensive metadata with all available specs
        watch = {
            'brand': row['brand'],
            'model': row['model'],
            'price': row.get('price_usd', 'Contact for price'),
            'image_url': row.get('main_image', ''),
            'product_url': row.get('url', ''),
            'description': f"{row['brand']} {row['model']}",
            'source': 'production_scrape_20250601',
            'specs': {
                # Basic info
                'brand': row['brand'],
                'model': row['model'],
                'serie': clean_value(row.get('serie', '')),
                'reference': clean_value(row.get('reference', '')),
                'watch_type': clean_value(row.get('watch_type', '')),
                'second_watch_type': clean_value(row.get('second_watch_type', '')),
                'url': clean_value(row.get('url', '')),
                
                # Pricing
                'price_usd': clean_value(row.get('price_usd', '')),
                'msrp_eur': clean_value(row.get('msrp_eur', '')),
                'launch_price_eur': clean_value(row.get('launch_price_eur', '')),
                'availability': clean_value(row.get('availability', '')),
                'quantity_produced': clean_value(row.get('quantity_produced', '')),
                'limited_edition_name': clean_value(row.get('limited_edition_name', '')),
                'launch_date': clean_value(row.get('launch_date', '')),
                
                # Case specifications
                'case_material': clean_value(row.get('case_material', '')),
                'bottom_case_material': clean_value(row.get('bottom_case_material', '')),
                'diameter_mm': clean_value(row.get('diameter_mm', '')),
                'thickness_with_crystal_mm': clean_value(row.get('thickness_with_crystal_mm', '')),
                'thickness_without_crystal_mm': clean_value(row.get('thickness_without_crystal_mm', '')),
                'lug_to_lug_mm': clean_value(row.get('lug_to_lug_mm', '')),
                'case_shape': clean_value(row.get('case_shape', '')),
                'case_coating_pvd_dlc': clean_value(row.get('case_coating_pvd_dlc', '')),
                'case_finishing': clean_value(row.get('case_finishing', '')),
                'drill_lugs_on_case': clean_value(row.get('drill_lugs_on_case', '')),
                
                # Dial specifications
                'dial_color': clean_value(row.get('dial_color', '')),
                'dial_material': clean_value(row.get('dial_material', '')),
                'dial_type': clean_value(row.get('dial_type', '')),
                'dial_pattern': clean_value(row.get('dial_pattern', '')),
                'indices_type': clean_value(row.get('indices_type', '')),
                'hands_types': clean_value(row.get('hands_types', '')),
                'full_lume': clean_value(row.get('full_lume', '')),
                'lume_1': clean_value(row.get('lume_1', '')),
                'lume_2': clean_value(row.get('lume_2', '')),
                'color_match_date_wheel': clean_value(row.get('color_match_date_wheel', '')),
                
                # Movement specifications
                'movement': clean_value(row.get('movement', '')),
                'winding': clean_value(row.get('winding', '')),
                'power_reserve_hour': clean_value(row.get('power_reserve_hour', '')),
                'mph': clean_value(row.get('mph', '')),
                'hacking': clean_value(row.get('hacking', '')),
                'manual_winding': clean_value(row.get('manual_winding', '')),
                
                # Complications
                'complication_chronograph': clean_value(row.get('complication_chronograph', '')),
                'complication_date': clean_value(row.get('complication_date', '')),
                'complication_dual_time': clean_value(row.get('complication_dual_time', '')),
                'complication_flying_tourbillon': clean_value(row.get('complication_flying_tourbillon', '')),
                'complication_gmt': clean_value(row.get('complication_gmt', '')),
                'complication_jump_hour': clean_value(row.get('complication_jump_hour', '')),
                'complication_power_reserve': clean_value(row.get('complication_power_reserve', '')),
                'complication_small_second': clean_value(row.get('complication_small_second', '')),
                'complication_sub_24_hour': clean_value(row.get('complication_sub_24_hour', '')),
                'complication_sub_second': clean_value(row.get('complication_sub_second', '')),
                'complication_2nd_bezel_timezone': clean_value(row.get('complication_2nd_bezel_timezone', '')),
                'complication_moonphase': clean_value(row.get('complication_moonphase', '')),
                'complication_world_time_zone': clean_value(row.get('complication_world_time_zone', '')),
                'complication_alarm': clean_value(row.get('complication_alarm', '')),
                'complication_chronoscope': clean_value(row.get('complication_chronoscope', '')),
                
                # Crystal and bezel
                'crystal_material': clean_value(row.get('crystal_material', '')),
                'crystal_type_shape': clean_value(row.get('crystal_type_shape', '')),
                'inner_coating': clean_value(row.get('inner_coating', '')),
                'outside_coating': clean_value(row.get('outside_coating', '')),
                'bezel_insert_material': clean_value(row.get('bezel_insert_material', '')),
                'external_bezel_material': clean_value(row.get('external_bezel_material', '')),
                'bezel_type': clean_value(row.get('bezel_type', '')),
                'uni_bi_direction_bezel': clean_value(row.get('uni_bi_direction_bezel', '')),
                'number_of_clicks': clean_value(row.get('number_of_clicks', '')),
                'internal_bezel': clean_value(row.get('internal_bezel', '')),
                
                # Crown and strap/bracelet
                'main_crown_type': clean_value(row.get('main_crown_type', '')),
                'other_crowns_function': clean_value(row.get('other_crowns_function', '')),
                'strap_1_material': clean_value(row.get('strap_1_material', '')),
                'strap_2_material': clean_value(row.get('strap_2_material', '')),
                'strap_3_material': clean_value(row.get('strap_3_material', '')),
                'width_mm': clean_value(row.get('width_mm', '')),
                'bracelet_tapper_to_clasp_mm': clean_value(row.get('bracelet_tapper_to_clasp_mm', '')),
                'bracelet_type': clean_value(row.get('bracelet_type', '')),
                'bracelet_links_type': clean_value(row.get('bracelet_links_type', '')),
                'bracelet_finishing': clean_value(row.get('bracelet_finishing', '')),
                'strap_bracelet_attachment_system': clean_value(row.get('strap_bracelet_attachment_system', '')),
                'clasp_type': clean_value(row.get('clasp_type', '')),
                'clasp_material': clean_value(row.get('clasp_material', '')),
                
                # Other specifications
                'waterproofing_meters': clean_value(row.get('waterproofing_meters', '')),
                'warranty_year': clean_value(row.get('warranty_year', '')),
                'brand_country': clean_value(row.get('brand_country', '')),
                'made_in': clean_value(row.get('made_in', '')),
                'assembled_in': clean_value(row.get('assembled_in', '')),
                'country': clean_value(row.get('country', '')),
                'waterproofing': clean_value(row.get('waterproofing', '')),
                'specific_info_from_brand': clean_value(row.get('specific_info_from_brand', '')),
                'brand_website': clean_value(row.get('brand_website', '')),
            }
        }
        
        watch_data.append(watch)
    
    print(f"Loaded {len(watch_data)} watches from production scrape")
    
    # Show dataset statistics
    brands = set(watch['brand'] for watch in watch_data)
    print(f"🏷️  Unique brands: {len(brands)}")
    
    # Show brand distribution
    brand_counts = {}
    for watch in watch_data:
        brand = watch['brand']
        brand_counts[brand] = brand_counts.get(brand, 0) + 1
    
    top_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"🏭 Top brands by watch count:")
    for brand, count in top_brands:
        print(f"   - {brand}: {count} watches")
    
    return watch_data


def main(max_watches: int = None):
    """Main execution function."""
    print("🕐 Watch Text Embedding Generator v2")
    print("===================================")
    print("🎯 Generates text-based embeddings compatible with existing image-based system")
    
    if max_watches:
        print(f"🎯 Running on {max_watches} watches (test mode)")
    else:
        print("🔄 Running on ALL watches (full generation)")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    # Load production scrape data (newer dataset)
    production_scrape_path = "../production_scrape_20250601_175426"
    if not os.path.exists(production_scrape_path):
        print(f"❌ Error: Production scrape not found at {production_scrape_path}")
        print("Please ensure the production scrape data exists.")
        return
    
    print(f"📊 Loading data from production scrape: {production_scrape_path}")
    watch_data = load_production_scrape_data(production_scrape_path)
    if not watch_data:
        print("❌ Error: No watch data loaded")
        return
    
    # Initialize generator
    generator = WatchTextEmbeddingGenerator()
    
    # Process watches with optional limit
    embedding_matrix, enhanced_watches = generator.process_watch_data(watch_data, max_watches=max_watches)
    
    if len(enhanced_watches) == 0:
        print("❌ Error: No watches were successfully processed")
        return
    
    # Save results to output directory
    generator.save_results(embedding_matrix, enhanced_watches)
    
    # Also save directly to backend directory for immediate use
    print(f"\n🔄 Saving backend-ready files...")
    generator.save_backend_ready_files(embedding_matrix, enhanced_watches)
    
    target_count = max_watches if max_watches else len(watch_data)
    print(f"\n✅ Successfully generated text embeddings for {len(enhanced_watches)}/{target_count} watches")
    print(f"📊 Embedding dimension: {embedding_matrix.shape[1] if embedding_matrix.size > 0 else 0}")
    print(f"🔄 Format: Compatible with existing image-based embedding system")
    print(f"📁 Files saved to both ./output/ and ../backend/data/")
    print(f"📅 Source: Production scrape 2025-06-01 (latest dataset)")
    print("🎉 Text embedding generation complete!")
    
    if max_watches and len(enhanced_watches) >= max_watches:
        print(f"\n📋 Next Steps:")
        print(f"1. Test the backend with these {len(enhanced_watches)} watches")
        print(f"2. If satisfied, run without max_watches limit for full generation")
        print(f"3. The script will resume from where it left off automatically")
    else:
        print(f"\n📋 Next Steps:")
        print(f"1. The backend files have been updated automatically")
        print(f"2. Restart your backend to use the new text-based embeddings")
        print(f"3. The watch finder will now use AI-generated descriptions for similarity")
        print(f"4. Users will get recommendations based on design language and aesthetic vibe")


if __name__ == "__main__":
    import sys
    
    # Check for command line argument for max watches
    max_watches = None
    if len(sys.argv) > 1:
        try:
            max_watches = int(sys.argv[1])
            print(f"🎯 Command line argument: limiting to {max_watches} watches")
        except ValueError:
            print(f"⚠️  Invalid argument '{sys.argv[1]}' - should be a number. Running on all watches.")
    
    main(max_watches) 