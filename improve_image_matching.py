#!/usr/bin/env python3
"""
Improve image URL matching with fuzzy logic and better name handling.
"""

import json
import pickle
import os
from typing import Dict, List, Any
import logging
from difflib import SequenceMatcher

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_name(name: str) -> str:
    """Normalize brand/model names for better matching."""
    if not name:
        return ""
    
    # Convert to lowercase and remove common variations
    normalized = name.lower().strip()
    
    # Remove common words that don't help matching
    remove_words = ['watch', 'company', 'co', 'ltd', 'limited', 'the']
    for word in remove_words:
        normalized = normalized.replace(f' {word} ', ' ')
        normalized = normalized.replace(f'{word} ', '')
        normalized = normalized.replace(f' {word}', '')
    
    # Remove special characters but keep spaces
    normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
    
    # Clean up multiple spaces
    normalized = ' '.join(normalized.split())
    
    return normalized

def calculate_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between two names."""
    if not name1 or not name2:
        return 0.0
    
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)
    
    # Exact match after normalization
    if norm1 == norm2:
        return 1.0
    
    # Sequence matcher for fuzzy matching
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    
    # Bonus for partial matches
    if norm1 in norm2 or norm2 in norm1:
        similarity += 0.2
    
    return min(similarity, 1.0)

def load_original_data() -> Dict[str, Dict[str, Any]]:
    """Load the original processed data with CDN URLs."""
    logger.info("Loading original processed data...")
    
    with open("data/processed/watches_processed.json", "r") as f:
        original_data = json.load(f)
    
    # Create multiple mappings for better matching
    url_mappings = {
        'exact': {},      # Exact brand_model match
        'fuzzy': {},      # Fuzzy brand_model match
        'brand_only': {}  # Brand-only match (fallback)
    }
    
    for item in original_data:
        brand = item.get("brand", "").strip()
        model = item.get("model_name", "").strip()
        
        # Exact key
        exact_key = f"{brand}_{model}"
        url_mappings['exact'][exact_key] = item
        
        # Normalized key for fuzzy matching
        norm_brand = normalize_name(brand)
        norm_model = normalize_name(model)
        fuzzy_key = f"{norm_brand}_{norm_model}"
        url_mappings['fuzzy'][fuzzy_key] = item
        
        # Brand-only mapping (for watches with same brand)
        if brand not in url_mappings['brand_only']:
            url_mappings['brand_only'][brand] = []
        url_mappings['brand_only'][brand].append(item)
    
    logger.info(f"Loaded {len(original_data)} original watch entries")
    logger.info(f"Created {len(url_mappings['exact'])} exact mappings")
    logger.info(f"Created {len(url_mappings['fuzzy'])} fuzzy mappings")
    logger.info(f"Created {len(url_mappings['brand_only'])} brand-only mappings")
    
    return url_mappings

def find_best_match(watch_info: Dict[str, Any], url_mappings: Dict[str, Dict]) -> Dict[str, Any]:
    """Find the best matching original data for a watch."""
    brand = watch_info.get("brand", "").strip()
    model = watch_info.get("model", "").strip()
    
    # Try exact match first
    exact_key = f"{brand}_{model}"
    if exact_key in url_mappings['exact']:
        return url_mappings['exact'][exact_key]
    
    # Try fuzzy match
    norm_brand = normalize_name(brand)
    norm_model = normalize_name(model)
    fuzzy_key = f"{norm_brand}_{norm_model}"
    if fuzzy_key in url_mappings['fuzzy']:
        return url_mappings['fuzzy'][fuzzy_key]
    
    # Try brand-only match with best model similarity
    if brand in url_mappings['brand_only']:
        best_match = None
        best_similarity = 0.0
        
        for candidate in url_mappings['brand_only'][brand]:
            candidate_model = candidate.get("model_name", "")
            similarity = calculate_similarity(model, candidate_model)
            
            if similarity > best_similarity and similarity > 0.6:  # Threshold
                best_similarity = similarity
                best_match = candidate
        
        if best_match:
            return best_match
    
    return None

def merge_urls_with_embeddings(url_mappings: Dict[str, Dict], 
                              embeddings_data: Dict[str, Any]) -> Dict[str, Any]:
    """Merge CDN URLs into the DINO embeddings with improved matching."""
    logger.info("Merging CDN URLs with DINO embeddings using improved matching...")
    
    watch_data = embeddings_data.get("watch_data", {})
    updated_count = 0
    missing_count = 0
    fuzzy_matches = 0
    
    for watch_id, watch_info in watch_data.items():
        # Find best match using improved logic
        original_item = find_best_match(watch_info, url_mappings)
        
        if original_item:
            # Add the CDN URL and other useful fields
            watch_info["image_url"] = original_item.get("image_url")
            watch_info["product_url"] = original_item.get("product_url")
            watch_info["price"] = original_item.get("price")
            watch_info["description"] = original_item.get("description")
            watch_info["source"] = original_item.get("source")
            
            # Map to frontend-expected fields
            watch_info["main_image"] = original_item.get("image_url")
            
            updated_count += 1
            
            # Check if this was a fuzzy match
            brand = watch_info.get("brand", "").strip()
            model = watch_info.get("model", "").strip()
            exact_key = f"{brand}_{model}"
            if exact_key not in url_mappings['exact']:
                fuzzy_matches += 1
                
        else:
            # Keep placeholder for watches without URLs
            watch_info["image_url"] = None
            watch_info["product_url"] = None
            watch_info["main_image"] = None
            missing_count += 1
    
    logger.info(f"Updated {updated_count} watches with CDN URLs")
    logger.info(f"  - Exact matches: {updated_count - fuzzy_matches}")
    logger.info(f"  - Fuzzy matches: {fuzzy_matches}")
    logger.info(f"Missing URLs for {missing_count} watches (using placeholders)")
    
    return embeddings_data

def save_updated_embeddings(embeddings_data: Dict[str, Any]) -> None:
    """Save the updated embeddings back to the backend directory."""
    logger.info("Saving updated embeddings...")
    
    output_path = "watch_finder_v2/backend/data/precomputed_embeddings.pkl"
    
    with open(output_path, "wb") as f:
        pickle.dump(embeddings_data, f)
    
    logger.info(f"Saved updated embeddings to {output_path}")

def main():
    """Main function to improve URL matching."""
    logger.info("Starting improved URL matching process...")
    
    try:
        # Load original data with CDN URLs
        url_mappings = load_original_data()
        
        # Load current DINO embeddings
        logger.info("Loading DINO embeddings...")
        with open("watch_finder_v2/backend/data/precomputed_embeddings.pkl", "rb") as f:
            embeddings_data = pickle.load(f)
        
        # Merge URLs into embeddings with improved matching
        updated_embeddings = merge_urls_with_embeddings(url_mappings, embeddings_data)
        
        # Save updated embeddings
        save_updated_embeddings(updated_embeddings)
        
        logger.info("✅ Successfully improved image URL matching!")
        
    except Exception as e:
        logger.error(f"❌ Error during improved matching process: {e}")
        raise

if __name__ == "__main__":
    main() 