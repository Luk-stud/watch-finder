import numpy as np
import re
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import difflib

class WatchVariantDetector:
    def __init__(self, watch_data: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Initialize variant detector with watch data and embeddings.
        
        Args:
            watch_data: List of watch dictionaries
            embeddings: Normalized watch embeddings
        """
        self.watch_data = watch_data
        self.embeddings = embeddings
        self.variant_groups = {}  # group_id -> [watch_indices]
        self.watch_to_group = {}  # watch_index -> group_id
        self.group_representatives = {}  # group_id -> representative_watch_index
        
        # Thresholds for variant detection
        self.visual_similarity_threshold = 0.90  # Very high visual similarity
        self.text_similarity_threshold = 0.80   # High text similarity
        self.hybrid_visual_threshold = 0.75     # Lower visual + high text
        
        print("Initializing variant detection...")
        self._detect_all_variants()
        print(f"Detected {len(self.variant_groups)} variant groups")

    def _detect_all_variants(self):
        """Detect all variant groups in the dataset."""
        processed = set()
        group_id = 0
        
        for i in range(len(self.watch_data)):
            if i in processed:
                continue
                
            # Start new variant group
            variant_group = [i]
            processed.add(i)
            
            # Find all variants of this watch
            for j in range(i + 1, len(self.watch_data)):
                if j in processed:
                    continue
                    
                if self._are_variants(i, j):
                    variant_group.append(j)
                    processed.add(j)
            
            # Store the group
            if len(variant_group) > 1:
                # Multiple variants found
                self.variant_groups[group_id] = variant_group
                representative = self._choose_representative(variant_group)
                self.group_representatives[group_id] = representative
                
                for watch_idx in variant_group:
                    self.watch_to_group[watch_idx] = group_id
                
                print(f"Variant group {group_id}: {len(variant_group)} variants of {self.watch_data[representative]['brand']} {self._get_base_model_name(self.watch_data[representative]['model'])}")
                group_id += 1
            else:
                # Single watch, no variants
                self.watch_to_group[i] = None

    def _are_variants(self, idx_a: int, idx_b: int) -> bool:
        """Check if two watches are variants of each other."""
        watch_a = self.watch_data[idx_a]
        watch_b = self.watch_data[idx_b]
        
        # Must be same brand
        if watch_a.get('brand', '').lower() != watch_b.get('brand', '').lower():
            return False
        
        # Calculate visual similarity
        emb_a = self.embeddings[idx_a:idx_a+1]
        emb_b = self.embeddings[idx_b:idx_b+1]
        visual_similarity = float(cosine_similarity(emb_a, emb_b)[0][0])
        
        # Calculate text similarity
        text_similarity = self._calculate_text_similarity(watch_a, watch_b)
        
        # Variant detection logic
        if visual_similarity >= self.visual_similarity_threshold:
            return True
        
        if (visual_similarity >= self.hybrid_visual_threshold and 
            text_similarity >= self.text_similarity_threshold):
            return True
        
        return False

    def _calculate_text_similarity(self, watch_a: Dict, watch_b: Dict) -> float:
        """Calculate text similarity between two watches."""
        # Get base model names (remove color/size variants)
        base_a = self._get_base_model_name(watch_a.get('model', ''))
        base_b = self._get_base_model_name(watch_b.get('model', ''))
        
        if not base_a or not base_b:
            return 0.0
        
        # Use SequenceMatcher for fuzzy string matching
        similarity = difflib.SequenceMatcher(None, base_a.lower(), base_b.lower()).ratio()
        return similarity

    def _get_base_model_name(self, model: str) -> str:
        """Extract base model name, removing variant indicators."""
        if not model:
            return ""
        
        # Common variant indicators to remove
        variant_patterns = [
            r'\s*\([^)]*\).*$',  # Remove parenthetical info and everything after
            r'\s*-.*$',          # Remove dash and everything after
            r'\s*\|.*$',         # Remove pipe and everything after
            r'\s*with.*$',       # Remove "with..." and after
            r'\s*\d+mm.*$',      # Remove size specifications
        ]
        
        base_name = model
        for pattern in variant_patterns:
            base_name = re.sub(pattern, '', base_name, flags=re.IGNORECASE)
        
        return base_name.strip().lower()

    def _choose_representative(self, variant_indices: List[int]) -> int:
        """Choose the best representative for a variant group."""
        # Prefer watches with better images (not placeholder)
        representatives = []
        
        for idx in variant_indices:
            watch = self.watch_data[idx]
            score = 0
            
            # Prefer watches with real images
            image_url = watch.get('image_url', '')
            if image_url and 'no-logo' not in image_url.lower():
                score += 10
            
            # Prefer watches with complete descriptions
            if len(watch.get('description', '')) > 50:
                score += 5
            
            # Prefer watches with specific model names (not empty or generic)
            model_name = watch.get('model', '')
            if len(model_name) > 5 and model_name.lower() not in ['-', 'n/a', 'unknown']:
                score += 5
            
            representatives.append((idx, score))
        
        # Sort by score and return best representative
        representatives.sort(key=lambda x: x[1], reverse=True)
        return representatives[0][0]

    def get_variant_group(self, watch_index: int) -> List[int]:
        """Get all variants of a given watch."""
        group_id = self.watch_to_group.get(watch_index)
        if group_id is not None:
            return self.variant_groups[group_id].copy()
        return [watch_index]  # Single watch, no variants

    def get_representative(self, watch_index: int) -> int:
        """Get the representative watch for a variant group."""
        group_id = self.watch_to_group.get(watch_index)
        if group_id is not None:
            return self.group_representatives[group_id]
        return watch_index  # Single watch

    def is_variant_of(self, watch_a: int, watch_b: int) -> bool:
        """Check if two watches are variants of each other."""
        group_a = self.watch_to_group.get(watch_a)
        group_b = self.watch_to_group.get(watch_b)
        return group_a is not None and group_a == group_b

    def filter_diverse_watches(self, watch_indices: List[int], max_variants_per_group: int = 1) -> List[int]:
        """Filter watch list to include only diverse models (limit variants per group)."""
        seen_groups = set()
        diverse_watches = []
        
        # First filter input indices to only include valid ones
        valid_indices = [idx for idx in watch_indices if 0 <= idx < len(self.watch_data)]
        
        for idx in valid_indices:
            group_id = self.watch_to_group.get(idx)
            
            if group_id is None:
                # Single watch, no variants
                diverse_watches.append(idx)
            elif group_id not in seen_groups:
                # First time seeing this variant group, use representative
                representative = self.group_representatives[group_id]
                # Ensure representative is valid
                if 0 <= representative < len(self.watch_data):
                    diverse_watches.append(representative)
                    seen_groups.add(group_id)
                else:
                    # Representative is invalid, use current index instead
                    print(f"Warning: Invalid representative {representative} for group {group_id}, using {idx} instead")
                    diverse_watches.append(idx)
                    seen_groups.add(group_id)
            # Skip additional variants from same group
        
        return diverse_watches

    def get_variant_stats(self) -> Dict[str, Any]:
        """Get statistics about variant detection."""
        total_watches = len(self.watch_data)
        watches_in_groups = sum(len(group) for group in self.variant_groups.values())
        single_watches = total_watches - watches_in_groups
        
        # Group size distribution
        group_sizes = [len(group) for group in self.variant_groups.values()]
        avg_group_size = np.mean(group_sizes) if group_sizes else 0
        max_group_size = max(group_sizes) if group_sizes else 0
        
        return {
            'total_watches': total_watches,
            'variant_groups': len(self.variant_groups),
            'watches_in_groups': watches_in_groups,
            'single_watches': single_watches,
            'average_group_size': avg_group_size,
            'max_group_size': max_group_size,
            'diversity_reduction': (total_watches - len(self.variant_groups) - single_watches) / total_watches * 100
        }

    def propagate_feedback_to_variants(self, watch_index: int, feedback_type: str, 
                                     confidence: float = 1.0, 
                                     variant_weight: float = 0.6) -> List[Tuple[int, str, float]]:
        """
        Propagate user feedback to variant watches with reduced weight.
        
        Returns:
            List of (watch_index, feedback_type, adjusted_confidence) for variants
        """
        variants = self.get_variant_group(watch_index)
        propagated_feedback = []
        
        for variant_idx in variants:
            if variant_idx != watch_index:  # Don't include original watch
                adjusted_confidence = confidence * variant_weight
                propagated_feedback.append((variant_idx, feedback_type, adjusted_confidence))
        
        return propagated_feedback 

    def show_variant_groups(self):
        """Show variant groups for debugging."""
        variant_groups = len(self.variant_groups)
        print(f"Found {variant_groups} variant groups:")
        for i, (representative, variants) in enumerate(list(self.variant_groups.items())[:5]):  # Show first 5
            print(f"  Group {i+1}: {self.watch_data[representative]['brand']} {self._get_base_model_name(self.watch_data[representative]['model'])}")
            print(f"    Variants: {len(variants)} watches")

    def are_variants(self, watch_a: Dict, watch_b: Dict) -> bool:
        """Check if two watches are variants of each other."""
        # Same brand required
        if watch_a.get('brand', '').lower() != watch_b.get('brand', '').lower():
            return False
        
        # Skip if either has missing/invalid model name
        base_a = self._get_base_model_name(watch_a.get('model', ''))
        base_b = self._get_base_model_name(watch_b.get('model', ''))
        
        if not base_a or not base_b:
            return False
        
        return base_a == base_b

    def has_meaningful_model_name(self, watch: Dict) -> bool:
        """Check if watch has a meaningful model name for variant detection."""
        model = watch.get('model', '')
        if len(model) > 5 and model.lower() not in ['-', 'n/a', 'unknown']:
            return True
        return False 