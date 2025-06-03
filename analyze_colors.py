#!/usr/bin/env python3
import pickle
import json
from collections import Counter
import re

def extract_colors_from_description(description):
    """Extract color mentions from a watch description."""
    # Common watch colors and their variations
    color_patterns = [
        r'black', r'blue', r'white', r'green', r'grey|gray', r'brown',
        r'silver', r'purple', r'yellow', r'orange', r'red', r'gold',
        r'rose gold', r'bronze', r'copper', r'platinum', r'titanium',
        r'cream', r'navy', r'burgundy', r'pink', r'salmon'
    ]
    
    # Create a pattern that matches any color, case insensitive
    combined_pattern = '|'.join(color_patterns)
    
    # Find all color mentions
    colors = re.findall(combined_pattern, description.lower())
    
    # Normalize grey/gray
    colors = ['grey' if c == 'gray' else c for c in colors]
    
    return colors

# Load watch metadata
print("Loading watch metadata...")
with open('backend/data/watch_metadata.pkl', 'rb') as f:
    watches = pickle.load(f)

# Initialize counters
color_mentions = Counter()
watches_with_color = 0
total_watches = len(watches)
color_contexts = {}  # Store example contexts for each color

for watch in watches:
    if 'description' in watch:
        description = watch['description']
        colors = extract_colors_from_description(description)
        
        if colors:
            watches_with_color += 1
            for color in colors:
                color_mentions[color] += 1
                
                # Store an example context if we don't have one yet
                if color not in color_contexts and len(color_contexts) < 50:
                    # Find the sentence containing the color
                    sentences = description.split('.')
                    for sentence in sentences:
                        if color.lower() in sentence.lower():
                            color_contexts[color] = sentence.strip()
                            break

# Print results
print(f"\n=== Color Analysis in AI Descriptions ===")
print(f"Total watches: {total_watches}")
print(f"Watches with color mentions: {watches_with_color} ({(watches_with_color/total_watches)*100:.1f}%)")

print(f"\nTop 20 Colors Mentioned:")
for color, count in color_mentions.most_common(20):
    print(f"  {color}: {count} mentions ({(count/total_watches)*100:.1f}%)")

print(f"\nExample Color Contexts:")
for color, context in color_contexts.items():
    if color_mentions[color] > 10:  # Only show contexts for colors with >10 mentions
        print(f"\n{color.title()}:")
        print(f"  {context}")

# Save detailed results
results = {
    'total_watches': total_watches,
    'watches_with_color': watches_with_color,
    'color_mentions': dict(color_mentions),
    'color_contexts': color_contexts
}

with open('color_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2) 