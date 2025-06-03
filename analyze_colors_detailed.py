#!/usr/bin/env python3
import pickle
import json
from collections import Counter, defaultdict
import re

def extract_colors_from_description(description):
    """Extract color mentions from a watch description."""
    # Extended list of watch colors and their variations
    color_patterns = {
        'black': ['black', 'onyx', 'ebony'],
        'blue': ['blue', 'navy', 'azure', 'cerulean', 'midnight blue'],
        'white': ['white', 'ivory', 'cream', 'pearl'],
        'green': ['green', 'olive', 'emerald', 'forest'],
        'grey': ['grey', 'gray', 'charcoal', 'slate'],
        'brown': ['brown', 'chocolate', 'coffee', 'tobacco'],
        'silver': ['silver', 'rhodium'],
        'purple': ['purple', 'violet', 'amethyst'],
        'yellow': ['yellow', 'champagne'],
        'orange': ['orange', 'coral', 'salmon'],
        'red': ['red', 'burgundy', 'maroon', 'crimson'],
        'gold': ['gold', 'gilt'],
        'rose gold': ['rose gold', 'pink gold'],
        'bronze': ['bronze', 'copper'],
        'platinum': ['platinum'],
        'titanium': ['titanium'],
        'pink': ['pink', 'salmon pink'],
        'cream': ['cream', 'eggshell'],
        'navy': ['navy'],
        'burgundy': ['burgundy'],
        'salmon': ['salmon']
    }
    
    # Create pattern groups
    all_patterns = []
    color_mapping = {}
    for main_color, variations in color_patterns.items():
        for variant in variations:
            all_patterns.append(variant)
            color_mapping[variant] = main_color
    
    # Create regex pattern
    pattern = r'\b(' + '|'.join(all_patterns) + r')\b'
    
    # Find all color mentions
    found_colors = re.findall(pattern, description.lower())
    
    # Map variants to main colors
    normalized_colors = [color_mapping[color] for color in found_colors]
    
    return normalized_colors

def analyze_color_context(description, colors):
    """Analyze the context in which colors are mentioned."""
    contexts = {
        'dial': ['dial', 'face'],
        'case': ['case', 'housing', 'body'],
        'bezel': ['bezel', 'ring'],
        'hands': ['hands', 'indices', 'markers'],
        'strap': ['strap', 'bracelet', 'band'],
        'aesthetic': ['aesthetic', 'design', 'style', 'look', 'appearance']
    }
    
    color_contexts = defaultdict(list)
    
    # For each color found
    for color in colors:
        # Get the sentence containing the color
        sentences = [s for s in description.lower().split('.') if color in s]
        for sentence in sentences:
            # Check each context type
            found_context = False
            for context_type, context_words in contexts.items():
                if any(word in sentence for word in context_words):
                    color_contexts[color].append(context_type)
                    found_context = True
            if not found_context:
                color_contexts[color].append('general')
    
    return dict(color_contexts)

# Load watch metadata
print("Loading watch metadata...")
with open('backend/data/watch_metadata.pkl', 'rb') as f:
    watches = pickle.load(f)

# Initialize analysis structures
color_mentions = Counter()
watches_with_color = 0
total_watches = len(watches)
color_contexts = defaultdict(list)
color_combinations = Counter()
context_distribution = defaultdict(Counter)
brand_color_preferences = defaultdict(Counter)
example_descriptions = defaultdict(list)

print(f"\nAnalyzing {total_watches} watches...")

for watch in watches:
    if 'ai_description' in watch:
        description = watch['ai_description']
        colors = extract_colors_from_description(description)
        
        if colors:
            watches_with_color += 1
            
            # Count individual colors
            for color in colors:
                color_mentions[color] += 1
                
                # Store example descriptions (up to 3 per color)
                if len(example_descriptions[color]) < 3:
                    example_descriptions[color].append({
                        'brand': watch['brand'],
                        'model': watch['model'],
                        'description': description
                    })
            
            # Analyze color combinations (if multiple colors)
            if len(colors) > 1:
                combo = tuple(sorted(set(colors)))  # Remove duplicates and sort
                color_combinations[combo] += 1
            
            # Analyze context
            contexts = analyze_color_context(description, colors)
            for color, context_list in contexts.items():
                for context in context_list:
                    context_distribution[color][context] += 1
            
            # Track brand preferences
            brand = watch['brand']
            for color in colors:
                brand_color_preferences[brand][color] += 1

# Calculate percentages
color_percentages = {color: (count/total_watches)*100 
                    for color, count in color_mentions.items()}

# Prepare results
results = {
    'total_watches': total_watches,
    'watches_with_color': watches_with_color,
    'color_percentages': color_percentages,
    'color_combinations': {' + '.join(combo): count 
                         for combo, count in color_combinations.most_common(10)},
    'context_distribution': {color: dict(contexts.most_common()) 
                           for color, contexts in context_distribution.items()},
    'brand_preferences': {brand: dict(colors.most_common(3)) 
                         for brand, colors in brand_color_preferences.items() 
                         if sum(colors.values()) >= 5},  # Only brands with 5+ color mentions
    'example_descriptions': example_descriptions
}

# Print analysis
print(f"\n=== Color Analysis in AI Descriptions ===")
print(f"Total watches: {total_watches}")
print(f"Watches with color mentions: {watches_with_color} ({(watches_with_color/total_watches)*100:.1f}%)")

print(f"\nTop 10 Colors by Mention Frequency:")
for color, count in color_mentions.most_common(10):
    print(f"  {color}: {count} mentions ({color_percentages[color]:.1f}%)")

print(f"\nTop 5 Color Combinations:")
for combo, count in list(color_combinations.most_common(5)):
    print(f"  {' + '.join(combo)}: {count} occurrences")

print(f"\nColor Usage Contexts (Top 3 per color):")
for color, contexts in context_distribution.items():
    if color_mentions[color] >= 10:  # Only show for colors with 10+ mentions
        print(f"\n{color.title()}:")
        for context, count in contexts.most_common(3):
            print(f"  {context}: {count} mentions")

print(f"\nExample Descriptions:")
for color, examples in example_descriptions.items():
    if color_mentions[color] >= 20:  # Only show for colors with 20+ mentions
        print(f"\n{color.title()}:")
        for example in examples[:1]:  # Show just 1 example per color
            print(f"  {example['brand']} {example['model']}:")
            print(f"  \"{example['description']}\"")

# Save detailed results
with open('color_analysis_detailed.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nDetailed results saved to color_analysis_detailed.json") 