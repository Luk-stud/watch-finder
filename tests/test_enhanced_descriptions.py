#!/usr/bin/env python3
import os
import json
import pickle
from openai import OpenAI
from dotenv import load_dotenv
import time
import random

# Load environment variables
load_dotenv()

class EnhancedWatchDescriptionGenerator:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.description_model = "gpt-4"  # Using GPT-4 for better quality
        
    def build_color_context(self, specs):
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

    def generate_description(self, brand, model, specs=None):
        """Generate an enhanced watch description using the structured multi-aspect prompt."""
        
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

def test_description_generation():
    """Test the enhanced description generation with a variety of watches."""
    
    # Load some test watches from the metadata
    print("Loading watch metadata...")
    with open('backend/data/watch_metadata.pkl', 'rb') as f:
        all_watches = pickle.load(f)
    
    # Get watches with good color information
    watches_with_color = [
        w for w in all_watches 
        if w.get('specs', {}).get('dial_color') 
        and w.get('specs', {}).get('dial_color') != '-'
    ]
    
    # Select 5 random watches with color information
    test_watches = random.sample(watches_with_color, min(5, len(watches_with_color)))
    
    # Initialize generator
    generator = EnhancedWatchDescriptionGenerator()
    
    # Test results
    results = []
    
    print(f"\nGenerating enhanced descriptions for {len(test_watches)} watches...")
    for watch in test_watches:
        print(f"\nProcessing {watch['brand']} {watch['model']}...")
        print(f"Specs: {json.dumps(watch.get('specs', {}), indent=2)}")
        
        # Generate description
        description = generator.generate_description(
            watch['brand'],
            watch['model'],
            watch.get('specs', {})
        )
        
        # Store result
        result = {
            'brand': watch['brand'],
            'model': watch['model'],
            'specs': watch.get('specs', {}),
            'enhanced_description': description
        }
        results.append(result)
        
        # Print description
        print(f"\n{watch['brand']} {watch['model']}:")
        print(f"{description}\n")
        print("-" * 80)
        
        # Rate limit
        time.sleep(1)
    
    # Save results
    with open('enhanced_description_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest results saved to enhanced_description_test_results.json")
    return results

if __name__ == "__main__":
    print("🎯 Testing Enhanced Watch Description Generation")
    print("=" * 50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        exit(1)
    
    results = test_description_generation() 