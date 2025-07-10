#!/usr/bin/env python3
import os
import json
import pickle
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_enhanced_description(watch):
    """Generate an enhanced description for a single watch."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    
    client = OpenAI(api_key=api_key)
    
    # Build color context
    specs = watch.get('specs', {})
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
    color_context = ", " + ", ".join(context_elements) if context_elements else ""
    
    # Enhanced system message
    system_message = """You are an expert watch designer and critic with deep knowledge of:
- Horological aesthetics and design language
- Watch materials, finishes, and their visual impact
- How different design elements interact
- The relationship between watches and lifestyle
- Color theory and material combinations in watchmaking

Focus on creating descriptions that are both precise and evocative, connecting specific details to their broader impact on the watch's character."""

    # Structured multi-aspect prompt
    prompt = f"""Describe the {watch['brand']} {watch['model']} watch{color_context} by covering these key aspects:

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
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=temperature
        )
        
        description = response.choices[0].message.content.strip()
        return description
        
    except Exception as e:
        print(f"Error generating description: {e}")
        # Fallback description incorporating available color context
        fallback = f"A distinctive timepiece from {watch['brand']}, the {watch['model']}"
        if color_context:
            fallback += f"{color_context}"
        fallback += " represents the brand's signature design approach with its unique aesthetic character."
        return fallback

def test_single_watch():
    """Test the enhanced description generation with a single watch."""
    
    # Load watch metadata
    print("Loading watch metadata...")
    with open('backend/data/watch_metadata.pkl', 'rb') as f:
        all_watches = pickle.load(f)
    
    # Find a watch with good color information
    test_watch = None
    for watch in all_watches:
        specs = watch.get('specs', {})
        if (specs.get('dial_color') and specs.get('dial_color') != '-' and
            specs.get('case_material') and specs.get('case_material') != '-'):
            test_watch = watch
            break
    
    if not test_watch:
        print("No suitable test watch found!")
        return
    
    print("\nTest watch selected:")
    print(f"Brand: {test_watch['brand']}")
    print(f"Model: {test_watch['model']}")
    print("\nSpecifications:")
    print(json.dumps(test_watch.get('specs', {}), indent=2))
    
    print("\nGenerating enhanced description...")
    description = generate_enhanced_description(test_watch)
    
    result = {
        'brand': test_watch['brand'],
        'model': test_watch['model'],
        'specs': test_watch.get('specs', {}),
        'enhanced_description': description
    }
    
    # Save result
    with open('single_watch_test_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\nEnhanced Description:")
    print("-" * 80)
    print(description)
    print("-" * 80)
    print("\nResult saved to single_watch_test_result.json")

if __name__ == "__main__":
    print("🎯 Testing Enhanced Watch Description Generation")
    print("=" * 50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        exit(1)
    
    test_single_watch() 