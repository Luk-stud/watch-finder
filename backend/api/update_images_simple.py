#!/usr/bin/env python3
"""
Simple script to update app.py to use local images
"""
import os

def update_app_py():
    # Read current app.py
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Update the image URL function in app.py
    new_get_image_url = '''def get_image_url(watch_data):
    """Get image URL for a watch - now uses local images"""
    if 'main_image' in watch_data and watch_data['main_image']:
        # Extract brand and model to create local filename
        brand = str(watch_data.get('brand', ''))
        model = str(watch_data.get('model', ''))
        
        # Clean the brand and model names to match downloaded files
        brand_clean = brand.replace(' ', '_').replace('/', '_')
        model_clean = model.replace(' ', '_').replace('/', '_')
        
        # Return local image path
        return f"/images/{brand_clean}_{model_clean}_main.jpg"
    
    return "/images/placeholder-watch.jpg"  # Fallback'''
    
    # Find and replace the existing get_image_url function
    if 'def get_image_url(watch_data):' in content:
        # Find start and end of the function
        lines = content.split('\n')
        new_lines = []
        skip_function = False
        
        for line in lines:
            if line.startswith('def get_image_url(watch_data):'):
                skip_function = True
                new_lines.extend(new_get_image_url.split('\n'))
                continue
            elif skip_function and line.startswith('def ') and not line.startswith('def get_image_url'):
                skip_function = False
                new_lines.append(line)
            elif not skip_function:
                new_lines.append(line)
        
        content = '\n'.join(new_lines)
    else:
        # Add the function if it doesn't exist
        content += '\n\n' + new_get_image_url + '\n'
    
    # Write back to app.py
    with open('app.py', 'w') as f:
        f.write(content)
    
    print("Updated app.py to use local images!")

if __name__ == "__main__":
    update_app_py() 