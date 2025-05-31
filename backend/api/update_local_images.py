#!/usr/bin/env python3
"""
Update CSV to use local images instead of external URLs
"""
import pandas as pd
import os

def main():
    # Read the CSV
    csv_path = "../../production_scrape_20250531_231634/data/final_scrape.csv"
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} watches from CSV")
    
    # Function to convert external URL to local image path
    def get_local_image_path(row):
        brand = row['brand']
        model = row['model']
        
        # Clean the brand and model names to match the downloaded file naming
        brand_clean = brand.replace(' ', '_').replace('/', '_')
        model_clean = model.replace(' ', '_').replace('/', '_')
        
        # The downloaded images follow this pattern: Brand_Model_main.jpg
        filename = f"{brand_clean}_{model_clean}_main.jpg"
        
        # Return the local path
        return f"/images/{filename}"
    
    # Update the main_image column
    df['main_image'] = df.apply(get_local_image_path, axis=1)
    
    # Save to a new CSV
    output_path = "../../watch_data/final_scrape_local_images.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Updated CSV saved to {output_path}")
    print("Sample local image paths:")
    print(df['main_image'].head(10).tolist())

if __name__ == "__main__":
    main() 