import pickle
import os
import textwrap

def inspect_descriptions():
    """
    Loads the watch metadata and prints the AI-generated descriptions
    for a few representative watches to analyze the content of the
    text embeddings.
    """
    
    # --- Configuration ---
    # Pick a few watches to inspect. These indices were chosen
    # after manually looking for a classic Diver and a classic Dress watch.
    WATCH_IDS_TO_INSPECT = {
        "Diver Watch": 25,
        "Dress Watch": 45
    }
    # ---

    print("--- 🕵️‍♂️ Inspecting AI Descriptions ---")
    
    # --- Load Data ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        metadata_path = os.path.join(script_dir, 'data/watch_text_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata_list = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading watch_text_metadata.pkl: {e}")
        return

    # --- Print Descriptions ---
    for name, watch_id in WATCH_IDS_TO_INSPECT.items():
        watch_data = next((w for w in metadata_list if w.get('index') == watch_id), None)
        
        print(f"\\n--- ✅ {name} (ID: {watch_id}) ---")
        if watch_data and 'ai_description' in watch_data:
            print(f"Brand: {watch_data.get('brand')}, Model: {watch_data.get('model')}")
            print("\\nDescription:")
            # Use textwrap for nice formatting
            wrapped_text = textwrap.fill(watch_data['ai_description'], width=80)
            print(wrapped_text)
        else:
            print(f"Watch {watch_id} not found or has no AI description.")

if __name__ == "__main__":
    inspect_descriptions() 