#!/usr/bin/env python3
"""
Run text embedding generation on 500 watches for testing.
"""

from generate_text_embeddings import main

if __name__ == "__main__":
    print("🧪 Running 500-watch test...")
    main(max_watches=500) 