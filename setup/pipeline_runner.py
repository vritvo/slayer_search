"""
Main pipeline runner that orchestrates the full data processing pipeline.
This script is run once to build the database and embeddings.
"""

from setup.data_processor import make_scene_chunks, insert_window_db
from utils import make_embeddings


def run_full_pipeline():
    """Run the complete data processing pipeline."""
    print("Starting full data pipeline...")
    
    print("\n1. Creating scene chunks...")
    make_scene_chunks()  # Create scene chunks in database
    
    print("\n2. Creating window chunks...")
    insert_window_db()  # Create window chunks in database
    
    print("\n3. Creating scene embeddings...")
    make_embeddings("scene")  # Create embeddings for scene chunks
    
    print("\n4. Creating window embeddings...")
    make_embeddings("window")  # Create embeddings for window chunks
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    run_full_pipeline()
