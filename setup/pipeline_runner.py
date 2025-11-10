"""
Main pipeline runner that orchestrates the full data processing pipeline.
This script is run once to build the database and embeddings.
"""

from setup.data_processor import (
    make_scene_chunks,
    insert_window_db,
)
from utils.database import init_scene_tables, init_window_tables, clear_table
from utils.models import initialize_models, make_embeddings
import time
import gc
import argparse


def run_full_pipeline(split_scenes):
    """Run the complete data processing pipeline."""
    print("Running embeddings")
    print("Starting full data pipeline...")
    # Initialize tables
    print("Initializing scene tables...")
    init_scene_tables()
    time.sleep(0.5)
    gc.collect()  # Force garbage collection

    print("Initializing window tables...")
    init_window_tables()
    time.sleep(0.5)
    gc.collect()

    # Clear existing data
    print("Clearing existing data...")
    for table in ["scene", "window", "window_vss"]:
        print(f"Clearing table: {table}")
        clear_table(table)
        time.sleep(0.5)
        gc.collect()

    print("\n1. Creating scene chunks...")
    make_scene_chunks()  # Create scene chunks in database
    time.sleep(1.0)
    gc.collect()

    print("\n2. Creating window chunks...")
    insert_window_db()  # Create window chunks in database
    time.sleep(1.0)
    gc.collect()

    print("\n3. Creating window embeddings...")
    initialize_models()
    make_embeddings()  # Create embeddings for window chunks only

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full data processing pipeline"
    )
    parser.add_argument(
        "--split-scenes",
        "-ss",
        action="store_true",
        help="Split scenes into smaller chunks",
    )
    args = parser.parse_args()

    split_scenes = args.split_scenes
    run_full_pipeline(split_scenes)
