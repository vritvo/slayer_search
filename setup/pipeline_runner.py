"""
Main pipeline runner that orchestrates the full data processing pipeline.
This script is run once to build the database and embeddings.
"""

from setup.data_processor import (
    make_scene_chunks,
    insert_window_db,
    tag_scene_locations,
)
from utils.database import init_scene_tables, init_window_tables, clear_table
from utils.models import initialize_models, make_embeddings
import time
import gc
import argparse


def run_full_pipeline(tag_locations=False, filter_episodes=None):
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

    # Optional: Tag scenes with locations
    if tag_locations:
        print("\n2. Tagging scene locations...")
        tag_scene_locations(filter_episodes=filter_episodes)
        time.sleep(1.0)
        gc.collect()

    print("\n3. Creating window chunks...")
    insert_window_db()  # Create window chunks in database
    time.sleep(1.0)
    gc.collect()

    print("\n4. Creating window embeddings...")
    initialize_models()
    make_embeddings()  # Create embeddings for window chunks only

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full data processing pipeline"
    )

    parser.add_argument(
        "--tag-locations",
        "-tl",
        action="store_true",
        help="Tag scenes with location information using LangExtract",
    )
    parser.add_argument(
        "--filter-episodes",
        "-fe",
        nargs="+",
        help="Only process specific episodes (e.g., '1x12 Prophecy Girl' '5x07 Fool For Love')",
    )
    args = parser.parse_args()

    tag_locations = args.tag_locations
    filter_episodes = args.filter_episodes
    
    run_full_pipeline(tag_locations=tag_locations, filter_episodes=filter_episodes)
