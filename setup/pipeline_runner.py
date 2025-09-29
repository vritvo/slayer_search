"""
Main pipeline runner that orchestrates the full data processing pipeline.
This script is run once to build the database and embeddings.
"""

from setup.data_processor import make_scene_chunks, insert_window_db
from utils import (
    make_embeddings,
    init_scene_tables,
    init_window_tables,
    clear_table,
    get_db_path,
)
import toml
import argparse
import time
import gc


def run_full_pipeline(embedding_model: str = "sbert"):
    """Run the complete data processing pipeline."""
    print(f"Running embeddings with {embedding_model}")
    print(f"Using database: {get_db_path(embedding_model)}")
    print("Starting full data pipeline...")

    # Initialize tables with the specified embedding model
    print("Initializing scene tables...")
    init_scene_tables("scene", embedding_model)
    time.sleep(0.5)
    gc.collect()  # Force garbage collection

    print("Initializing window tables...")
    init_window_tables("window", embedding_model)
    time.sleep(0.5)
    gc.collect()

    # Clear existing data
    print("Clearing existing data...")
    for table in ["scene", "window"]:
        print(f"Clearing table: {table}")
        clear_table(table, embedding_model)
        time.sleep(0.5)
        gc.collect()

    print("\n1. Creating scene chunks...")
    make_scene_chunks(embedding_model)  # Create scene chunks in database
    time.sleep(1.0)
    gc.collect()

    print("\n2. Creating window chunks...")
    insert_window_db(embedding_model)  # Create window chunks in database
    time.sleep(1.0)
    gc.collect()

    print("\n3. Creating scene embeddings...")
    make_embeddings("scene", embedding_model)  # Create embeddings for scene chunks

    print("\n4. Creating window embeddings...")
    make_embeddings("window", embedding_model)  # Create embeddings for window chunks

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data processing pipeline")
    parser.add_argument(
        "--embedding_model",
        "--e",
        type=str,
        help="Embedding model to use (sbert or openAI)",
        default="sbert",
    )

    args = parser.parse_args()

    # Validate embedding model
    if args.embedding_model not in ["sbert", "openAI"]:
        raise ValueError("Invalid embedding_model. Must be 'sbert' or 'openAI'.")

    run_full_pipeline(args.embedding_model)
