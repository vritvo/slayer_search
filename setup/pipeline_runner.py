"""
Main pipeline runner that orchestrates the full data processing pipeline.
This script is run once to build the database and embeddings.
"""

from setup.data_processor import (
    make_scene_chunks,
    insert_window_db,
    tag_scene_locations,
    load_locations_from_json,
)
from utils.database import init_scene_tables, init_window_tables, clear_table, get_db_path
from utils.models import initialize_models, make_embeddings
import time
import gc
import argparse
import toml


def run_full_pipeline(tag_locations=False, filter_episodes=None, use_gpu=False, model_name=None, include_meta=None, db_tag=None):
    """Run the complete data processing pipeline.
    
    Args:
        tag_locations: Tag scenes with location information
        filter_episodes: Only process specific episodes
        use_gpu: Use GPU for batch embedding building
        model_name: Optional model name. If any of model_name, include_meta, db_tag is provided, all three must be provided.
        include_meta: Optional flag for including metadata. If any is provided, all three must be provided.
        db_tag: Optional experiment tag. If any is provided, all three must be provided.
    """
    # Get config for defaults
    config = toml.load("config.toml")
    
    # If no params provided, use config defaults
    # Otherwise, _resolve_model_config() will validate all-or-nothing
    if model_name is None and include_meta is None and db_tag is None:
        model_name = config["EMBEDDING_MODEL"]["model_name"]
        include_meta = config["EMBEDDING_MODEL"].get("include_meta", True)
        db_tag = config["EMBEDDING_MODEL"].get("db_tag", "")
    
    print("Running embeddings")
    print("Starting full data pipeline...")
    print(f"Model: {model_name}")
    print(f"Include metadata: {include_meta}")
    print(f"DB tag: {db_tag if db_tag else '(empty)'}")
    
    # Initialize tables
    print("Initializing scene tables...")
    init_scene_tables(model_name=model_name, include_meta=include_meta, db_tag=db_tag)
    time.sleep(0.5)
    gc.collect()  # Force garbage collection

    print("Initializing window tables...")
    init_window_tables(model_name=model_name, include_meta=include_meta, db_tag=db_tag)
    time.sleep(0.5)
    gc.collect()

    # Clear existing data
    print("Clearing existing data...")
    db_path = get_db_path(model_name, include_meta, db_tag)
    for table in ["scene", "window", "window_vss"]:
        print(f"Clearing table: {table}")
        clear_table(table, db_path=db_path)
        time.sleep(0.5)
        gc.collect()

    print("\n1. Creating scene chunks...")
    make_scene_chunks(db_path=db_path)  # Create scene chunks in database
    time.sleep(1.0)
    gc.collect()

    # Optional: Tag NEW scenes with locations (updates JSON only)
    if tag_locations:
        print("\n2. Tagging NEW scene locations (updates JSON)...")
        tag_scene_locations(filter_episodes=filter_episodes, db_path=db_path)
        time.sleep(1.0)
        gc.collect()

    print(
        f"\n{3 if tag_locations else 2}. Loading location data from JSON to database..."
    )
    load_locations_from_json(db_path=db_path)  # Load ALL location data from JSON into DB
    time.sleep(0.5)
    gc.collect()

    print(f"\n{4 if tag_locations else 3}. Creating window chunks...")
    insert_window_db(db_path=db_path)  # Create window chunks in database
    time.sleep(1.0)
    gc.collect()

    print(f"\n{5 if tag_locations else 4}. Creating window embeddings...")
    initialize_models(model_name=model_name, include_meta=include_meta, use_gpu_for_building=use_gpu)
    make_embeddings(include_meta=include_meta, model_name=model_name, db_tag=db_tag)  # Create embeddings for window chunks only

    # Print summary
    db_path = get_db_path(model_name, include_meta, db_tag)
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print(f"Database created at: {db_path}")
    print(f"Model: {model_name}")
    print(f"Metadata included: {include_meta}")
    if db_tag:
        print(f"DB tag: {db_tag}")
    print("="*60)


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
    parser.add_argument(
        "--use-gpu",
        "-gpu",
        action="store_true",
        help="Use GPU for batch embedding building",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Model name to use (e.g., 'jxm/cde-small-v2'). If any of --model, --include-meta, --db-tag is provided, all three must be provided.",
    )
    parser.add_argument(
        "--include-meta",
        type=str,
        choices=['true', 'false'],
        help="Whether to include location metadata in embeddings (true/false). If any is provided, all three must be provided.",
    )
    parser.add_argument(
        "--db-tag",
        type=str,
        default=None,
        help="Optional experiment tag (e.g., 'x2' for double-sampling). If any is provided, all three must be provided.",
    )
    args = parser.parse_args()

    # Validate all-or-nothing: if any of model, include_meta, db_tag is provided, all must be provided
    model_provided = args.model is not None
    meta_provided = args.include_meta is not None
    tag_provided = args.db_tag is not None
    
    params_provided = [model_provided, meta_provided, tag_provided]
    if any(params_provided) and not all(params_provided):
        missing = []
        if not model_provided:
            missing.append("--model")
        if not meta_provided:
            missing.append("--include-meta")
        if not tag_provided:
            missing.append("--db-tag")
        parser.error(
            f"Must provide all three parameters (--model, --include-meta, --db-tag) or none. "
            f"Missing: {', '.join(missing)}. "
            f"This prevents accidentally inheriting config values (like db_tag) meant for a different model."
        )
    
    # Convert include_meta string to boolean
    include_meta = None
    if args.include_meta is not None:
        include_meta = args.include_meta.lower() == 'true'
    
    # db_tag can be empty string if explicitly provided (e.g., --db-tag "" to override config)
    db_tag = args.db_tag

    tag_locations = args.tag_locations
    filter_episodes = args.filter_episodes
    use_gpu = args.use_gpu

    run_full_pipeline(
        tag_locations=tag_locations,
        filter_episodes=filter_episodes,
        use_gpu=use_gpu,
        model_name=args.model,
        include_meta=include_meta,
        db_tag=db_tag
    )
