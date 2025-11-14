import toml
import os
import hashlib
import sqlite3
from utils.database import get_db_connection
from utils.data_access import iter_scenes
import json
from utils.models import tag_text


def make_scene_chunks():
    """Process script chunks and insert them row-by-row into the database."""

    con = get_db_connection()
    cur = con.cursor()

    table_name = "scene"

    config = toml.load("config.toml")
    scene_split_markers = config["WINDOW"]["scene_split_markers"]

    with open("scene_splits/all_scene_splits.json", "r") as file:
        print('Loading scene splits...')
        all_scene_splits = json.load(file)
        exception_scene_dict = convert_scene_splits_to_dict(all_scene_splits)

    try:
        # Process all files in the scripts directory
        for file_name in os.listdir("scripts"):
        # for file_name in [
        #     "1x02 The Harvest.txt",
        #     "2x01 When She Was Bad.txt",
        #     "3x01 Anne.txt", 
        #     "4x01 The Freshman.txt",
        #     "5x07 Fool For Love.txt",
        #     "6x08 Tabula Rasa .txt",
        #     "7x02 Beneath You.txt"
        # ]:
            if not file_name.endswith(".txt"):
                continue

            print(f"Splitting file: {file_name}")

            with open(f"scripts/{file_name}", "r") as f:
                script = f.read()

            # Split on lines that start with "cut" (case insensitive)
            lines = script.split("\n")

            # Initialize variables.
            # curr_chunk = a list of individual lines for the current scene. Gets reset each time a "cut" is encountered.
            curr_chunk = []  # current scene chunk
            chunk_index = 0
            episode_name = file_name.replace(".txt", "")

            for line in lines:
                # If we've hit a new scene, start a new chunk.
                line_lower = line.strip().lower()

                # If the line starts with a scene split marker or is in the exception scene list, start a new chunk.
                if any(line_lower.startswith(marker) for marker in scene_split_markers) | (episode_name in exception_scene_dict and line_lower in exception_scene_dict[episode_name]):
                    
                    # If we have accumulated lines, save as a chunk and insert into DB
                    if curr_chunk:
                        chunk_text = "\n".join(curr_chunk).strip()
                        if chunk_text:
                            print(f"  Splitting chunk {chunk_index}")

                            # Compute hash for this scene
                            scene_bytes = chunk_text.encode("utf-8")
                            scene_hash = hashlib.sha256(scene_bytes).hexdigest()

                            # Insert into main table
                            cur.execute(
                                f"""
                                INSERT INTO {table_name}
                                (file_name, scene_id_in_episode, scene_text, scene_hash)
                                VALUES (?, ?, ?, ?)
                            """,
                                (
                                    episode_name,
                                    chunk_index,
                                    chunk_text,
                                    scene_hash,
                                ),
                            )
                            chunk_index += 1

                    # Reset current_scene_chunk to be this first line of the new scene
                    curr_chunk = [line]

                # If the new line is not a new scene, we append the line to the current scene list.
                else:
                    curr_chunk.append(line)

            # Add the final chunk if it exists
            if curr_chunk:
                chunk_text = "\n".join(curr_chunk).strip()
                if chunk_text:
                    print(f"  Processing chunk {chunk_index}")

                    # Compute hash for this scene
                    scene_bytes = chunk_text.encode("utf-8")
                    scene_hash = hashlib.sha256(scene_bytes).hexdigest()

                    # Insert into main table
                    cur.execute(
                        f"""
                        INSERT INTO {table_name}
                        (file_name, scene_id_in_episode, scene_text, scene_hash)
                        VALUES (?, ?, ?, ?)
                    """,
                        (
                            episode_name,
                            chunk_index,
                            chunk_text,
                            scene_hash,
                        ),
                    )

        con.commit()
        print(f"Successfully inserted scene chunks into table: `{table_name}`")

    except Exception as e:
        print(f"Error in make_scene_chunks: {e}")
    finally:
        con.close()


def make_window_chunk(chunk):
    """Create window chunks from a text chunk."""

    # Load window configuration
    config = toml.load("config.toml")
    window_size = config["WINDOW"]["window_size"]
    step_size = config["WINDOW"]["step_size"]

    # Split script into "formatted lines" (by double newlines)
    # This keeps speaker names with their dialogue
    formatted_lines = chunk.split("\n\n")

    # Remove empty formatted lines
    formatted_lines = [line.strip() for line in formatted_lines if line.strip()]

    script_chunks = []
    w_start = 0

    # slices of formatted lines
    slices = []

    while w_start < len(formatted_lines):
        # Create window of formatted lines
        w_end = min(w_start + window_size, len(formatted_lines))
        window_formatted_lines = formatted_lines[w_start:w_end]
        slices.append([w_start, w_end])

        # Join the formatted lines back with double newlines
        chunk_text = "\n\n".join(window_formatted_lines).strip()
        if chunk_text:
            script_chunks.append(chunk_text)

        # Move window forward by step_size
        w_start += step_size

        # Break if we've reached the end
        if w_end >= len(formatted_lines):
            break

    return script_chunks, slices


def iter_windows_from_scenes():
    """Read each scene, make windows from them."""

    for scene in iter_scenes():
        windows, slices = make_window_chunk(scene["text"])

        for w_idx, (w_text, slice) in enumerate(zip(windows, slices)):
            yield {
                "scene_id": scene["scene_id"],
                "window_id_in_scene": w_idx,
                "window_start": slice[0],
                "window_end": slice[1],
                "window_text": w_text,
                "file_name": scene["file_name"],
            }


def insert_window_db():
    """Insert window chunks into the database using batch processing."""

    con = get_db_connection()
    cur = con.cursor()

    table_name = "window"
    batch_size = 1000  # Process windows in batches
    batch = []
    total_processed = 0

    try:
        for row in iter_windows_from_scenes():
            batch.append(
                (
                    row["scene_id"],
                    row["window_id_in_scene"],
                    row["window_start"],
                    row["window_end"],
                    row["window_text"],
                    row["file_name"],
                )
            )

            # Process batch when it reaches batch_size
            if len(batch) >= batch_size:
                cur.executemany(
                    f"""
                    INSERT INTO {table_name}
                    (scene_id, window_id_in_scene, window_start, window_end, window_text, file_name)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    batch,
                )
                con.commit()  # Commit each batch
                total_processed += len(batch)
                print(f"Processed {total_processed} window chunks...")
                batch = []  # Reset batch

        # Process any remaining windows in the final batch
        if batch:
            cur.executemany(
                f"""
                INSERT INTO {table_name}
                (scene_id, window_id_in_scene, window_start, window_end, window_text, file_name)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                batch,
            )
            con.commit()
            total_processed += len(batch)

        print(
            f"Successfully inserted {total_processed} window chunks into table: `{table_name}`"
        )

    except Exception as e:
        print(f"Error in insert_window_db: {e}")
        con.rollback()  # Rollback on error
    finally:
        con.close()

def convert_scene_splits_to_dict(all_scene_splits: list[dict]) -> dict:
    """Convert scene splits to dictionary where key is the episode title."""
    exception_scene_dict = {}
    for episode in all_scene_splits:
        exception_scene_dict[episode["episode_title"]] = [scene.lower() for scene in episode["scene_breaks"]]
    return exception_scene_dict


def load_locations_from_json(json_path="scene_locations.json"):
    """
    Load location data from JSON and populate the scene table.
    This is run at the start of each pipeline rebuild to restore location data.
    
    Args:
        json_path: Path to JSON file containing location data
    """
    try:
        with open(json_path, "r") as f:
            location_data = json.load(f)
        
        if not location_data:
            print("Location JSON is empty")
            return
        
        con = get_db_connection()
        cur = con.cursor()
        
        updated_count = 0
        for scene_hash, data in location_data.items():
            cur.execute("""
                UPDATE scene 
                SET location_text = ?, location_descr = ?
                WHERE scene_hash = ?
            """, (data.get('location_text', ''), data.get('location_descr', ''), scene_hash))
            if cur.rowcount > 0:
                updated_count += 1
        
        con.commit()
        con.close()
        print(f"✓ Loaded {updated_count} location tags from {json_path}")
        
    except FileNotFoundError:
        print(f"No location data found at {json_path} - scenes will not have location tags")
    except Exception as e:
        print(f"Error loading locations from JSON: {e}")


def tag_scene_locations(filter_episodes=None, json_path="scene_locations.json"):
    """
    Generate and store location tags for NEW scenes only.
    Saves ONLY to JSON (persistent storage). Database is populated later by load_locations_from_json().
    Checks JSON first to skip already-tagged scenes.
    
    Args:
        filter_episodes: Optional list of episode names to process (e.g., ["1x12 Prophecy Girl"])
                        If None, processes all scenes.
        json_path: Path to JSON file for persistent location storage
    """
    
    # Load existing location data from JSON
    existing_locations = {}
    try:
        with open(json_path, "r") as f:
            existing_locations = json.load(f)
        print(f"Loaded {len(existing_locations)} existing location tags from {json_path}")
    except FileNotFoundError:
        print(f"No existing location data found - will create new {json_path}")
    
    con = get_db_connection()
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    
    failed_scenes = []
    success_count = 0
    skipped_count = 0
    already_tagged_count = 0
    
    try:
        for scene_row in iter_scenes():
            # Skip if filtering and this episode isn't in the list
            if filter_episodes and scene_row["file_name"] not in filter_episodes:
                skipped_count += 1
                continue
            
            scene_hash = scene_row["scene_hash"]
            
            # Check if this scene already has location data in JSON
            if scene_hash in existing_locations:
                already_tagged_count += 1
                continue  # Skip - already tagged, will be loaded to DB later
                
            print(f"\nProcessing scene {scene_row['scene_id']} from {scene_row['file_name']}")
            print(f"  Scene text preview: {scene_row['text'][:100]}...")
            
            try:
                # Generate location tags
                result = tag_text(scene_row["text"], generate_html=False)
                
                extraction_list = []
                location_descr_list = []
                
                for extraction in result:
                    extraction_list.append(extraction.extraction_text)
                    if extraction.attributes and "location_descr" in extraction.attributes:
                        location_descr_list.append(extraction.attributes["location_descr"])
                
                location_text = " | ".join(extraction_list)
                location_descr = " | ".join(location_descr_list)
                
                # Save to JSON (persistent storage) - database updated later
                existing_locations[scene_hash] = {
                    "scene_id": scene_row["scene_id"],
                    "file_name": scene_row["file_name"],
                    "location_text": location_text,
                    "location_descr": location_descr,
                }
                
                # Write JSON after each successful tag to save progress
                with open(json_path, "w") as f:
                    json.dump(existing_locations, f, indent=2)
                
                success_count += 1
                print(f"  ✓ Tagged: {location_descr}")
                
            except Exception as e:
                print(f"  ✗ FAILED: {type(e).__name__}: {str(e)}")
                failed_scenes.append({
                    "scene_id": scene_row["scene_id"],
                    "scene_hash": scene_hash,
                    "file_name": scene_row["file_name"],
                    "error": str(e)
                })
                continue
        
        print(f"\n{'='*60}")
        print(f"Location tagging complete!")
        print(f"  Already tagged (skipped): {already_tagged_count} scenes")
        print(f"  Newly tagged: {success_count} scenes")
        if skipped_count > 0:
            print(f"  Skipped (filtered): {skipped_count} scenes")
        if failed_scenes:
            print(f"  Failed: {len(failed_scenes)} scenes")
            # Save failed scenes to separate JSON for review
            with open("location_tagging_failures.json", "w") as f:
                json.dump(failed_scenes, f, indent=2)
            print(f"  Failed scenes saved to: location_tagging_failures.json")
        print(f"  Total in JSON: {len(existing_locations)} scenes")
        print(f"{'='*60}\n")
                
    except Exception as e:
        print(f"Error in tag_scene_locations: {e}")
    finally:
        con.close()