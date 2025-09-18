import toml
import os
from utils import make_embedding
import argparse
from db import (
    init_scene_tables,
    init_window_tables,
    insert_into_vss_table,
    clear_table,
)
import sqlite3
from db import get_db_connection



def log_oversized_chunk(file_name, chunk_index, chunk_length):
    """Log an oversized chunk that couldn't be embedded."""
    config = toml.load("config.toml")
    logs_folder = config["LOGS_FOLDER"]

    # Create logs folder if it doesn't exist
    os.makedirs(logs_folder, exist_ok=True)

    log_file = os.path.join(logs_folder, "oversized_chunks.log")

    # Simple append to log file
    with open(log_file, "a") as f:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f"{timestamp} - Oversized chunk skipped - File: {file_name}, Chunk: {chunk_index}, Length: {chunk_length} chars\n"
        )


def make_scene_chunks():
    """Process script chunks and insert them row-by-row into the database."""

    init_scene_tables("scene")

    # Clear existing data
    clear_table("scene")

    con = get_db_connection()
    cur = con.cursor()

    table_name = "scene"

    try:
        # Process all files in the scripts directory
        # for file_name in os.listdir("scripts"):
        for file_name in [
            "4x12 A New Man.txt",
        ]:
            if not file_name.endswith(".txt"):
                continue

            print(f"Splitting file: {file_name}")

            with open(f"scripts/{file_name}", "r") as f:
                script = f.read()

            # Split on lines that start with "cut to" (case insensitive)
            lines = script.split("\n")

            # Initialize variables.
            # curr_chunk = a list of individual lines for the current scene. Gets reset each time a "cut to" is encountered.
            curr_chunk = []  # current scene chunk
            chunk_index = 0

            for line in lines:
                # If we've hit a new scene, start a new chunk.
                if line.strip().lower().startswith(
                    "cut to"
                ) | line.strip().lower().startswith("(cut to"):
                    # If we have accumulated lines, save as a chunk and insert into DB
                    if curr_chunk:
                        chunk_text = "\n".join(curr_chunk).strip()
                        if chunk_text:
                            print(f"  Splitting chunk {chunk_index}")

                            # Insert into main table
                            cur.execute(
                                f"""
                                INSERT INTO {table_name}
                                (file_name, scene_id_in_episode, scene_text)
                                VALUES (?, ?, ?)
                            """,
                                (
                                    file_name,
                                    chunk_index,
                                    chunk_text,
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

                    # Insert into main table
                    cur.execute(
                        f"""
                        INSERT INTO {table_name}
                        (file_name, scene_id_in_episode, scene_text)
                        VALUES (?, ?, ?)
                    """,
                        (
                            file_name,
                            chunk_index,
                            chunk_text,
                        ),
                    )

        con.commit()
        print(f"Successfully inserted scene chunks into table: `{table_name}`")

    except Exception as e:
        print(f"Error in make_scene_chunks: {e}")
    finally:
        con.close()


def make_embeddings(chunk_type: str = "scene"):
    """Create embeddings for the specified chunk type and insert into DB."""

    clear_table(f"{chunk_type}_vss")

    if chunk_type == "scene":
        # Collect all scene data first to avoid connection conflicts
        iter_chunk = list(iter_scenes())
        index_field = "scene_id_in_episode"
        id_field = "scene_id"

    elif chunk_type == "window":
        # Collect all window data first to avoid connection conflicts
        iter_chunk = list(iter_windows())
        index_field = "window_id_in_scene"
        id_field = "window_id"
    else:
        raise ValueError("Invalid chunk_type. Must be 'scene' or 'window'.")

    # Process the collected data
    for db_chunk_row in iter_chunk:
        chunk = db_chunk_row["text"]

        # print episode, chunk info, and chunk id being processed.
        if chunk_type == "scene":
            print(
                f"Processing {db_chunk_row['file_name']} scene {db_chunk_row[index_field]} (ID {db_chunk_row[id_field]})"
            )
        else:  # window
            print(
                f"Processing {db_chunk_row['file_name']} window {db_chunk_row[index_field]} from scene {db_chunk_row['scene_id']} (ID {db_chunk_row[id_field]})"
            )

        try:
            embedding = make_embedding(chunk)
        except Exception as e:
            if "maximum context length" in str(e):
                log_oversized_chunk(
                    db_chunk_row["file_name"], db_chunk_row[index_field], len(chunk)
                )
                print(
                    f"Skipping oversized chunk: {db_chunk_row['file_name']} ({chunk_type} {db_chunk_row[index_field]}) - {len(chunk)} characters"
                )
                continue
            else:
                raise e  # Re-raise other errors

        # insert into db
        insert_into_vss_table(chunk_type, db_chunk_row[id_field], embedding)


def make_window_chunk(chunk):
    init_window_tables()

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
    i = 0

    while i < len(formatted_lines):
        # Create window of formatted lines
        window_end = min(i + window_size, len(formatted_lines))
        window_formatted_lines = formatted_lines[i:window_end]

        # Join the formatted lines back with double newlines
        chunk_text = "\n\n".join(window_formatted_lines).strip()
        if chunk_text:
            script_chunks.append(chunk_text)

        # Move window forward by step_size
        i += step_size

        # Break if we've reached the end
        if window_end >= len(formatted_lines):
            break

    return script_chunks


def iter_scenes(batch_size: int = 500):
    """Yield scene rows from the DB in batches as dicts."""
    con = get_db_connection()
    try:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("""
            SELECT scene_id, file_name, scene_id_in_episode, scene_text
            FROM scene
            ORDER BY file_name, scene_id_in_episode
        """)
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            for r in rows:
                yield {
                    "scene_id": r["scene_id"],
                    "file_name": r["file_name"],
                    "scene_id_in_episode": r["scene_id_in_episode"],
                    "text": r["scene_text"],
                }

            con.commit()

    except Exception as e:
        print(f"Error in iter_scenes: {e}")
    finally:
        con.close()


def iter_windows(batch_size: int = 500):
    """Yield window rows from the DB in batches as dicts."""
    con = get_db_connection()
    try:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("""
            SELECT window_id, scene_id, window_id_in_scene, window_text, file_name
            FROM window
            ORDER BY file_name, scene_id, window_id_in_scene
        """)
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            for r in rows:
                yield {
                    "window_id": r["window_id"],
                    "scene_id": r["scene_id"],
                    "window_id_in_scene": r["window_id_in_scene"],
                    "text": r["window_text"],
                    "file_name": r["file_name"],
                }

            con.commit()

    except Exception as e:
        print(f"Error in iter_windows: {e}")
    finally:
        con.close()


def iter_windows_from_scenes():
    """Example: read each scene, make windows, do something with them."""
    for scene in iter_scenes():
        windows = make_window_chunk(scene["text"])  # your function
        # do whatever you want with windows (insert to DB, write CSV, etc.)
        for w_idx, w_text in enumerate(windows):
            yield {
                "scene_id": scene["scene_id"],
                "window_id_in_scene": w_idx,
                "window_text": w_text,
                "file_name": scene["file_name"],
            }


def insert_window_db():
    con = get_db_connection()
    cur = con.cursor()

    table_name = "window"
    try:
        for row in iter_windows_from_scenes():
            cur.execute(
                f"""
                INSERT INTO {table_name}
                (scene_id, window_id_in_scene, window_text, file_name)
                VALUES (?, ?, ?, ?)
            """,
                (
                    row["scene_id"],
                    row["window_id_in_scene"],
                    row["window_text"],
                    row["file_name"],
                ),
            )

        con.commit()
        print(f"Successfully inserted window chunks into table: `{table_name}`")

    except Exception as e:
        print(f"Error in insert_window_db: {e}")
    finally:
        con.close()


if __name__ == "__main__":
    make_scene_chunks()  # Create scene chunks in database
    insert_window_db()  # Create window chunks in database
    make_embeddings("scene")  # Create embeddings for scene chunks
    make_embeddings("window")  # Create embeddings for window chunks


