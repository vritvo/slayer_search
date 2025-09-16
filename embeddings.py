import toml
import os
from utils import make_embedding
import argparse
from db import (
    init_scene_tables,
    init_window_tables,
    insert_embedding_batch,
    clear_embeddings_table,
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
    """Process script chunks and create embeddings for semantic search."""

    init_scene_tables("scene")

    # Clear existing data
    clear_embeddings_table("scene")

    episode_data = []

    # Process all files in the scripts directory
    # for file_name in os.listdir("scripts"):
    for file_name in [
        "1x01 Welcome to the Hellmouth.txt",
    ]:
        if not file_name.endswith(".txt"):
            continue

        print(f"Processing file: {file_name}")

        with open(f"scripts/{file_name}", "r") as f:
            script = f.read()

        # Split on lines that start with "cut to" (case insensitive)
        lines = script.split("\n")

        # Initialize variables.
        # all_chunks = a list of complete text chunks, split at each scene.
        # curr_chunk = a list of individual lines for the current scene. Gets reset each time a "cut to" is encountered.

        all_chunks = []  # all scene chunks
        curr_chunk = []  # current scene chunk

        for line in lines:
            # If we've hit a new scene, start a new chunk.
            if line.strip().lower().startswith(
                "cut to"
            ) | line.strip().lower().startswith("(cut to"):
                # If we have accumulated lines, save as a chunk
                if curr_chunk:
                    all_chunks.append("\n".join(curr_chunk))

                # Reset current_scene_chunk to be this first line of the new scene
                curr_chunk = [line]

            # If the new line is not a new scene, we append the line to the current scene list.
            else:
                curr_chunk.append(line)

        # Add the final chunk if it exists
        if curr_chunk:
            all_chunks.append("\n".join(curr_chunk))

        for i, chunk in enumerate(all_chunks):
            if chunk.strip() == "":
                continue

            print(f"  Processing chunk {i}")

            # Create embedding for this chunk
            try:
                embedding = make_embedding(chunk)
            except Exception as e:
                if "maximum context length" in str(e):
                    log_oversized_chunk(file_name, i, len(chunk))
                    print(
                        f"Skipping oversized chunk: {file_name} (chunk {i}) - {len(chunk)} characters"
                    )
                    continue
                else:
                    raise e  # Re-raise other errors

            episode_data.append(
                {
                    "file_name": file_name,
                    "chunk_index": i,
                    "chunk_text": chunk,
                    "embedding": embedding,  # Already a list from make_embedding
                }
            )

    # Insert into database
    if episode_data:
        count = insert_embedding_batch("scene", episode_data)
        print(f"Inserted {count} embeddings into database")


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
            SELECT id, file_name, chunk_index, chunk_text
            FROM scene_embeddings
            ORDER BY file_name, chunk_index
        """)
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            for r in rows:
                yield {
                    "scene_id": r["id"],
                    "file_name": r["file_name"],
                    "scene_index": r["chunk_index"],
                    "text": r["chunk_text"],
                }

            con.commit()

    except Exception as e:
        print(f"Error in iter_scenes: {e}")
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
                "window_index": w_idx,
                "window_text": w_text,
                "file_name": scene["file_name"],
            }


def insert_window_db():
    con = get_db_connection()
    cur = con.cursor()

    table_name = "window_embedding"
    try:
        for row in iter_windows_from_scenes():
            cur.execute(
                f"""
                INSERT INTO {table_name}
                (scene_id, window_index, window_text, file_name)
                VALUES (?, ?, ?, ?)
            """,
                (
                    row["scene_id"],
                    row["window_index"],
                    row["window_text"],
                    row["file_name"],
                ),
            )

        con.commit()

    except Exception as e:
        print(f"Error in insert_window_db: {e}")
    finally:
        con.close()


if __name__ == "__main__":
    make_scene_chunks()

    insert_window_db()
