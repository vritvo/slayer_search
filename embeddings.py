import pandas as pd
import toml
import os
from utils import make_embedding
import argparse
from db import (
    init_embeddings_table,
    insert_embedding_batch,
    clear_embeddings_table,
)


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


def chunk_scripts(chunk_type: str = "line", output_type: str = "csv") -> pd.DataFrame:
    """Process script chunks and create embeddings for semantic search.

    Args:
        chunk_type: Either "line" (split on double newlines), "scene" (split on "cut to"),
                   or "window" (sliding window of lines)
        output_type: Either "csv" or "db" for output format
    """
    if chunk_type not in ["line", "scene", "window"]:
        raise ValueError("chunk_type must be either 'line', 'scene', or 'window'")

    if output_type not in ["csv", "db"]:
        raise ValueError("output_type must be either 'csv' or 'db'")

    # Initialize database table if using db output
    if output_type == "db":
        init_embeddings_table(chunk_type)
        clear_embeddings_table(chunk_type)  # Clear existing data

    episode_data = []

    # Process all files in the scripts directory
    # for file_name in os.listdir("scripts"):
    for file_name in [
        "1x01 Welcome to the Hellmouth.txt",
        "4x12 A New Man.txt",
    ]:
        if not file_name.endswith(".txt"):
            continue

        print(f"Processing file: {file_name}")

        with open(f"scripts/{file_name}", "r") as f:
            script = f.read()

        # Split script into chunks based on chunk_type
        if chunk_type == "line":
            # Split script into chunks by double newlines. This keeps the speaker and dialogue together.
            script_chunks = script.split("\n\n")
        elif chunk_type == "scene":
            # Split on lines that start with "cut to" (case insensitive)
            lines = script.split("\n")
            script_chunks = []
            current_chunk = []

            for line in lines:
                if line.strip().lower().startswith(
                    "cut to"
                ) | line.strip().lower().startswith("(cut to"):
                    # If we have accumulated lines, save as a chunk
                    if current_chunk:
                        script_chunks.append("\n".join(current_chunk))
                    # Start new chunk with the "cut to" line
                    current_chunk = [line]
                else:
                    current_chunk.append(line)

            # Add the final chunk if it exists
            if current_chunk:
                script_chunks.append("\n".join(current_chunk))
        elif chunk_type == "window":
            # Load window configuration
            config = toml.load("config.toml")
            window_size = config["WINDOW"]["window_size"]
            step_size = config["WINDOW"]["step_size"]

            # Split script into lines and create sliding windows
            lines = script.split("\n")
            script_chunks = []

            i = 0
            while i < len(lines):
                # Create window of lines
                window_end = min(i + window_size, len(lines))
                window_lines = lines[i:window_end]

                # Only add non-empty chunks
                chunk_text = "\n".join(window_lines).strip()
                if chunk_text:
                    script_chunks.append(chunk_text)

                # Move window forward, but skip extra if current line is empty
                if i < len(lines) and lines[i].strip() == "":
                    i += step_size + 1  # Skip one extra line if empty
                else:
                    i += step_size

                # Break if we've reached the end
                if window_end >= len(lines):
                    break

        for i, chunk in enumerate(script_chunks):
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

    # Output based on type
    if output_type == "csv":
        # Create DataFrame from collected data
        df = pd.DataFrame(episode_data)

        embeddings_folder = toml.load("config.toml")["EMBEDDINGS_FOLDER"]
        # Save embeddings to CSV file with chunk_type in filename
        filename = f"{embeddings_folder}/embeddings_{chunk_type}.csv"
        df.to_csv(filename, index=False)

        print(f"Saved {len(df)} embeddings to {filename}")

    elif output_type == "db":
        # Insert into database
        if episode_data:
            count = insert_embedding_batch(chunk_type, episode_data)
            print(f"Inserted {count} embeddings into database")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk_type", "--c", choices=["line", "scene", "window"], default="scene"
    )
    parser.add_argument(
        "--output_type",
        "--o",
        choices=["csv", "db"],
        default="db",
        help="Output format: csv file or database",
    )
    args = parser.parse_args()

    print(f"Chunk type: {args.chunk_type}, Output type: {args.output_type}")
    chunk_scripts(chunk_type=args.chunk_type, output_type=args.output_type)
