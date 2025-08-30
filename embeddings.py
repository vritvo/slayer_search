import pandas as pd
import numpy as np
import toml
import os
from utils import make_embedding


def chunk_scripts(chunk_type: str = "line") -> pd.DataFrame:
    """Process script chunks and create embeddings for semantic search.

    Args:
        chunk_type: Either "line" (split on double newlines) or "scene" (split on "cut to")
    """
    if chunk_type not in ["line", "scene"]:
        raise ValueError("chunk_type must be either 'line' or 'scene'")

    episode_data = []

    # Process all files in the scripts directory
    for file_name in os.listdir("scripts"):
        # for file_name in ["4x09 Something Blue.txt", "4x12 A New Man.txt"]:
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

        for i, chunk in enumerate(script_chunks):
            if chunk.strip() == "":
                continue

            print(f"  Processing chunk {i}")

            # Create embedding for this chunk
            embedding = make_embedding(chunk)
            episode_data.append(
                {
                    "file_name": file_name,
                    "chunk_index": i,
                    "chunk_text": chunk,
                    "embedding": embedding,  # Already a list from make_embedding
                }
            )

    # Create DataFrame from collected data
    df = pd.DataFrame(episode_data)

    embeddings_folder = toml.load("config.toml")["EMBEDDINGS_FOLDER"]
    # Save embeddings to CSV file with chunk_type in filename
    filename = f"{embeddings_folder}/embeddings_{chunk_type}.csv"
    df.to_csv(filename, index=False)

    print(f"Saved {len(df)} embeddings to {filename}")
    return df


if __name__ == "__main__":
    chunk_scripts(chunk_type="scene")
