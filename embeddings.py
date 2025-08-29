from openai import OpenAI
from dotenv import load_dotenv
import os
import pickle
import numpy as np
from utils import make_embedding


def chunk_scripts() -> list[tuple[str, int, str, np.ndarray]]:
    """Process script chunks and create embeddings for semantic search."""
    # TODO: Process all files in scripts directory
    file_name = "4x12 A New Man.txt"
    episode_embeddings = []

    with open(f"scripts/{file_name}", "r") as f:
        script = f.read()

    # Split script into chunks by double newlines. This keeps the speaker and dialogue together.
    script_chunks = script.split("\n\n")

    for i, chunk in enumerate(script_chunks):
        if chunk.strip() == "":
            continue

        print(f"Processing chunk {i}")
        
        # Create embedding for this chunk
        embedding = make_embedding(chunk)
        episode_embeddings.append((file_name, i, chunk, embedding))

    # Save embeddings to pickle file
    with open("embeddings.pkl", "wb") as file:
        pickle.dump(episode_embeddings, file)

    return episode_embeddings


if __name__ == "__main__":
    chunk_scripts()
