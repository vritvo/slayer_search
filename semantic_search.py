import pandas as pd
import numpy as np
from utils import make_embedding
import toml
import argparse

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def rank_ep_cos_sim(
    search_query: np.ndarray, chunk_type: str
) -> pd.DataFrame:
    """Rank episode lines by cosine similarity to search query embedding.
    
    Args:
        search_query: The embedding vector for the search query.
        chunk_type: The type of chunking used for the embeddings (e.g., "line" or "scene").
    """

    if chunk_type not in ["line", "scene"]:
        raise ValueError("Invalid chunk_type. Must be 'line' or 'scene'.")

    embeddings_folder = toml.load("config.toml")["EMBEDDINGS_FOLDER"]
    filename = f"{embeddings_folder}/embeddings_{chunk_type}.csv"
    
    # Load DataFrame from CSV
    df = pd.read_csv(filename)
    
    # Convert embedding string (from csv) back to numpy arrays
    df["embedding_array"] = df.embedding.apply(eval).apply(np.array)
    
    # Calculate cosine similarity for each row
    df['cosine_similarity'] = df['embedding_array'].apply(
        lambda embedding: cosine_similarity(search_query, embedding)
    )
    
    # Sort by cosine similarity, highest first
    df_sorted = df.sort_values('cosine_similarity', ascending=False)
    
    return df_sorted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic search for episode lines")
    parser.add_argument("--query", "--q", type=str, help="Search query")
    parser.add_argument("--chunk_type", "--c", type=str, help="Type of chunking (line or scene)")
    args = parser.parse_args()

    # make sure chunk type is either line or scene
    if args.chunk_type not in ["line", "scene"]:
        raise ValueError("Invalid chunk_type. Must be 'line' or 'scene'.")

    print(f"Chunk type: {args.chunk_type}")


    if args.query:
        search_query = args.query
    else:
        # Get search query from user input
        user_input = input("Enter your search query: ").strip()

        if user_input == "":
            search_query = "giles talking about how the building maze-like"
            print("Using default search query for debugging")
        else:
            search_query = user_input

    print(f"Searching for: '{search_query}'\n")

    search_embedding = make_embedding(search_query)
    result_df = rank_ep_cos_sim(search_embedding, chunk_type="scene")

    # Display top 5 search results
    print("Top 5 search results:")
    print("-" * 50)
    
    for idx in range(min(5, len(result_df))):
        row = result_df.iloc[idx]
        similarity_score = row['cosine_similarity']
        episode_name = row['file_name']
        text_content = row['chunk_text']
        
        print(f"COSINE SIM: {similarity_score:.3f}")
        print(f"EPISODE: {episode_name}")
        print(f"TEXT:\n{text_content}\n")
        print("-" * 30)
