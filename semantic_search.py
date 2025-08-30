import pickle
import numpy as np
from utils import make_embedding


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def rank_ep_cos_sim(search_query: np.ndarray) -> list[tuple[float, str, str]]:
    """Rank episode lines by cosine similarity to search query embedding."""
    
    with open("embeddings.pkl", "rb") as file:
        embeddings = pickle.load(file)

    cos_script = []
    for embedding in embeddings: 
        # embedding: (episode name, line number, script text, embedding_vector)
        cos_sim = cosine_similarity(search_query, embedding[3])
        cos_script.append((cos_sim, embedding[0], embedding[2]))

    # Sort by similarity score, highest first
    cos_script = sorted(cos_script, key=lambda x: x[0], reverse=True)
    return cos_script


if __name__ == "__main__":
    # Get search query from user input
    user_input = input("Enter your search query: ").strip()
    
    if user_input == "":
        search_query = "giles talking about how the building maze-like"
        print("Using default search query for debugging")
    else:
        search_query = user_input
    
    print(f"Searching for: '{search_query}'\n")
    
    search_embedding = make_embedding(search_query)
    rank_output = rank_ep_cos_sim(search_embedding)

    for r in rank_output[:5]:
        cos_sim, episode, line = r
        print(f"COSINE SIM: {cos_sim:.3f}")
        print(f"EPISODE: {episode}")
        print(f"LINE:\n {line}\n")
