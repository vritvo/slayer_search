import pickle
import numpy as np
from utils import make_embedding


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    return dot_product / (norm_vec1 * norm_vec2)


def rank_ep_cos_sim(search_query):
    with open("embeddings.pkl", "rb") as file:
        embeddings = pickle.load(file)

    cos_script = []
    for l_embedding in embeddings:  # tuple
        cos_sim = cosine_similarity(
            search_query, l_embedding[3]
        )  # Todo make it a dictionary
        cos_script.append((cos_sim, l_embedding[0], l_embedding[2]))

    cos_script = sorted(cos_script, key=lambda tup: tup[0], reverse=True)
    return cos_script


if __name__ == "__main__":
    search_query = "giles talking about how the building maze-like"
    search_embedding = make_embedding(search_query)

    rank_output = rank_ep_cos_sim(search_embedding)

    for r in rank_output[:5]:
        cos_sim = r[0]
        episode = r[1]
        line = r[2]
        print(f"COSINE SIM:\t {cos_sim:.2f}\nEPISODE:\t{episode}\nLINE:\t{line}\n\n")
