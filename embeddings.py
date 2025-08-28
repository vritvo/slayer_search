from openai import OpenAI
from dotenv import load_dotenv
import os
import pickle
import numpy as np


# ep title, line #, embedding []


def chunk_scripts():
    # for file_name in os.listdir("scripts"):

    file_name = "test.txt"
    episode_embeddings = []
    with open("scripts/" + file_name, "r") as f:
        script = f.read()

        script_list = script.split("\n")

        for i, l in enumerate(script_list):
            if l.strip() == "":
                continue

            print(i)

            e = make_embeddings(l)
            episode_embeddings.append((file_name, i, l, e))

        with open("embeddings.pkl", "wb") as file:
            pickle.dump(episode_embeddings, file)

    return episode_embeddings


def make_embeddings(script):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(input=script, model="text-embedding-3-small")

    return response.data[0].embedding


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
        cos_script.append((cos_sim, l_embedding[2]))

    cos_script = sorted(cos_script, key=lambda tup: tup[0], reverse=True)
    return cos_script


if __name__ == "__main__":
    # chunk_scripts()

    search_query = "exercise location in flames"
    search_embedding = make_embeddings(search_query)

    rank_output = rank_ep_cos_sim(search_embedding)

    print(rank_output[:5])
