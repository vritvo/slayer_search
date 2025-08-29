from openai import OpenAI
from dotenv import load_dotenv
import os
import pickle
import numpy as np
from utils import make_embedding

# ep title, line #, embedding []


def chunk_scripts():
    # for file_name in os.listdir("scripts"):

    file_name = "4x12 A New Man.txt"
    episode_embeddings = []

    with open("scripts/" + file_name, "r") as f:
        script = f.read()

    script_list = script.split("\n\n")

    for i, l in enumerate(script_list):
        if l.strip() == "":
            continue

        print(i)

        e = make_embedding(l)
        episode_embeddings.append((file_name, i, l, e))

    with open("embeddings.pkl", "wb") as file:
        pickle.dump(episode_embeddings, file)

    return episode_embeddings


if __name__ == "__main__":
    chunk_scripts()
