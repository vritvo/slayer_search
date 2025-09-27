import pandas as pd
import toml
import os
from utils import semantic_search
import logging

logger = logging.getLogger(__name__)


def evaluate_semantic_search(
    chunk_type: str = "window", embedding_model: str = "openAI", initial_k: int = 10
):
    # Load config
    config = toml.load("config.toml")
    search_queries = config["EVALUATION"]["search_queries"]
    output_path = config["EVALUATION"]["output_path"]

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Collect all results first, then create DataFrame once
    results = []

    for query in search_queries:
        rows = semantic_search(query, chunk_type, embedding_model, initial_k)

        for episode, index, text, dist in rows:
            results.append(
                {
                    "query": query,
                    "episode": episode,
                    "index": index,
                    "text": text,
                    "dist": dist,
                    "chunk_type": chunk_type,
                    "embedding_model": embedding_model,
                    "initial_k": initial_k,
                }
            )

    print(results)
    # Create Dataframe:
    eval_df = pd.DataFrame(results)

    # Save to CSV
    eval_df.to_csv(output_path, index=False)

    return eval_df


if __name__ == "__main__":
    evaluate_semantic_search()
