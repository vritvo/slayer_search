import pandas as pd
import toml
import os
from utils import cross_encoder
import logging

logger = logging.getLogger(__name__)


def evaluate_semantic_search(
    chunk_type: str = "window",
    embedding_model: str = "sbert",
    initial_k: int = 100,
    final_k=10,
):
    # Load config
    config = toml.load("config.toml")
    search_queries = config["EVALUATION"]["search_queries"]
    output_path = config["EVALUATION"]["params"]["output_path"]
    cross_encoder_model = config["EMBEDDING_MODEL"]["crossencoder_model"]
    bi_encoder_model = config["EMBEDDING_MODEL"]["sbert_model"]
    chunk_size = config["WINDOW"]["window_size"]
    overlap = config["WINDOW"]["step_size"]
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Collect all results first, then create DataFrame once
    results = pd.DataFrame()

    for key, value in search_queries.items():
        search_query = key
        print(search_query)
        rows_df = cross_encoder(
            search_query, chunk_type, embedding_model, initial_k, final_k
        )
        # Add metadata columns
        rows_df["cross_encoder_model"] = cross_encoder_model
        rows_df["bi_encoder_model"] = bi_encoder_model
        rows_df["chunk_type"] = chunk_type
        rows_df["correct_match"] = rows_df["text"].str.contains(
            value, case=False, regex=False
        )
        rows_df["embedding_model"] = embedding_model
        rows_df["chunk_size"] = chunk_size
        rows_df["overlap"] = overlap
        rows_df["initial_k"] = initial_k
        rows_df["final_k"] = final_k

        # Append to results DataFrame
        results = pd.concat([results, rows_df], ignore_index=True)
        results["date"] = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
        results["evaluation_id"] = pd.util.hash_pandas_object(results).sum()

    # Check if CSV exists and is non-empty
    header = True
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        header = False

    # Create Dataframe:
    eval_df = pd.DataFrame(results)

    # Save to CSV
    eval_df.to_csv(output_path, mode="a", index=False, header=header)

    # Create Dataframe:
    eval_df = pd.DataFrame(results)

    # Save to CSV
    eval_df.to_csv(output_path, mode="a", index=False)

    return eval_df


if __name__ == "__main__":
    evaluate_semantic_search()
