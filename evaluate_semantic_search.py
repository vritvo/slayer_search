import pandas as pd
import toml
import os
import argparse
from utils import semantic_search, initialize_models
import logging

logger = logging.getLogger(__name__)


def evaluate_semantic_search(notes=""):
    # Load models once at the start
    initialize_models()

    # Load config
    config = toml.load("config.toml")
    initial_k = config["SEARCH"]["initial_k"]
    final_k = config["SEARCH"]["final_k"]
    search_queries = config["EVALUATION"]["search_queries"]
    output_path = config["EVALUATION"]["params"]["output_path"]
    cross_encoder_model = config["EMBEDDING_MODEL"]["crossencoder_model"]
    bi_encoder_model = config["EMBEDDING_MODEL"]["model_name"]
    chunk_size = config["WINDOW"]["window_size"]
    overlap = config["WINDOW"]["step_size"]

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Collect all results first, then create DataFrame once
    results = pd.DataFrame()

    query_num = 0
    for listed_query, correct_answer in search_queries.items():
        search_query = listed_query
        print(search_query)
        rows = semantic_search(search_query, initial_k)

        rows_df = pd.DataFrame(rows)

        # Add metadata column
        rows_df["query"] = search_query
        rows_df["query_num"] = query_num
        rows_df["rank"] = rows_df.groupby("query").cumcount() + 1
        rows_df["cross_encoder_model"] = cross_encoder_model
        rows_df["bi_encoder_model"] = bi_encoder_model
        rows_df["chunk_type"] = "window"  # Always window now
        rows_df["chunk_size"] = chunk_size
        rows_df["overlap"] = overlap
        rows_df["initial_k"] = initial_k
        rows_df["final_k"] = final_k
        rows_df["notes"] = notes

        # group by evaluation ID & query

        rows_df["correct_match"] = rows_df["text"].str.contains(
            correct_answer, case=False, regex=False
        )

        results = pd.concat([results, rows_df], ignore_index=True)
        results["date"] = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
        results["evaluation_id"] = pd.util.hash_pandas_object(results).sum()
        query_num += 1

    # Check if CSV exists and is non-empty
    header = True
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        header = False

    # Create Dataframe:
    eval_df = pd.DataFrame(results)

    # Save to CSV
    eval_df.to_csv(output_path, mode="a", index=False, header=header)

    #

    return eval_df


def meta_evaluator(notes=""):
    # Load config
    config = toml.load("config.toml")
    final_k = config["SEARCH"]["final_k"]

    eval_path = config["EVALUATION"]["params"]["output_path"]
    eval_df = pd.read_csv(eval_path)

    # Fill NaN values in notes column with empty string
    eval_df["notes"] = eval_df["notes"].fillna("")
    meta_columns = [
        "evaluation_id",
        "query",
        "bi_encoder_model",
        "chunk_size",
        "overlap",
        "initial_k",
        "final_k",
        "date",
        "notes",
    ]

    # 2) Calculate metrics for each query/evaluation combination
    query_metrics = []

    for _, group in eval_df.groupby(["evaluation_id", "query"]):
        eval_id = group["evaluation_id"].iloc[0]
        query = group["query"].iloc[0]

        # Check if correct answer is found in initial_k and final_k
        has_match_initial = group["correct_match"].any()
        has_match_final = group[group["rank"] <= final_k]["correct_match"].any()

        # Get rank if match exists
        rank_if_match = (
            group[group["correct_match"]]["rank"].iloc[0] if has_match_initial else None
        )

        query_metrics.append(
            {
                "evaluation_id": eval_id,
                "query": query,
                "pct_correct_in_initial_k": 1 if has_match_initial else 0,
                "pct_correct_in_final_k": 1 if has_match_final else 0,
                "rank_if_in_initial_k": rank_if_match if has_match_initial else None,
                "rank_if_in_final_k": rank_if_match if has_match_final else None,
            }
        )

    query_metrics_df = pd.DataFrame(query_metrics)

    # 3) Merge with metadata
    meta_data = eval_df[meta_columns].drop_duplicates()
    final_results = pd.merge(query_metrics_df, meta_data, on=["evaluation_id", "query"])

    # 4) Aggregate by evaluation (excluding query)
    meta_columns.remove("query")
    aggregation_funcs = {
        "pct_correct_in_initial_k": "mean",
        "pct_correct_in_final_k": "mean",
        "rank_if_in_initial_k": "mean",  # Only averages non-null values
        "rank_if_in_final_k": "mean",  # Only averages non-null values
    }

    final_results = final_results.groupby(meta_columns).agg(aggregation_funcs)

    # Convert percentages to actual percentages (0-100)
    final_results["pct_correct_in_initial_k"] = (
        final_results["pct_correct_in_initial_k"] * 100
    )
    final_results["pct_correct_in_final_k"] = (
        final_results["pct_correct_in_final_k"] * 100
    )

    final_results.to_csv("eval/meta_evaluation.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate semantic search")
    parser.add_argument(
        "--notes",
        "--n",
        type=str,
        default="",
        help="Optional notes about this specific evaluation run",
    )
    args = parser.parse_args()

    evaluate_semantic_search(notes=args.notes)
    meta_evaluator(notes=args.notes)
