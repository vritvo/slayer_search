import pandas as pd
import toml
import os
from utils import semantic_search, initialize_models
import logging

logger = logging.getLogger(__name__)


def evaluate_semantic_search():
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


def meta_evaluator():
    # Load config
    config = toml.load("config.toml")
    initial_k = config["SEARCH"]["initial_k"]

    eval_path = config["EVALUATION"]["params"]["output_path"]
    eval_df = pd.read_csv(eval_path)
    meta_columns = [
        "evaluation_id",
        "query",
        "bi_encoder_model",
        "chunk_size",
        "overlap",
        "initial_k",
        "final_k",
        "date",
    ]

    # 1) Find the rank for each query in each evaluation (including where the value isn't returned, assuming it's then at initial_k + 1)
    just_query_and_run = eval_df[["evaluation_id", "query"]].drop_duplicates()
    where_match = eval_df[eval_df["correct_match"]]
    where_match = pd.merge(
        just_query_and_run, where_match, how="left", on=["evaluation_id", "query"]
    )[["evaluation_id", "query", "rank"]][["evaluation_id", "query", "rank"]].fillna(
        initial_k + 1
    )

    # 2) Get whether or not the correct answer was returned at all.
    query_and_run_table = (
        eval_df.groupby(meta_columns)["correct_match"].sum().reset_index()
    )

    # 3) Merge:
    final_results = pd.merge(
        where_match, query_and_run_table, on=["evaluation_id", "query"]
    )
    # 4 Merge by evaluation_id:
    meta_columns.remove("query")
    print(meta_columns)
    final_results=final_results.groupby(meta_columns)[["rank", "correct_match"]].mean()

    final_results.to_csv("eval/meta_evaluation.csv")


if __name__ == "__main__":
    evaluate_semantic_search()
    meta_evaluator()
