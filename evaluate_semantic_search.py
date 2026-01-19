import pandas as pd
import toml
import os
import argparse
import glob
from utils.search import semantic_search
from utils.models import initialize_models
from utils.database import get_db_path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def coordinate_evaluation(notes=""):
    search_queries_config = toml.load("eval/evaluation_queries.toml")
    models = search_queries_config["EVALUATION"]["embedding_models"]
    print(models[0])

    # Get the current time, format as string
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    for m in models:
        evaluate_semantic_search(
            bi_encoder_model=m["model_name"],
            include_meta=m["include_meta"],
            db_tag=m.get("db_tag", ""),
            time_stamp=now_str,
            notes=notes,
        )
    
    # Run meta-evaluation on the file we just created
    eval_path = f"eval/evaluation_{now_str}.csv"
    meta_evaluator(eval_path=eval_path, notes=notes)

    return


def evaluate_semantic_search(
    bi_encoder_model: str, include_meta: bool, db_tag: str, time_stamp: str, notes: str = ""
):
    """
    Runs an evaluation - a set of queries for a particular model.

    Args:
        bi_encoder_model (str): The model name (e.g., "all-MiniLM-L6-v2" or "jxm/cde-small-v2")
        include_meta (bool): Whether this database includes injected metadata in the embedding (e.g. location)
        db_tag (str): Optional experiment tag that is added to the db name)
        time_stamp (str): The time this evaluation is run.
        notes (str, optional): Any notes to add for this evaluation. Defaults to "".

    Returns:
        pandas.DataFrame: The dataframe that is also output to the .csv file.
    """
    # Compute the database path for this model configuration
    db_path = get_db_path(bi_encoder_model, include_meta, db_tag)
    print(f"\n{'='*60}")
    print(f"Evaluating: {bi_encoder_model}")
    print(f"Database: {db_path}")
    print(f"Include meta: {include_meta}")
    print(f"{'='*60}\n")

    # Load the specific model for this evaluation
    # (Pass parameters to override config.toml defaults)
    initialize_models(model_name=bi_encoder_model, include_meta=include_meta)

    # Load config
    config = toml.load("config.toml")
    initial_k = config["SEARCH"]["initial_k"]
    final_k = config["SEARCH"]["final_k"]
    chunk_size = config["WINDOW"]["window_size"]
    overlap = config["WINDOW"]["step_size"]

    # Get search queries
    search_queries_config = toml.load("eval/evaluation_queries.toml")
    search_queries = search_queries_config["EVALUATION"]["search_queries"]

    # Collect all results first, then create DataFrame once
    results = pd.DataFrame()

    output_path = f"eval/evaluation_{time_stamp}.csv"
    print(output_path)
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    query_num = 0
    for query_item in search_queries:
        for location_type in [True, False]:
            query_text = query_item["query_text"]

            if location_type:
                location = query_item["location"] + ", "
            else:
                location = ""

            search_query = f"{location}{query_text}"
            print(search_query)

            correct_answer = query_item["script_text"]
            rows = semantic_search(
                search_query, initial_k, initial_k_buffer=2, 
                model_name=bi_encoder_model, db_path=db_path
            )

            rows_df = pd.DataFrame(rows)

            # Add metadata column
            rows_df["query"] = search_query
            rows_df["location_type"] = location_type
            rows_df["query_num"] = query_num
            rows_df["rank"] = rows_df.groupby("query").cumcount() + 1
            rows_df["bi_encoder_model"] = bi_encoder_model
            rows_df["db_tag"] = db_tag
            rows_df["meta_data_included"] = include_meta
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
    eval_df.sort_values(
        by=["evaluation_id", "bi_encoder_model", "query"]
    )

    # Save to CSV
    eval_df.to_csv(output_path, mode="a", index=False, header=header)

    #

    return eval_df


def calculate_evaluation_metrics(eval_df, initial_k, final_k):
    """
    Calculate evaluation metrics from evaluation DataFrame.
    
    This function processes the raw evaluation results and calculates aggregated metrics
    per model configuration.
    
    Args:
        eval_df (pd.DataFrame): Raw evaluation results DataFrame
        initial_k (int): Value of initial_k used in evaluation
        final_k (int): Value of final_k used in evaluation
    
    Returns:
        tuple: (aggregated_df, query_metrics_df) where:
            - aggregated_df: Aggregated metrics per model configuration
            - query_metrics_df: Per-query metrics before aggregation (for detailed analysis)
    """
    # Fill NaN values
    eval_df = eval_df.copy()
    eval_df["notes"] = eval_df["notes"].fillna("")
    eval_df["db_tag"] = eval_df["db_tag"].fillna("")
    
    meta_columns = [
        "evaluation_id",
        "query",
        "location_type",
        "bi_encoder_model",
        "db_tag",
        "meta_data_included",
        "chunk_size",
        "overlap",
        "initial_k",
        "final_k",
        "date",
        "notes",
    ]

    # Calculate metrics for each query/evaluation/model combination
    query_metrics = []

    for _, group in eval_df.groupby(["evaluation_id", "query", "location_type", "bi_encoder_model", "db_tag", "meta_data_included"]):
        eval_id = group["evaluation_id"].iloc[0]
        query = group["query"].iloc[0]
        location_type = group["location_type"].iloc[0]
        bi_encoder_model = group["bi_encoder_model"].iloc[0]
        db_tag = group["db_tag"].iloc[0]
        meta_data_included = group["meta_data_included"].iloc[0]

        # Check if correct answer is found in initial_k and final_k
        has_match_initial = group["correct_match"].any()
        has_match_final = group[group["rank"] <= final_k]["correct_match"].any()

        # Get rank if match exists, otherwise use initial_k + 1 for overall rank
        if has_match_initial:
            rank_if_match = group[group["correct_match"]]["rank"].iloc[0]
            rank_overall = rank_if_match
        else:
            rank_if_match = None
            rank_overall = initial_k + 1

        query_metrics.append(
            {
                "evaluation_id": eval_id,
                "query": query,
                "location_type": location_type,
                "bi_encoder_model": bi_encoder_model,
                "db_tag": db_tag,
                "meta_data_included": meta_data_included,
                "pct_correct_in_initial_k": 1 if has_match_initial else 0,
                "pct_correct_in_final_k": 1 if has_match_final else 0,
                "avg_rank_overall": rank_overall,  # Includes all queries, with initial_k + 1 for non-matches
                "rank_if_in_initial_k": rank_if_match if has_match_initial else None,
                "rank_if_in_final_k": rank_if_match if has_match_final else None,
            }
        )

    query_metrics_df = pd.DataFrame(query_metrics)

    # Merge with metadata
    meta_data = eval_df[meta_columns].drop_duplicates()
    merged_results = pd.merge(
        query_metrics_df, meta_data, on=["evaluation_id", "query", "location_type", "bi_encoder_model", "db_tag", "meta_data_included"]
    )

    # Save query-level results (before aggregation) for detailed analysis
    # Convert percentages to actual percentages (0-100) for query-level data
    query_metrics_with_meta = merged_results.copy()
    query_metrics_with_meta["pct_correct_in_initial_k"] = (
        query_metrics_with_meta["pct_correct_in_initial_k"] * 100
    )
    query_metrics_with_meta["pct_correct_in_final_k"] = (
        query_metrics_with_meta["pct_correct_in_final_k"] * 100
    )
    
    # Aggregate by evaluation (excluding query)
    meta_columns_for_agg = meta_columns.copy()
    meta_columns_for_agg.remove("query")
    aggregation_funcs = {
        "pct_correct_in_initial_k": "mean",
        "pct_correct_in_final_k": "mean",
        "avg_rank_overall": "mean",  # Includes all queries, with initial_k + 1 for non-matches
        "rank_if_in_initial_k": "mean",  # Only averages non-null values
        "rank_if_in_final_k": "mean",  # Only averages non-null values
    }

    aggregated = merged_results.groupby(meta_columns_for_agg).agg(aggregation_funcs)

    # Convert percentages to actual percentages (0-100) for aggregated data
    aggregated["pct_correct_in_initial_k"] = (
        aggregated["pct_correct_in_initial_k"] * 100
    )
    aggregated["pct_correct_in_final_k"] = (
        aggregated["pct_correct_in_final_k"] * 100
    )

    return aggregated.reset_index(), query_metrics_with_meta


def meta_evaluator(eval_path=None, notes=""):
    """
    Run meta-evaluation on evaluation results and save to CSV.
    
    Args:
        eval_path (str, optional): Path to evaluation CSV file. If None, uses most recent 
                                   evaluation_{timestamp}.csv file in eval/ directory.
        notes (str, optional): Notes for this meta-evaluation run.
    """
    # Load config
    config = toml.load("config.toml")
    initial_k = config["SEARCH"]["initial_k"]
    final_k = config["SEARCH"]["final_k"]

    # Auto-detect most recent evaluation file if not provided
    if eval_path is None:
        eval_files = glob.glob("eval/evaluation_*.csv")
        if not eval_files:
            raise FileNotFoundError(
                "No evaluation files found in eval/ directory. "
                "Run an evaluation first or specify eval_path."
            )
        # Sort by modification time, most recent first
        eval_path = max(eval_files, key=os.path.getmtime)
        print(f"Using most recent evaluation file: {eval_path}")
    
    eval_df = pd.read_csv(eval_path)

    # Print models found in the CSV
    unique_models = eval_df[["bi_encoder_model", "db_tag", "meta_data_included"]].drop_duplicates()
    print(f"\nFound {len(unique_models)} model configurations in CSV:")
    for _, row in unique_models.iterrows():
        print(f"  - {row['bi_encoder_model']} (db_tag: '{row['db_tag']}', meta: {row['meta_data_included']})")
    print()
    
    # Calculate metrics using shared function
    aggregated, _ = calculate_evaluation_metrics(eval_df, initial_k, final_k)
    
    # Save to CSV
    aggregated.to_csv("eval/meta_evaluation.csv")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Evaluate semantic search")
    # parser.add_argument(
    #     "--notes",
    #     "--n",
    #     type=str,
    #     default="",
    #     help="Optional notes about this specific evaluation run",
    # )
    # parser.add_argument(
    #     "--meta-only",
    #     action="store_true",
    #     default=False,
    #     help="Only run meta evaluator, skip new evaluation",
    # )
    # args = parser.parse_args()

    # if not args.meta_only:
    #     evaluate_semantic_search(notes=args.notes)
    # meta_evaluator(notes=args.notes)

    coordinate_evaluation()
