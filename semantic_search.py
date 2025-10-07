from utils import (
    simple_search_db,
    cross_encoder,
)
import logging
import argparse
from evaluate_semantic_search import evaluate_semantic_search
import toml

logger = logging.getLogger(__name__)


def log_eval():
    logging.basicConfig(filename="evaluate_queries.log", level=logging.INFO)


if __name__ == "__main__":
    config = toml.load("config.toml")
    initial_k = config["SEARCH"]["initial_k"]
    final_k = config["SEARCH"]["final_k"]

    parser = argparse.ArgumentParser(description="Semantic search for episode lines")
    
    parser.add_argument(
        "--cross_encoder",
        "--x",
        action="store_true",
        help="Use cross encoder for reranking",
    )

    parser.add_argument("--eval", "--e", action="store_true")

    args = parser.parse_args()

    print(f"cross encoder: {args.cross_encoder}")

    if not args.eval:
        # Get search query from user input
        user_input = input("Enter your search query: ").strip()
        
        if user_input == "":
            search_query = "giles talking about how the building maze-like"
            print("Using default search query for debugging")
        else:
            search_query = user_input

        print(f"Searching for: '{search_query}'\n")

        if args.cross_encoder:
            print("Using cross encoder for reranking")
            cross_encoder(
                search_query,
                initial_k=initial_k,
                final_k=final_k,
            )
        else:
            print("Using simple search")
            simple_search_db(
                search_query,
                initial_k=initial_k,
            )
    else:
        log_eval()
        evaluate_semantic_search()
