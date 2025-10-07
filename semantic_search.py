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
        "--chunk_type",
        "--c",
        type=str,
        help="Type of chunking (line, scene, or window)",
        default="scene",
    )

    parser.add_argument(
        "--cross_encoder",
        "--x",
        action="store_true",
        help="Use cross encoder for reranking",
    )

    parser.add_argument("--eval", "--e", action="store_true")

    args = parser.parse_args()

    # make sure chunk type is either line, scene, or window
    if args.chunk_type not in ["scene", "window"]:
        raise ValueError("Invalid chunk_type. Must be 'scene', or 'window'.")

    print(f"Chunk type: {args.chunk_type}")
    print(f"eval value is {args.eval}")
    print(f"cross encoder: {args.cross_encoder}")

    if not args.eval:
        # Get search query from user input
        user_input = input("Enter your search query: ").strip()
    else:
        log_eval()
        evaluate_semantic_search()
    if user_input == "":
        search_query = "giles talking about how the building maze-like"
        print("Using default search query for debugging")
    else:
        search_query = user_input

    print(f"Searching for: '{search_query}'\n")

    # simple_search_db(search_query)

    if args.cross_encoder:
        # add a check to make sure embedding model is sbert since cross encoder only works with sbert

        print("Using cross encoder for reranking")
        cross_encoder(
            search_query,
            chunk_type=args.chunk_type,
            initial_k=initial_k,
            final_k=final_k,
        )
    else:
        print("Using simple search")
        simple_search_db(
            search_query,
            chunk_type=args.chunk_type,
            initial_k=initial_k,
        )
