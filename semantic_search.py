from utils import (
    simple_search_db,
    cross_encoder,
)
import logging
import argparse
from evaluate_semantic_search import evaluate_semantic_search

logger = logging.getLogger(__name__)


def log_eval():
    logging.basicConfig(filename="evaluate_queries.log", level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic search for episode lines")
    parser.add_argument(
        "--chunk_type",
        "--c",
        type=str,
        help="Type of chunking (line, scene, or window)",
        default="scene",
    )
    parser.add_argument(
        "--model",
        "--m",
        type=str,
        help="Embedding model to use (sbert or openAI)",
        default="sbert",
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

    if args.cross_encoder and args.model != "openAI":
        # add a check to make sure embedding model is sbert since cross encoder only works with sbert
        if args.model != "sbert":
            raise ValueError("Cross encoder only works with sbert embedding model")

        print("Using cross encoder for reranking")
        cross_encoder(
            search_query,
            chunk_type=args.chunk_type,
            embedding_model=args.model,
            initial_k=100,
            final_k=15,
        )
    else:
        print("Using simple search")
        simple_search_db(
            search_query,
            chunk_type=args.chunk_type,
            embedding_model=args.model,
            initial_k=10,
        )
