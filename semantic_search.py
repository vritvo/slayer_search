import numpy as np
from utils import make_embedding, get_db_connection
import argparse
import toml

from sentence_transformers import SentenceTransformer, CrossEncoder


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""

    dot_product = np.dot(vec1, vec2)  # dot product
    norm_vec1 = np.linalg.norm(vec1)  # magnitude of the vector
    norm_vec2 = np.linalg.norm(vec2)  # magnitude of the vector
    return dot_product / (norm_vec1 * norm_vec2)


def simple_search_db(
    search_query: str, chunk_type: str = "window", embedding_model: str = "sbert"
):
    rows = semantic_search(search_query, chunk_type, embedding_model)

    for fname, idx, text, dist in rows:
        print(f"{fname} [{idx}] [distance={dist:.4f}] \n{text[:200]}...)")
        print("-----\n\n")


def semantic_search(
    search_query: str,
    chunk_type: str = "window",
    embedding_model: str = "sbert",
    initial_k=10,
):
    # connect
    con = get_db_connection(embedding_model)
    cur = con.cursor()

    config = toml.load("config.toml")

    # convert to np array and then to bytes (BLOB for sqlite)
    if embedding_model == "sbert":
        model = SentenceTransformer(config["EMBEDDING_MODEL"]["sbert_model"])
        search_vec = np.asarray(model.encode(search_query), dtype=np.float32).tobytes()
    elif embedding_model == "openAI":
        search_vec = np.asarray(
            make_embedding(search_query), dtype=np.float32
        ).tobytes()

    # search using the VSS virtual table and join with main table
    table_name = chunk_type
    vss_table_name = f"{table_name}_vss"

    # Handle different table schemas with standardized column names
    if chunk_type == "window":
        index_col = "window_id_in_scene"
        text_col = "window_text"
    else:  # scene or other chunk types
        index_col = "scene_id_in_episode"
        text_col = "scene_text"

    rows = cur.execute(
        f"""
    SELECT e.file_name, e.{index_col}, e.{text_col}, v.distance
    FROM {vss_table_name} v
    JOIN {table_name} e ON e.rowid = v.rowid
    WHERE vss_search(
        v.embedding,
        vss_search_params(?, ?)
    )
    ORDER BY v.distance
    """,
        (search_vec, initial_k),
    ).fetchall()

    con.close()

    return rows


def cross_encoder(
    search_query: str,
    chunk_type: str = "window",
    embedding_model: str = "sbert",
    initial_k: int = 100,
    final_k: int = 10,
):
    config = toml.load("config.toml")
    initial_candidates = semantic_search(
        search_query, chunk_type, embedding_model, initial_k
    )

    print(f"Retrieved {len(initial_candidates)} initial candidates")

    cross_encoder = CrossEncoder(config["EMBEDDING_MODEL"]["crossencoder_model"])

    query_doc_pairs = []
    candidate_metadata = []

    for fname, idx, text, bi_encoder_dist in initial_candidates:
        query_doc_pairs.append([search_query, text])
        candidate_metadata.append((fname, idx, text, bi_encoder_dist))

    print("Reranking with cross-encoder...")

    # Score all pairs with cross-encoder
    cross_encoder_scores = cross_encoder.predict(query_doc_pairs)

    reranked_results = list(zip(cross_encoder_scores, candidate_metadata))
    reranked_results[0][1][3]  # cross encoder score, meta data, then id of meta data

    # Combine scores with metadata and sort by cross-encoder score
    reranked_results.sort(key=lambda x: x[0], reverse=True)

    # Display results:
    print(f"\nTop {final_k} results after reranking:\n")

    for i, (cross_encoder_score, (fname, idx, text, bi_score)) in enumerate(
        reranked_results[:final_k]
    ):
        print(f"#{i + 1}: {fname} [{idx}]")
        print(
            f"Cross-encoder score: {cross_encoder_score:.4f} | Bi-encoder distance: {bi_score:.4f}"
        )
        print(f"{text[:200]}...")
        print("-----\n")
    #


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic search for episode lines")
    parser.add_argument("--query", "--q", type=str, help="Search query")
    parser.add_argument(
        "--chunk_type",
        "--c",
        type=str,
        help="Type of chunking (line, scene, or window)",
        default="scene",
    )
    parser.add_argument(
        "--embedding_model",
        "--e",
        type=str,
        help="Embedding model to use (sbert or openAI)",
        default="sbert",
    )
    parser.add_argument(
        "--cross_encoder", "--x", default=True, help="Use cross encoder for reranking"
    )

    args = parser.parse_args()

    # make sure chunk type is either line, scene, or window
    if args.chunk_type not in ["scene", "window"]:
        raise ValueError("Invalid chunk_type. Must be 'scene', or 'window'.")

    print(f"Chunk type: {args.chunk_type}")

    if args.query:
        search_query = args.query
    else:
        # Get search query from user input
        user_input = input("Enter your search query: ").strip()

        if user_input == "":
            search_query = "giles talking about how the building maze-like"
            print("Using default search query for debugging")
        else:
            search_query = user_input

    print(f"Searching for: '{search_query}'\n")

    # simple_search_db(search_query)

    if args.cross_encoder and args.embedding_model != "openAI":
        # add a check to make sure embedding model is sbert since cross encoder only works with sbert
        if args.embedding_model != "sbert":
            raise ValueError("Cross encoder only works with sbert embedding model")

        print("Using cross encoder for reranking")
        cross_encoder(
            search_query,
            chunk_type=args.chunk_type,
            embedding_model=args.embedding_model,
            initial_k=100,
            final_k=10,
        )
    else:
        print("Using simple search")
        simple_search_db(
            search_query,
            chunk_type=args.chunk_type,
            embedding_model=args.embedding_model,
        )
