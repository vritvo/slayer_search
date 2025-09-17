import pandas as pd
import numpy as np
from utils import make_embedding
import toml
import argparse
import sqlite3
import sqlite_vss


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""

    dot_product = np.dot(vec1, vec2)  # dot product
    norm_vec1 = np.linalg.norm(vec1)  # magnitude of the vector
    norm_vec2 = np.linalg.norm(vec2)  # magnitude of the vector
    return dot_product / (norm_vec1 * norm_vec2)


def search_db(search_query: str, chunk_type: str = "scene"):
    # connect
    con = sqlite3.connect("./vector_db.db")

    # Load sqlite-vss extension
    con.enable_load_extension(True)
    sqlite_vss.load(con)

    cur = con.cursor()

    # convert to np array and then to bytes (BLOB for sqlite)
    search_vec = np.asarray(make_embedding(search_query), dtype=np.float32).tobytes()

    # search using the VSS virtual table and join with main table
    table_name = chunk_type
    vss_table_name = f"{table_name}_vss"

    # TODO: change the column names so they're standardized.
    # Handle different table schemas
    if chunk_type == "window":
        index_col = "window_index"
        text_col = "window_text"
    else:  # scene or other chunk types
        index_col = "chunk_index"
        text_col = "chunk_text"

    rows = cur.execute(
        f"""
    SELECT e.file_name, e.{index_col}, e.{text_col}, v.distance
    FROM {vss_table_name} v
    JOIN {table_name} e ON e.rowid = v.rowid
    WHERE vss_search(
        v.embedding,
        vss_search_params(?, 10)
    )
    ORDER BY v.distance
    """,
        (search_vec,),
    ).fetchall()

    for fname, idx, text, dist in rows:
        print(f"{fname} [{idx}] [distance={dist:.4f}] \n{text[:200]}...)")
        print("-----\n\n")

    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic search for episode lines")
    parser.add_argument("--query", "--q", type=str, help="Search query")

    # TODO: this parser is not necessary anymore, will need to process of how this works.
    parser.add_argument(
        "--chunk_type",
        "--c",
        type=str,
        help="Type of chunking (line, scene, or window)",
        default="scene",
    )
    args = parser.parse_args()

    # make sure chunk type is either line, scene, or window
    if args.chunk_type not in ["line", "scene", "window"]:
        raise ValueError("Invalid chunk_type. Must be 'line', 'scene', or 'window'.")

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

    search_db(search_query)
