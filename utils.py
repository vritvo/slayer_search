from dotenv import load_dotenv
import os
import sqlite3
import toml
import numpy as np
import sqlite_vss
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder  # sbert
import pandas as pd


def get_db_connection():
    """Get database connection with the correct path for the embedding model."""
    config = toml.load("config.toml")
    db_path = config["db_path"]

    con = sqlite3.connect(db_path, timeout=20.0)  # Add timeout

    # Load the sqlite-vss extension
    con.enable_load_extension(True)
    try:
        sqlite_vss.load(con)
    except Exception as e:
        print(f"Warning: Could not load sqlite-vss extension: {e}")
    finally:
        con.enable_load_extension(False)

    # Set some pragmas for better concurrency
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")

    return con


def log_oversized_chunk(file_name, chunk_index, chunk_length):
    """Log an oversized chunk that couldn't be embedded."""
    config = toml.load("config.toml")
    logs_folder = config["LOGS_FOLDER"]

    # Create logs folder if it doesn't exist
    os.makedirs(logs_folder, exist_ok=True)

    log_file = os.path.join(logs_folder, "oversized_chunks.log")

    # Simple append to log file
    with open(log_file, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f"{timestamp} - Oversized chunk skipped - File: {file_name}, Chunk: {chunk_index}, Length: {chunk_length} chars\n"
        )


def init_scene_tables():
    """Initialize the scene table (needed for data processing)."""
    con = get_db_connection()
    cur = con.cursor()

    # Create the scene table
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS scene (
            scene_id INTEGER PRIMARY KEY AUTOINCREMENT,
            scene_id_in_episode INTEGER,
            scene_text TEXT,
            file_name TEXT
        )
    """)
    con.commit()
    con.close()


def init_window_tables():
    """Initialize the window table and VSS virtual table."""
    config = toml.load("config.toml")

    con = get_db_connection()
    cur = con.cursor()

    embedding_dim = config["EMBEDDING_MODEL"]["model_dim"]

    # Create the window table
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS window (
            window_id INTEGER PRIMARY KEY AUTOINCREMENT,
            scene_id INTEGER,
            window_id_in_scene INTEGER,
            window_text TEXT,
            file_name TEXT
        )
    """)

    # Create virtual table for vector search using sqlite-vss
    cur.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS window_vss USING vss0(
            embedding({embedding_dim})
        )
    """)
    con.commit()
    con.close()


def iter_scenes(batch_size: int = 500):
    """Yield scene rows from the DB in batches as dicts. Used for data processing."""
    con = get_db_connection()
    try:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("""
            SELECT scene_id, scene_id_in_episode, scene_text, file_name
            FROM scene
            ORDER BY file_name, scene_id_in_episode
        """)
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            for r in rows:
                yield {
                    "scene_id": r["scene_id"],
                    "scene_id_in_episode": r["scene_id_in_episode"],
                    "text": r["scene_text"],
                    "file_name": r["file_name"],
                }

            con.commit()

    except Exception as e:
        print(f"Error in iter_scenes: {e}")
    finally:
        con.close()


def insert_into_vss_table(row_id: int, embedding):
    """Insert a single embedding into the VSS virtual table."""
    con = get_db_connection()
    cur = con.cursor()

    cur.execute(
        f"""
        INSERT INTO window_vss(rowid, embedding)
        VALUES (?, ?)
    """,
        (row_id, json.dumps(embedding)),
    )

    con.commit()
    con.close()


def batch_insert_into_vss_table(embeddings_data):
    """Insert multiple embeddings into the VSS virtual table using batch processing."""
    con = get_db_connection()
    cur = con.cursor()

    batch_size = 500  # Smaller batch size for embeddings due to memory usage

    try:
        for i in range(0, len(embeddings_data), batch_size):
            batch = embeddings_data[i : i + batch_size]

            # Prepare batch data - convert embeddings to JSON strings
            batch_values = [
                (row_id, json.dumps(embedding)) for row_id, embedding in batch
            ]

            cur.executemany(
                f"""
                INSERT INTO window_vss(rowid, embedding)
                VALUES (?, ?)
            """,
                batch_values,
            )

            con.commit()  # Commit each batch
            print(
                f"Inserted batch {i // batch_size + 1}: {len(batch)} embeddings into window_vss"
            )

    except Exception as e:
        print(f"Error in batch_insert_into_vss_table: {e}")
        con.rollback()
        raise
    finally:
        con.close()


def clear_table(table_name):
    """Clear all data from a table."""
    con = get_db_connection()
    cur = con.cursor()

    cur.execute(f"DELETE FROM {table_name}")

    con.commit()
    con.close()


def make_embeddings():
    """Create embeddings for window chunks and insert into DB."""
    clear_table("window_vss")

    # Collect all window data first to avoid connection conflicts
    iter_chunk = list(iter_windows())

    config = toml.load("config.toml")
    model_name = config["EMBEDDING_MODEL"]["model_name"]
    model = SentenceTransformer(model_name)

    all_chunks = []
    all_ids = []
    for db_chunk_row in iter_chunk:
        all_chunks.append(
            f"episode: {db_chunk_row['file_name']}:\n{db_chunk_row['text']}"
        )
        all_ids.append(db_chunk_row["window_id"])

    print(f"Creating embeddings for {len(all_chunks)} window chunks...")
    all_embeddings = model.encode(all_chunks)
    # TODO: encode vs encode_document https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html

    # Prepare embeddings data for batch insertion
    embeddings_data = []
    for chunk_id, embedding in zip(all_ids, all_embeddings):
        embeddings_data.append((chunk_id, embedding.tolist()))

    # Actually insert the embeddings into the database
    print(f"Batch inserting {len(embeddings_data)} window embeddings...")
    batch_insert_into_vss_table(embeddings_data)


def iter_windows(batch_size: int = 500):
    """Yield window rows from the DB in batches as dicts."""
    con = get_db_connection()
    try:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("""
            SELECT window_id, scene_id, window_id_in_scene, window_text, file_name
            FROM window
            ORDER BY file_name, scene_id, window_id_in_scene
        """)
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            for r in rows:
                yield {
                    "window_id": r["window_id"],
                    "scene_id": r["scene_id"],
                    "window_id_in_scene": r["window_id_in_scene"],
                    "text": r["window_text"],
                    "file_name": r["file_name"],
                }

            con.commit()

    except Exception as e:
        print(f"Error in iter_windows: {e}")
    finally:
        con.close()


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)  # dot product
    norm_vec1 = np.linalg.norm(vec1)  # magnitude of the vector
    norm_vec2 = np.linalg.norm(vec2)  # magnitude of the vector
    return dot_product / (norm_vec1 * norm_vec2)


def simple_search_db(search_query: str, initial_k=10):
    rows = semantic_search(search_query, initial_k=initial_k)

    for result in rows:
        print(f"{result['episode']} [{result['scene_id']}] [distance={result['distance']:.4f}] \n{result['text'][:200]}...)")
        print("-----\n\n")


def semantic_search(search_query: str, initial_k=10):
    # connect
    con = get_db_connection()
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    config = toml.load("config.toml")

    # convert to np array and then to bytes (BLOB for sqlite)
    model_name = SentenceTransformer(config["EMBEDDING_MODEL"]["model_name"])
    search_vec = np.asarray(model_name.encode(search_query), dtype=np.float32).tobytes()

    # search using the VSS virtual table and join with main table
    rows = cur.execute(
        f"""
    SELECT e.file_name, e.window_id_in_scene, e.window_text, v.distance
    FROM window_vss v
    JOIN window e ON e.rowid = v.rowid
    WHERE vss_search(
        v.embedding,
        vss_search_params(?, ?)
    )
    ORDER BY v.distance
    """,
        (search_vec, initial_k),
    ).fetchall()

    con.close()
    results = []

    for i, row in enumerate(rows):
        text_content = row["window_text"]
        preview = (
            text_content[:200] + "..." if len(text_content) > 200 else text_content
        )

        results.append(
            {
                "rank": i + 1,
                "episode": row["file_name"],
                "scene_id": row["window_id_in_scene"],
                "text": text_content,
                "score": f"{1 - row['distance']:.3f}",  # Convert distance to similarity
                "preview": preview,
                "distance": row["distance"],  # Keep original distance for reference
            }
        )

    return results


def cross_encoder(search_query: str, initial_k: int = 100, final_k: int = 10):
    config = toml.load("config.toml")
    initial_candidates = semantic_search(search_query, initial_k)

    print(f"Retrieved {len(initial_candidates)} initial candidates")

    cross_encoder = CrossEncoder(config["EMBEDDING_MODEL"]["crossencoder_model"])

    query_doc_pairs = []

    for c in initial_candidates:
        query_doc_pairs.append([search_query, c["text"]])

    print("Reranking with cross-encoder...")

    # Score all pairs with cross-encoder
    cross_encoder_scores = cross_encoder.predict(query_doc_pairs)

    # Add cross-encoder scores to candidates - convert to Python float for JSON serialization
    for i, candidate in enumerate(initial_candidates):
        candidate["x_score"] = float(
            cross_encoder_scores[i]
        )  # Convert numpy float32 to Python float

    # Combine scores with metadata and sort by cross-encoder score (top score first)
    reranked_results = sorted(
        initial_candidates, key=lambda x: x["x_score"], reverse=True
    )[:final_k]

    # Update results with final formatting
    results = []
    for i, val in enumerate(reranked_results):
        results.append(
            {
                "rank": i + 1,
                "episode": val["episode"],
                "scene_id": val["scene_id"],
                "text": val["text"],
                "score": f"{val['x_score']:.3f}",  # Use cross-encoder score
                "preview": val["preview"],
                "bi_encoder_dist": val["distance"],
                "cross_encoder_score": float(
                    val["x_score"]
                ),  # Ensure it's a Python float
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv("search_output.csv", index=False)

    return results
