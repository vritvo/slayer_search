from openai import OpenAI
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


def make_embedding(script):
    """Create an embedding for the given script using OpenAI API."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(input=script, model="text-embedding-3-small")

    return response.data[0].embedding


def get_db_path(embedding_model: str) -> str:
    """Get the correct database path based on embedding model."""
    config = toml.load("config.toml")

    if embedding_model not in ["sbert", "openAI"]:
        raise ValueError("Invalid embedding_model. Must be 'sbert' or 'openAI'.")

    return config["DATABASE"][embedding_model]


def get_db_connection(embedding_model: str = "sbert"):
    """Get database connection with the correct path for the embedding model."""
    db_path = get_db_path(embedding_model)
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


def init_scene_tables(table_name: str = "scene", embedding_model: str = None):
    """Initialize the scene table and VSS virtual table for the given table name."""
    config = toml.load("config.toml")
    if embedding_model is None:
        embedding_model = config["EMBEDDING_MODEL"]["embedding_model"]

    con = get_db_connection(embedding_model)
    cur = con.cursor()

    if embedding_model == "sbert":
        embedding_dim = config["EMBEDDING_MODEL"]["sbert_dim"]
    elif embedding_model == "openAI":
        embedding_dim = config["EMBEDDING_MODEL"]["oai_dim"]

    # Create the Scene table
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            scene_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            scene_id_in_episode INTEGER,
            scene_text TEXT
        )
    """)

    # Create virtual table for vector search using sqlite-vss

    cur.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {table_name}_vss USING vss0(
            embedding({embedding_dim})
        )
    """)
    con.commit()
    con.close()


def init_window_tables(table_name: str = "window", embedding_model: str = None):
    """Initialize the window table and VSS virtual table for the given table name."""
    config = toml.load("config.toml")
    if embedding_model is None:
        embedding_model = config["EMBEDDING_MODEL"]["embedding_model"]

    con = get_db_connection(embedding_model)
    cur = con.cursor()

    if embedding_model == "sbert":
        embedding_dim = config["EMBEDDING_MODEL"]["sbert_dim"]
    elif embedding_model == "openAI":
        embedding_dim = config["EMBEDDING_MODEL"]["oai_dim"]

    # Create the window table
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            window_id INTEGER PRIMARY KEY AUTOINCREMENT,
            scene_id INTEGER,
            window_id_in_scene INTEGER,
            window_text TEXT,
            file_name TEXT
        )
    """)

    # Create virtual table for vector search using sqlite-vss

    cur.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {table_name}_vss USING vss0(
            embedding({embedding_dim})
        )
    """)
    con.commit()
    con.close()


def insert_into_vss_table(
    chunk_type: str, row_id: int, embedding, embedding_model: str = None
):
    """Insert a single embedding into the VSS virtual table."""
    con = get_db_connection(embedding_model)
    cur = con.cursor()

    table_name = chunk_type
    vss_table_name = f"{table_name}_vss"

    cur.execute(
        f"""
        INSERT INTO {vss_table_name}(rowid, embedding)
        VALUES (?, ?)
    """,
        (row_id, json.dumps(embedding)),
    )

    con.commit()
    con.close()


def batch_insert_into_vss_table(
    chunk_type: str, embeddings_data, embedding_model: str = None
):
    """Insert multiple embeddings into the VSS virtual table using batch processing."""
    con = get_db_connection(embedding_model)
    cur = con.cursor()

    table_name = chunk_type
    vss_table_name = f"{table_name}_vss"
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
                INSERT INTO {vss_table_name}(rowid, embedding)
                VALUES (?, ?)
            """,
                batch_values,
            )

            con.commit()  # Commit each batch
            print(
                f"Inserted batch {i // batch_size + 1}: {len(batch)} embeddings into {vss_table_name}"
            )

    except Exception as e:
        print(f"Error in batch_insert_into_vss_table: {e}")
        con.rollback()
        raise
    finally:
        con.close()


def clear_table(table_name, embedding_model: str = None):
    """Clear all data from a table."""
    con = get_db_connection(embedding_model)
    cur = con.cursor()

    cur.execute(f"DELETE FROM {table_name}")

    con.commit()
    con.close()


def make_embeddings(chunk_type: str = "scene", embedding_model="sbert"):
    """Create embeddings for the specified chunk type and insert into DB."""

    clear_table(f"{chunk_type}_vss", embedding_model)

    # Establish what embeddings are being made
    if chunk_type == "scene":
        # Collect all scene data first to avoid connection conflicts
        iter_chunk = list(iter_scenes(embedding_model=embedding_model))
        index_field = "scene_id_in_episode"
        id_field = "scene_id"

    elif chunk_type == "window":
        # Collect all window data first to avoid connection conflicts
        iter_chunk = list(iter_windows(embedding_model=embedding_model))
        index_field = "window_id_in_scene"
        id_field = "window_id"
    else:
        raise ValueError("Invalid chunk_type. Must be 'scene' or 'window'.")

    # Which embedding model
    if embedding_model == "sbert":
        config = toml.load("config.toml")
        sbert_model_name = config["EMBEDDING_MODEL"]["sbert_model"]
        model = SentenceTransformer(sbert_model_name)

        all_chunks = []
        all_ids = []
        for db_chunk_row in iter_chunk:
            all_chunks.append(db_chunk_row["text"])
            all_ids.append(db_chunk_row[id_field])

        print(f"Creating embeddings for {len(all_chunks)} {chunk_type} chunks...")
        all_embeddings = model.encode(all_chunks)
        # TODO: encode vs encode_document https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html

        # Prepare embeddings data for batch insertion
        embeddings_data = []
        for chunk_id, embedding in zip(all_ids, all_embeddings):
            embeddings_data.append((chunk_id, embedding.tolist()))

        print(f"Batch inserting {len(embeddings_data)} {chunk_type} embeddings...")
        batch_insert_into_vss_table(chunk_type, embeddings_data, embedding_model)

    elif embedding_model == "openAI":
        # Process the collected data and accumulate embeddings for batch insertion
        embeddings_data = []

        for db_chunk_row in iter_chunk:
            chunk = db_chunk_row["text"]

            # print episode, chunk info, and chunk id being processed.
            if chunk_type == "scene":
                print(
                    f"Processing {db_chunk_row['file_name']} scene {db_chunk_row[index_field]} (ID {db_chunk_row[id_field]})"
                )
            else:  # window
                print(
                    f"Processing {db_chunk_row['file_name']} window {db_chunk_row[index_field]} from scene {db_chunk_row['scene_id']} (ID {db_chunk_row[id_field]})"
                )

            try:
                embedding = make_embedding(chunk)
                # Accumulate embeddings for batch insertion
                embeddings_data.append((db_chunk_row[id_field], embedding))
            except Exception as e:
                if "maximum context length" in str(e):
                    log_oversized_chunk(
                        db_chunk_row["file_name"], db_chunk_row[index_field], len(chunk)
                    )
                    print(
                        f"Skipping oversized chunk: {db_chunk_row['file_name']} ({chunk_type} {db_chunk_row[index_field]}) - {len(chunk)} characters"
                    )
                    continue
                else:
                    raise e  # Re-raise other errors

        # Batch insert all collected embeddings
        if embeddings_data:
            print(f"Batch inserting {len(embeddings_data)} {chunk_type} embeddings...")
            batch_insert_into_vss_table(chunk_type, embeddings_data, embedding_model)


def iter_scenes(batch_size: int = 500, embedding_model: str = None):
    """Yield scene rows from the DB in batches as dicts."""
    con = get_db_connection(embedding_model)
    try:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("""
            SELECT scene_id, file_name, scene_id_in_episode, scene_text
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
                    "file_name": r["file_name"],
                    "scene_id_in_episode": r["scene_id_in_episode"],
                    "text": r["scene_text"],
                }

            con.commit()
    except Exception as e:
        print(f"Error in iter_scenes: {e}")
    finally:
        con.close()


def iter_windows(batch_size: int = 500, embedding_model: str = None):
    """Yield window rows from the DB in batches as dicts."""
    con = get_db_connection(embedding_model)
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


def simple_search_db(
    search_query: str,
    chunk_type: str = "window",
    embedding_model: str = "sbert",
    initial_k=10,
):
    rows = semantic_search(
        search_query, chunk_type, embedding_model, initial_k=initial_k
    )

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

    results = []
    for i, (cross_encoder_score, (fname, idx, text, bi_score)) in enumerate(
        reranked_results[:final_k]
    ):
        print(f"#{i + 1}: {fname} [{idx}]")
        print(
            f"Cross-encoder score: {cross_encoder_score:.4f} | Bi-encoder distance: {bi_score:.4f}"
        )
        print(f"{text[:200]}...")
        print("-----\n")

        results.append(
            {
                "i": i,
                "query": search_query,
                "x_score": cross_encoder_score,
                "b_score": bi_score,
                "episode": fname,
                "index": idx,
                "text": text,
                "embedding_model": embedding_model,
            }
        )

        results_df = pd.DataFrame(results)
        results_df.to_csv("search_output.csv", index=False)

    return results_df

    #
