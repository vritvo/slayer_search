import sqlite3
import toml
import numpy as np
import sqlite_vss
import json
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


def init_scene_tables():
    """Initialize the scene table (needed for data processing)."""
    con = get_db_connection()
    cur = con.cursor()

    # Create the scene table
    cur.execute("""
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
    cur.execute("""
        CREATE TABLE IF NOT EXISTS window (
            window_id INTEGER PRIMARY KEY AUTOINCREMENT,
            scene_id INTEGER,
            window_id_in_scene INTEGER,
            window_text TEXT,
            file_name TEXT,
            window_start INTEGER,
            window_end INTEGER
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


def batch_insert_into_vss_table(embeddings_data):
    """Insert multiple embeddings into the VSS virtual table using batch processing."""
    con = get_db_connection()
    cur = con.cursor()

    batch_size = 1000

    try:
        for i in range(0, len(embeddings_data), batch_size):
            batch = embeddings_data[i : i + batch_size]

            batch_values = [
                (row_id, json.dumps(embedding)) for row_id, embedding in batch
            ]

            cur.executemany(
                """
                INSERT INTO window_vss(rowid, embedding)
                VALUES (?, ?)
            """,
                batch_values,
            )

            con.commit()
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
    if not model_name.startswith("cde"):
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


# Global model storage
_models = {}


def initialize_models():
    """Load all models at startup and store them globally."""
    print("Loading models...")

    config = toml.load("config.toml")

    # Load bi-encoder model
    print(f"Loading bi-encoder: {config['EMBEDDING_MODEL']['model_name']}")
    _models["bi_encoder"] = SentenceTransformer(config["EMBEDDING_MODEL"]["model_name"])

    # Load cross-encoder model
    print(f"Loading cross-encoder: {config['EMBEDDING_MODEL']['crossencoder_model']}")
    _models["cross_encoder"] = CrossEncoder(
        config["EMBEDDING_MODEL"]["crossencoder_model"]
    )

    print("Models loaded successfully")


def get_scene_from_id(scene_ids: tuple) -> dict:
    """Fetch scene text for given scene IDs."""
    if not scene_ids:
        return {}

    con = get_db_connection()
    try:
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        # Use parameterized query for safety
        placeholders = ",".join(
            "?" * len(scene_ids)
        )  # technically we could just use the scene_ids tuple directly, but this is more robust
        rows = cur.execute(
            f"""
            SELECT scene_id, scene_text, file_name
            FROM scene
            WHERE scene_id IN ({placeholders})
            ORDER BY scene_id
            """,
            scene_ids,
        ).fetchall()

        results = {}
        for row in rows:
            results[row["scene_id"]] = row["scene_text"]

        return results
    except Exception as e:
        print(f"Error in get_scene_from_id: {e}")
        return {}
    finally:
        con.close()


def semantic_search(search_query: str, initial_k=10):
    # connect
    con = get_db_connection()
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Config
    config = toml.load("config.toml")
    initial_k_buffer = config["SEARCH"]["initial_k_buffer"]

    # Use cached model for app performance
    model = _models["bi_encoder"]

    # convert to np array and then to bytes (BLOB for sqlite)
    search_vec = np.asarray(model.encode(search_query), dtype=np.float32).tobytes()

    # search using the VSS virtual table and join with main table
    rows = cur.execute(
        """
    SELECT e.file_name, e.scene_id, e.window_start, e.window_end, e.window_id_in_scene, e.window_text, v.distance
    FROM window_vss v
    JOIN window e ON e.rowid = v.rowid
    WHERE vss_search(
        v.embedding,
        vss_search_params(?, ?)
    )
    ORDER BY v.distance
    """,
        (
            search_vec,
            initial_k * initial_k_buffer,
        ),  # We would just do initial K here, but we need a buffer because we'll deduplicate chunks within a scene
    ).fetchall()

    con.close()
    results = []
    included_scenes = set()
    initial_k_counter = 0

    for i, row in enumerate(rows):
        text_content = row["window_text"]
        preview = (
            text_content[:200] + "..." if len(text_content) > 200 else text_content
        )

        # If the scene has not already been included in one of the top results, we skip it.
        if row["scene_id"] not in included_scenes:
            included_scenes.add(row["scene_id"])
            results.append(
                {
                    "rank": i + 1,
                    "episode": row["file_name"],
                    "scene_id": row["scene_id"],
                    "window_start": row["window_start"],
                    "window_end": row["window_end"],
                    "chunk_id": row["window_id_in_scene"],
                    "text": text_content,
                    "score": f"{1 - row['distance']:.3f}",  # Convert distance to similarity
                    "preview": preview,
                    "distance": row["distance"],  # Keep original distance for reference
                }
            )
            initial_k_counter += 1
            if initial_k_counter >= initial_k:
                return results
    return results


def cross_encoder(search_query: str, initial_k: int = 100, final_k: int = 10):
    initial_candidates = semantic_search(search_query, initial_k)
    print(f"Retrieved {len(initial_candidates)} initial candidates")

    # Use the pre-loaded cross-encoder (no loading time)
    cross_encoder_model = _models["cross_encoder"]

    query_doc_pairs = []
    for c in initial_candidates:
        query_doc_pairs.append([search_query, c["text"]])

    print("Reranking with cross-encoder...")

    # Score all pairs with cross-encoder
    cross_encoder_scores = cross_encoder_model.predict(query_doc_pairs)

    # Add cross-encoder scores to candidates - convert to Python float for JSON serialization
    for i, candidate in enumerate(initial_candidates):
        candidate["x_score"] = float(
            cross_encoder_scores[i]
        )  # Convert numpy float32 to Python float

    # Combine scores with metadata and sort by cross-encoder score (top score first)
    reranked_results = sorted(
        initial_candidates, key=lambda x: x["x_score"], reverse=True
    )[:final_k]

    scene_ids = set()
    for i, result in enumerate(reranked_results):
        scene_ids.add(result["scene_id"])
    scene_ids = tuple(scene_ids)
    scene_id_dict = get_scene_from_id(scene_ids)

    # Update results with final formatting
    results = []
    for i, val in enumerate(reranked_results):
        results.append(
            {
                "rank": i + 1,
                "episode": val["episode"],
                "scene_id": val["scene_id"],
                "scene_text": scene_id_dict[val["scene_id"]],
                "window_start": val["window_start"],
                "window_end": val["window_end"],
                "chunk_id": val["chunk_id"],
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
