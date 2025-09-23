from openai import OpenAI
from dotenv import load_dotenv
import os
import sqlite3
import toml
import sqlite_vss
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer


def make_embedding(script):
    """Create an embedding for the given script using OpenAI API."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(input=script, model="text-embedding-3-small")

    return response.data[0].embedding


def get_db_connection():
    """Get a connection to the vector database."""
    config = toml.load("config.toml")
    db_path = config.get("DB_PATH")
    con = sqlite3.connect(db_path)

    con.enable_load_extension(True)  # temporarily allow extension loading
    sqlite_vss.load(con)  # load the sqlite-vss extension
    con.enable_load_extension(False)  # disable extension loading again

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


def init_scene_tables(table_name: str = "scene"):
    """Initialize the scene table and VSS virtual table for the given table name."""
    con = get_db_connection()
    cur = con.cursor()
    config = toml.load("config.toml")
    embedding_model = config["EMBEDDING_MODEL"]["embedding_model"]
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


def init_window_tables(table_name: str = "window"):
    """Initialize the window table and VSS virtual table for the given table name."""
    con = get_db_connection()
    cur = con.cursor()

    config = toml.load("config.toml")
    embedding_model = config["EMBEDDING_MODEL"]["embedding_model"]
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


def insert_into_vss_table(chunk_type: str, row_id: int, embedding):
    """Insert a single embedding into the VSS virtual table."""
    con = get_db_connection()
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


def clear_table(table_name):
    """Clear all data from a table."""
    con = get_db_connection()
    cur = con.cursor()

    print(table_name)

    cur.execute(f"DELETE FROM {table_name}")

    con.commit()
    con.close()


def make_embeddings(chunk_type: str = "scene", embedding_model="sbert"):
    """Create embeddings for the specified chunk type and insert into DB."""

    clear_table(f"{chunk_type}_vss")

    # Establish what embeddings are being made
    if chunk_type == "scene":
        # Collect all scene data first to avoid connection conflicts
        iter_chunk = list(iter_scenes())
        index_field = "scene_id_in_episode"
        id_field = "scene_id"

    elif chunk_type == "window":
        # Collect all window data first to avoid connection conflicts
        iter_chunk = list(iter_windows())
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

        for chunk_id, embedding, chunk_text in zip(all_ids, all_embeddings, all_chunks):
            # Find the file name for this chunk (we need to look it up since zip doesn't preserve it)
            chunk_info = next(item for item in iter_chunk if item[id_field] == chunk_id)

            print(
                f"processing {chunk_type}s: episode {chunk_info['file_name']}, id: {chunk_id}"
            )

            insert_into_vss_table(chunk_type, chunk_id, embedding.tolist())

    elif embedding_model == "openAI":
        # Process the collected data
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

            # insert into db
            insert_into_vss_table(chunk_type, db_chunk_row[id_field], embedding)


def iter_scenes(batch_size: int = 500):
    """Yield scene rows from the DB in batches as dicts."""
    con = get_db_connection()
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
