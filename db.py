import sqlite3
import toml
import json
import sqlite_vss


def get_db_connection():
    """Get a connection to the vector database."""

    config = toml.load("config.toml")
    db_path = config.get("DB_PATH")
    con = sqlite3.connect(db_path)

    con.enable_load_extension(True)  # temporarily allow extension loading
    sqlite_vss.load(con)  # load the sqlite-vss extension
    con.enable_load_extension(False)  # disable extension loading again

    return con


def init_embeddings_table(chunk_type: str):
    """Initialize the embeddings table for the given chunk type."""
    con = get_db_connection()
    cur = con.cursor()

    table_name = f"{chunk_type}_embeddings"

    # Create the main table
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            chunk_index INTEGER,
            chunk_text TEXT
        )
    """)

    # Create virtual table for vector search using sqlite-vss
    # Assuming embeddings are 1536 dimensions (OpenAI default)
    cur.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {table_name}_vss USING vss0(
            embedding(1536)
        )
    """)
    con.commit()
    con.close()


def insert_embedding_batch(chunk_type: str, embeddings_data):
    """Insert a batch of embeddings into the database."""
    con = get_db_connection()
    cur = con.cursor()

    table_name = f"{chunk_type}_embeddings"
    vss_table_name = f"{table_name}_vss"

    # Insert into main table first and collect row IDs
    inserted_rows = []

    for item in embeddings_data:
        cur.execute(
            f"""
            INSERT INTO {table_name}
            (file_name, chunk_index, chunk_text)
            VALUES (?, ?, ?)
        """,
            (
                item["file_name"],
                item["chunk_index"],
                item["chunk_text"],
            ),
        )

        # Save (row_id, raw embedding) for virtual table
        row_id = cur.lastrowid
        inserted_rows.append((row_id, item["embedding"]))

    # Insert into VSS table for vector search
    for row_id, embedding in inserted_rows:
        cur.execute(
            f"""
            INSERT INTO {vss_table_name}(rowid, embedding)
            VALUES (?, ?)
        """,
            (row_id, json.dumps(embedding)),
        )

    con.commit()
    con.close()

    return len(embeddings_data)


def clear_embeddings_table(chunk_type: str):
    """Clear all data from the embeddings table."""
    con = get_db_connection()
    cur = con.cursor()

    table_name = f"{chunk_type}_embeddings"
    vss_table_name = f"{table_name}_vss"

    cur.execute(f"DELETE FROM {table_name}")
    cur.execute(f"DELETE FROM {vss_table_name}")

    con.commit()
    con.close()
