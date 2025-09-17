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


def init_scene_tables(table_name: str = "scene"):
    """Initialize the scene table and VSS virtual table for the given table name."""
    con = get_db_connection()
    cur = con.cursor()

    # Create the Scene table
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


def init_window_tables(table_name: str = "window"):
    """Initialize the window table and VSS virtual table for the given table name."""
    con = get_db_connection()
    cur = con.cursor()

    # Create the window table
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scene_id INTEGER,
            window_index INTEGER,
            window_text TEXT,
            file_name TEXT
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
    con = get_db_connection()
    cur = con.cursor()

    cur.execute(f"DELETE FROM {table_name}")

    con.commit()
    con.close()
