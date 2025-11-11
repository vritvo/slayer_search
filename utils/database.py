import toml
import sqlite3
import sqlite_vss


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
            file_name TEXT,
            location_text TEXT,
            location_descr TEXT
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

def clear_table(table_name):
    """Clear all data from a table."""
    con = get_db_connection()
    cur = con.cursor()
    cur.execute(f"DELETE FROM {table_name}")
    con.commit()
    con.close()
