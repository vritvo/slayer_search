import toml
import sqlite3
import sqlite_vss
from threading import Lock

# Connection pool for reusing database connections
_connection_pool = []
_pool_lock = Lock()
_pool_size = 5


def get_model_dim(model_name: str) -> int:
    """Look up embedding dimension for a model.
    Args:
        model_name: The model name (e.g., "all-MiniLM-L6-v2")
    Returns:
        The embedding dimension for the model
    """
    config = toml.load("config.toml")
    dimensions = config["EMBEDDING_MODEL"]["dimensions"]
    
    if model_name not in dimensions:
        raise KeyError(
            f"Model '{model_name}' not found in config dimensions lookup. "
            f"Add it to [EMBEDDING_MODEL.dimensions] in config.toml. "
            f"Available models: {list(dimensions.keys())}"
        )
    
    return dimensions[model_name]


def get_db_path(model_name: str, include_meta: bool, db_tag: str = "") -> str:
    """Derive database path from model configuration.
    
    Args:
        model_name: The model identifier (e.g., "all-MiniLM-L6-v2" or "jxm/cde-small-v2")
        include_meta: Whether embeddings include location/episode metadata
        db_tag: Optional experiment tag (e.g., "x2" for double-sampling)
    
    Returns:
        The database file path (e.g., "./vector_db_cde-small-v2-x2_no_meta.db")
    """
    # Sanitize model name: "jxm/cde-small-v2" → "cde-small-v2"
    base = model_name.split("/")[-1]
    
    tag_part = f"-{db_tag}" if db_tag else ""
    meta_part = "" if include_meta else "_no_meta"
    
    return f"./vector_db_{base}{tag_part}{meta_part}.db"


def get_db_connection(db_path: str = None):
    """Get database connection from pool or create new one.
    
    Args:
        db_path: Optional explicit database path. If provided, bypasses the pool
                 and creates a fresh connection (useful for evaluation across multiple dbs).
                 If None, derives path from config and uses connection pooling.
    """
    use_pool = db_path is None
    
    if use_pool:
        with _pool_lock:
            # Try to get connection from pool
            if _connection_pool:
                con = _connection_pool.pop()
                # Test if connection is still alive
                try:
                    con.execute("SELECT 1")
                    return con
                except:
                    # Connection is dead, create new one
                    pass

    # Create new connection
    if db_path is None:
        config = toml.load("config.toml")
        model_name = config["EMBEDDING_MODEL"]["model_name"]
        include_meta = config["EMBEDDING_MODEL"].get("include_meta", True)
        db_tag = config["EMBEDDING_MODEL"].get("db_tag", "")
        db_path = get_db_path(model_name, include_meta, db_tag)

    con = sqlite3.connect(db_path, timeout=20.0, check_same_thread=False)

    # Load the sqlite-vss extension
    con.enable_load_extension(True)
    try:
        sqlite_vss.load(con)
    except Exception as e:
        print(f"Warning: Could not load sqlite-vss extension: {e}")
    finally:
        con.enable_load_extension(False)

    # Set pragmas for better performance and larger cache
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-64000")  # 64MB cache (negative means KB)
    con.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp tables
    con.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O

    return con


def return_db_connection(con):
    """Return a connection to the pool for reuse."""
    with _pool_lock:
        if len(_connection_pool) < _pool_size:
            _connection_pool.append(con)
        else:
            # Pool is full, close the connection
            con.close()


def _resolve_model_config(model_name: str = None, include_meta: bool = None, db_tag: str = None):
    """Resolve model configuration parameters with all-or-nothing validation.
    
    Args:
        model_name: Optional model name. If any parameter is provided, all three must be provided.
        include_meta: Optional flag for including metadata. If any parameter is provided, all three must be provided.
        db_tag: Optional experiment tag. If any parameter is provided, all three must be provided.
    
    Returns:
        tuple: (model_name, include_meta, db_tag, db_path) - all resolved values
    
    Raises:
        ValueError: If only some parameters are provided (must provide all or none).
    """
    # All-or-nothing: if any parameter is provided, all must be provided (otherwise, config defaults are used)
    params_provided = [model_name is not None, include_meta is not None, db_tag is not None]
    if any(params_provided) and not all(params_provided):
        missing = []
        if model_name is None:
            missing.append("model_name")
        if include_meta is None:
            missing.append("include_meta")
        if db_tag is None:
            missing.append("db_tag")
        raise ValueError(
            f"Must provide all three parameters (model_name, include_meta, db_tag) or none. "
            f"Missing: {', '.join(missing)}. "
            f"This prevents accidentally inheriting config values (like db_tag) meant for a different model."
        )
    
    # If no params provided, read all from config
    if not any(params_provided):
        config = toml.load("config.toml")
        model_name = config["EMBEDDING_MODEL"]["model_name"]
        include_meta = config["EMBEDDING_MODEL"].get("include_meta", True)
        db_tag = config["EMBEDDING_MODEL"].get("db_tag", "")
    
    db_path = get_db_path(model_name, include_meta, db_tag)
    return model_name, include_meta, db_tag, db_path


def init_scene_tables(model_name: str = None, include_meta: bool = None, db_tag: str = None):
    """Initialize the scene table (needed for data processing).
    
    Args:
        model_name: Optional model name. If any parameter is provided, all three must be provided.
        include_meta: Optional flag for including metadata. If any parameter is provided, all three must be provided.
        db_tag: Optional experiment tag. If any parameter is provided, all three must be provided.
    
    Raises:
        ValueError: If only some parameters are provided (must provide all or none).
    """
    _, _, _, db_path = _resolve_model_config(model_name, include_meta, db_tag)
    con = get_db_connection(db_path=db_path)
    cur = con.cursor()

    # Create the scene table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scene (
            scene_id INTEGER PRIMARY KEY AUTOINCREMENT,
            scene_id_in_episode INTEGER,
            scene_text TEXT,
            file_name TEXT,
            scene_hash TEXT,
            location_text TEXT,
            location_descr TEXT
        )
    """)
    con.commit()
    con.close()


def init_window_tables(model_name: str = None, include_meta: bool = None, db_tag: str = None):
    """Initialize the window table and VSS virtual table.
    
    Args:
        model_name: Optional model name. If any parameter is provided, all three must be provided.
        include_meta: Optional flag for including metadata. If any parameter is provided, all three must be provided.
        db_tag: Optional experiment tag. If any parameter is provided, all three must be provided.
    
    Raises:
        ValueError: If only some parameters are provided (must provide all or none).
    """
    resolved_model_name, _, _, db_path = _resolve_model_config(model_name, include_meta, db_tag)
    embedding_dim = get_model_dim(resolved_model_name)
    
    con = get_db_connection(db_path=db_path)
    cur = con.cursor()

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


def clear_table(table_name, db_path: str = None):
    """Clear all data from a table.
    
    Args:
        table_name: Name of the table to clear
        db_path: Optional explicit database path. If None, reads from config.
    """
    con = get_db_connection(db_path=db_path)
    cur = con.cursor()
    cur.execute(f"DELETE FROM {table_name}")
    con.commit()
    con.close()
