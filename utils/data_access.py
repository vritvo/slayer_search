import sqlite3
import json
from utils.database import get_db_connection, return_db_connection 
import time

def iter_scenes(batch_size: int = 500):
    """Yield scene rows from the DB in batches as dicts. Used for data processing."""
    con = get_db_connection()
    try:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("""
            SELECT scene_id, scene_id_in_episode, scene_text, file_name, scene_hash, location_text, location_descr
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
                    "scene_hash": r["scene_hash"],
                    "location_text": r["location_text"],
                    "location_descr": r["location_descr"],
                }
            con.commit()
    except Exception as e:
        print(f"Error in iter_scenes: {e}")
    finally:
        con.close()

def iter_windows(batch_size: int = 500):
    """Yield window rows from the DB in batches as dicts with location info from parent scene."""
    con = get_db_connection()
    try:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("""
            SELECT 
                w.window_id, 
                w.scene_id, 
                w.window_id_in_scene, 
                w.window_text, 
                w.file_name,
                s.scene_hash,
                s.location_text,
                s.location_descr
            FROM window w
            JOIN scene s ON w.scene_id = s.scene_id
            ORDER BY w.file_name, w.scene_id, w.window_id_in_scene
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
                    "scene_hash": r["scene_hash"],
                    "location_text": r["location_text"],
                    "location_descr": r["location_descr"],
                }
            con.commit()
    except Exception as e:
        print(f"Error in iter_windows: {e}")
    finally:
        con.close()
        

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
        return_db_connection(con)
        
        
def batch_insert_into_vss_table(embeddings_data):
    """Insert multiple embeddings into the VSS virtual table using batch processing."""
    con = get_db_connection()
    cur = con.cursor()

    batch_size = 1000
    num_batches = (len(embeddings_data) + batch_size - 1) // batch_size

    try:
        for i in range(0, len(embeddings_data), batch_size):
            batch_start = time.time()
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
            batch_time = time.time() - batch_start
            batch_num = i // batch_size + 1
            print(
                f"  Batch {batch_num}/{num_batches}: Inserted {len(batch)} embeddings in {batch_time:.2f}s"
            )

    except Exception as e:
        print(f"Error in batch_insert_into_vss_table: {e}")
        con.rollback()
        raise
    finally:
        con.close()


