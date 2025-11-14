import toml
import sqlite3
import re
import numpy as np
from utils.database import get_db_connection
from utils.models import _models
from utils.data_access import get_scene_from_id
import pandas as pd

def semantic_search(search_query: str, initial_k=10):
    config = toml.load("config.toml")
    # connect
    con = get_db_connection()
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Config
    initial_k_buffer = config["SEARCH"]["initial_k_buffer"]
    model_name = config["EMBEDDING_MODEL"]["model_name"]

    # Use cached model for app performance
    model = _models["bi_encoder"]

    # Keyword generation and text preprocessing:
    keywords = re.findall(r'"(.*?)"', search_query)
    search_query = search_query.replace('"', "")

    # Find all speakers: 
    speakers = re.findall(r'\b(\w+):', search_query)
    search_query = search_query.replace(':', '')

    # Initialize context_embeddings for all model types
    context_embeddings = None

    if model_name.startswith("nomic"):
        search_query = "search_query: " + search_query

    if model_name.startswith("jxm/cde"):
        context_embeddings = _models.get("context_embeddings")

    # convert to np array and then to bytes (BLOB for sqlite)
    if model_name.startswith("jxm/cde"):
        # encode with CDE method
        search_vec = model.encode(
            search_query,
            prompt_name="query",
            dataset_embeddings=context_embeddings,
            convert_to_tensor=False,
        )
        # Convert to bytes for sqlite-vss
        search_vec = np.asarray(search_vec, dtype=np.float32).tobytes()
    else:
        search_vec = np.asarray(model.encode(search_query), dtype=np.float32).tobytes()

    # Params to pass to SQL query
    query_params = [search_vec, initial_k * initial_k_buffer]

    # There may be a variable number of keywords, so for each one, we add a condition, and add it to query_params - each keyword must appear anywhere in the text (case insensitive)
    keyword_conditions = []
    if keywords:
        for keyword in keywords:
            keyword_conditions.append("LOWER(e.window_text) LIKE ?")
            query_params.append(f"%{keyword.lower()}%")

        keyword_filter = " AND " + " AND ".join(keyword_conditions)
    else:
        keyword_filter = ""

    # Similar search for speakers. But unlike for keywords, the speaker will always be the first word of a new line. 
    # We search for the speaker name after a newline
    speaker_conditions = []
    if speakers:
        for speaker in speakers:
            speaker_conditions.append("LOWER(e.window_text) LIKE ?")
            # Match speaker on its own line: preceded by a newline.
            query_params.append(f"%\n{speaker.lower()}%")
        
        speaker_filter = " AND " + " AND ".join(speaker_conditions)
    else:
        speaker_filter = ""

    # Find similar embeddings (keyword_filter is either empty or has AND conditions)
    sql_query = f"""
    SELECT e.file_name, e.scene_id, e.window_start, e.window_end, e.window_id_in_scene, e.window_text, v.distance
    FROM window_vss v
    JOIN window e ON e.rowid = v.rowid
    WHERE vss_search(
        v.embedding,
        vss_search_params(?, ?)
    ){keyword_filter}{speaker_filter}
    ORDER BY v.distance
    """
    rows = cur.execute(
        sql_query, query_params
    ).fetchall()  # query params gives values to fill in the ?s

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
                break
    # Fetch scene texts for all results (like in cross_encoder)
    scene_ids = tuple(result["scene_id"] for result in results)
    scene_id_dict = get_scene_from_id(scene_ids)

    # Add scene_text to each result
    for result in results:
        result["scene_text"] = scene_id_dict.get(result["scene_id"], "")

    return results


def cross_encoder(search_query: str, initial_k: int = 100, final_k: int = 10):
    # Use cached context embeddings instead of loading from disk every time
    initial_candidates = semantic_search(search_query, initial_k=initial_k)

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
