from flask import Flask, render_template, request, jsonify
from utils import simple_search_db, semantic_search, cross_encoder
import json
import traceback

app = Flask(__name__)

# Configuration - you can move these to config.toml later
DEFAULT_CHUNK_TYPE = "window"
DEFAULT_EMBEDDING_MODEL = "sbert"
USE_CROSS_ENCODER = True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json()
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"error": "Please enter a search query"}), 400

        results = perform_search(query)
        return jsonify({"results": results, "query": query})

    except Exception as e:
        print(f"Search error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": "Search failed. Please try again."}), 500


def perform_search(query: str):
    """
    Execute search and format results for web display
    """

    if USE_CROSS_ENCODER:
        # This returns a DataFrame - need to convert to dict format
        df_results = cross_encoder(
            query,
            chunk_type=DEFAULT_CHUNK_TYPE,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            initial_k=200,
            final_k=15,
        )
        return format_cross_encoder_results(df_results)
    else:
        # This returns list of tuples (fname, idx, text, dist)
        raw_results = semantic_search(
            query,
            chunk_type=DEFAULT_CHUNK_TYPE,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            initial_k=15,
        )
        return format_semantic_results(raw_results)


def format_semantic_results(raw_results):
    """Convert semantic search tuples to web-friendly format"""
    formatted = []
    for i, (fname, idx, text, distance) in enumerate(raw_results):
        # Clean up episode name - remove .txt extension and format nicely
        episode_name = fname.replace(".txt", "") if fname.endswith(".txt") else fname

        formatted.append(
            {
                "rank": i + 1,
                "episode": episode_name,
                "scene_id": idx,
                "text": text,
                "score": f"{1 - distance:.3f}",  # Convert distance to similarity
                "preview": text[:200] + "..." if len(text) > 200 else text,
            }
        )
    return formatted


def format_cross_encoder_results(df_results):
    """Convert cross-encoder DataFrame to web-friendly format"""
    formatted = []
    for i, row in df_results.iterrows():
        # Clean up episode name - remove .txt extension and format nicely
        episode_name = (
            row["episode"].replace(".txt", "")
            if row["episode"].endswith(".txt")
            else row["episode"]
        )

        formatted.append(
            {
                "rank": i + 1,
                "episode": episode_name,
                "scene_id": row["index"],
                "text": row["text"],
                "score": f"{row['x_score']:.3f}",  # Use cross-encoder score
                "preview": row["text"][:200] + "..."
                if len(row["text"]) > 200
                else row["text"],
            }
        )
    return formatted


if __name__ == "__main__":
    # app.run(debug=True, port=5001)
    app.run(debug=False, use_reloader=False, port=5002)
