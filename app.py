from flask import Flask, render_template, request, jsonify
from utils import semantic_search, cross_encoder
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
    """Execute search"""
    if USE_CROSS_ENCODER:
        return cross_encoder(
            query,
            chunk_type=DEFAULT_CHUNK_TYPE,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            initial_k=200,
            final_k=15,
        )
    else:
        return semantic_search(
            query,
            chunk_type=DEFAULT_CHUNK_TYPE,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            initial_k=15,
        )


if __name__ == "__main__":
    # app.run(debug=True, port=5001)
    app.run(debug=False, use_reloader=False, port=5002)
