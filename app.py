from flask import Flask, render_template, request, jsonify
# from utils import semantic_search, cross_encoder, initialize_models
from utils.models import initialize_models
from utils.search import semantic_search, cross_encoder
import traceback
import toml


app = Flask(__name__)

USE_CROSS_ENCODER = False

# Load models once at startup
initialize_models()


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
    config = toml.load("config.toml")
    initial_k = config["SEARCH"]["initial_k"]
    final_k = config["SEARCH"]["final_k"]

    if USE_CROSS_ENCODER:
        return cross_encoder(
            query,
            initial_k=initial_k,
            final_k=final_k,
        )
    else:  # Without reranker - worse but faster results.
        return semantic_search(
            query,
            initial_k=initial_k,
        )


if __name__ == "__main__":
    # app.run(debug=True, port=5001)
    app.run(debug=False, use_reloader=False, port=5002)
