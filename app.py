from flask import Flask, render_template, request, jsonify
# from utils import semantic_search, cross_encoder, initialize_models
from utils.models import initialize_models
from utils.search import semantic_search, cross_encoder, set_profiling
import traceback
import toml
import sys
import time


app = Flask(__name__)

USE_CROSS_ENCODER = False
ENABLE_PROFILING = False  # Global flag controlled by command-line arg

# Load config once at startup
print("Loading config...")
CONFIG = toml.load("config.toml")

# Load models once at startup
print("Initializing models...")
start_time = time.time()
initialize_models()
init_time = time.time() - start_time
print(f"Models initialized in {init_time:.3f}s")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    try:
        total_start = time.time()
        
        data = request.get_json()
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"error": "Please enter a search query"}), 400

        search_start = time.time()
        results = perform_search(query)
        search_time = time.time() - search_start
        
        total_time = time.time() - total_start
        
        if ENABLE_PROFILING:
            print("\n=== SEARCH PROFILING ===")
            print(f"Query: '{query}'")
            print(f"Search execution: {search_time:.3f}s")
            print(f"Total request time: {total_time:.3f}s")
            print("========================\n")
        
        return jsonify({"results": results, "query": query})

    except Exception as e:
        print(f"Search error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": "Search failed. Please try again."}), 500


def perform_search(query: str):
    """Execute search"""
    initial_k = CONFIG["SEARCH"]["initial_k"]
    final_k = CONFIG["SEARCH"]["final_k"]
    initial_k_buffer = CONFIG["SEARCH"]["initial_k_buffer"]
    model_name = CONFIG["EMBEDDING_MODEL"]["model_name"]

    if USE_CROSS_ENCODER:
        return cross_encoder(
            query,
            initial_k=initial_k,
            final_k=final_k,
            initial_k_buffer=initial_k_buffer,
            model_name=model_name,
        )
    else:  # Without reranker - worse but faster results.
        return semantic_search(
            query,
            initial_k=initial_k,
            initial_k_buffer=initial_k_buffer,
            model_name=model_name,
        )


if __name__ == "__main__":
    # Check for --profile flag
    if "--profile" in sys.argv:
        ENABLE_PROFILING = True
        set_profiling(True)
        print("🔍 Profiling enabled")
    
    # app.run(debug=True, port=5001)
    app.run(debug=False, use_reloader=False, port=5002)
