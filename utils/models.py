# Global model storage
import toml
import torch
import random
import time
from sentence_transformers import SentenceTransformer, CrossEncoder  # sbert
from utils.database import clear_table
from utils.data_access import iter_windows, batch_insert_into_vss_table

_models = {}

def initialize_models():
    """Load all models at startup and store them globally."""
    print("Loading models...")

    config = toml.load("config.toml")
    model_name = config["EMBEDDING_MODEL"]["model_name"]
    # Load bi-encoder model
    print(f"Loading bi-encoder: {config['EMBEDDING_MODEL']['model_name']}")

    if model_name.startswith("nomic"):
        _models["bi_encoder"] = SentenceTransformer(model_name, trust_remote_code=True)
        _models["context_embeddings"] = None
    elif model_name.startswith("jxm/cde"):
        _models["bi_encoder"] = SentenceTransformer(model_name, trust_remote_code=True)
        # Try to load context embeddings if they exist, but don't fail if they don't
        try:
            _models["context_embeddings"] = torch.load("buffy_dataset_context.pt")
            print("Loaded existing context embeddings for CDE model")
        except FileNotFoundError:
            print(
                "Context embeddings not found - will be created when running make_embeddings()"
            )
            _models["context_embeddings"] = None
    else:
        _models["bi_encoder"] = SentenceTransformer(model_name)
        _models["context_embeddings"] = None

    # Load cross-encoder model
    print(f"Loading cross-encoder: {config['EMBEDDING_MODEL']['crossencoder_model']}")
    _models["cross_encoder"] = CrossEncoder(
        config["EMBEDDING_MODEL"]["crossencoder_model"]
    )

    print("Models loaded successfully")

def make_embeddings():
    """Create embeddings for window chunks and insert into DB."""
    clear_table("window_vss")

    # Collect all window data first to avoid connection conflicts
    iter_chunk = list(iter_windows())

    config = toml.load("config.toml")
    model_name = config["EMBEDDING_MODEL"]["model_name"]
    print(model_name)
    # model = SentenceTransformer(model_name)
    model = _models["bi_encoder"]

    if model_name.startswith("nomic"):
        doc_formatting = "search_document: "
    else:
        doc_formatting = ""

    all_chunks = []
    all_ids = []
    for db_chunk_row in iter_chunk:
        embedded_text = f"{doc_formatting}episode: {db_chunk_row['file_name']}:\n{db_chunk_row['text']}"
        all_chunks.append(embedded_text)
        all_ids.append(db_chunk_row["window_id"])

    print(f"Creating embeddings for {len(all_chunks)} window chunks...")
    if model_name.startswith("jxm/cde"):
        print("Train Model 1")
        # Train Model 1
        # make the mini corpus.

        # Divide chunks by season by creating a dict: {season: [chunks]}
        seasons = {}
        for chunk in all_chunks:
            season = chunk.split("episode: ")[1].split("x")[0]
            if season not in seasons:
                seasons[season] = []
            seasons[season].append(chunk)
        # Randomly shuffle the chunks in each season
        for season in seasons:
            random.shuffle(seasons[season])

        context_size = model[0].config.transductive_corpus_size
        multiples = 2

        start_time = time.time()

        context_embeddings_multiple = []
        for m in range(multiples):
            mini_corpus = []

            # Iterate over the chunks for each season, and sample evenly.
            for season_name, season_chunks in seasons.items():
                k = context_size // len(seasons)

                # Check if we have enough chunks left in this season
                if len(season_chunks) >= k:
                    # Sample k chunks from the end (or could use random.sample)
                    sampled_chunks = season_chunks[-k:]
                    mini_corpus.extend(sampled_chunks)
                    # Remove the used chunks from the season
                    seasons[season_name] = season_chunks[:-k]
                else:
                    # If not enough chunks left, take all remaining
                    mini_corpus.extend(season_chunks)
                    seasons[season_name] = []

            # In case of rounding issues or exhausted seasons, fill up to context_size
            while len(mini_corpus) < context_size:
                choice = random.choice(all_chunks)
                if choice not in mini_corpus:
                    mini_corpus.append(choice)

            print(f"Mini corpus {m + 1} size: {len(mini_corpus)}")

            # Compute the dataset context embeddings
            context_embeddings = model.encode(
                mini_corpus, prompt_name="document", convert_to_tensor=True
            )
            context_embeddings_multiple.append(context_embeddings)

        context_embeddings = torch.mean(torch.stack(context_embeddings_multiple), dim=0)

        end_time = time.time()
        print(f"Computed context embeddings in {end_time - start_time:.2f} seconds.")

        # Persist for reuse (both indexing and queries will need the same tensor)
        torch.save(context_embeddings, "buffy_dataset_context.pt")

        # Update the cached embeddings in global storage
        _models["context_embeddings"] = context_embeddings

        # ---- Stage 2: embed all documents conditioned on the cached context ----
        start_time = time.time()
        print("Train Model 2 (document embeddings)")
        all_embeddings = model.encode(
            all_chunks,  # full corpus at retrieval granularity
            prompt_name="document",
            dataset_embeddings=context_embeddings,
            convert_to_tensor=False,
        )
        end_time = time.time()
        print(f"Computed all embeddings in {end_time - start_time:.2f} seconds.")

    else:
        # Non-CDE path:
        start_time = time.time()
        all_embeddings = model.encode(all_chunks)
        end_time = time.time()
        print(f"Computed all embeddings in {end_time - start_time:.2f} seconds.")
        # TODO: encode vs encode_document https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html

    # Prepare embeddings data for batch insertion
    embeddings_data = []
    for chunk_id, embedding in zip(all_ids, all_embeddings):
        # ensure plain list for sqlite-vss adapter
        embeddings_data.append((chunk_id, embedding.tolist()))

    # Actually insert the embeddings into the database
    print(f"Batch inserting {len(embeddings_data)} window embeddings...")
    batch_insert_into_vss_table(embeddings_data)
