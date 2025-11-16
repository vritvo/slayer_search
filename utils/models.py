# Global model storage
import toml
import torch
import random
import time
from sentence_transformers import SentenceTransformer, CrossEncoder  # sbert
from utils.database import clear_table
from utils.data_access import iter_windows, batch_insert_into_vss_table
import langextract as lx
import textwrap
import os
from dotenv import load_dotenv

_models = {}


def initialize_models(use_gpu_for_building=False):
    """Load all models at startup and store them globally.
    
    Args:
        use_gpu_for_building: If True, auto-detect and use best available GPU 
                             (CUDA/MPS) for batch embedding building.
                             If False, use CPU (better for single-query search).
    """
    print("Loading models...")

    config = toml.load("config.toml")
    model_name = config["EMBEDDING_MODEL"]["model_name"]
    
    # Choose device based on use case
    if use_gpu_for_building:
        # Auto-detect best available GPU
        if torch.cuda.is_available():
            device = 'cuda'
            print("Using CUDA (NVIDIA GPU) for batch embedding building")
        elif torch.backends.mps.is_available():
            device = 'mps'
            print("Using MPS (Apple GPU) for batch embedding building")
        else:
            device = 'cpu'
            print("GPU requested but not available, using CPU")
    else:
        device = 'cpu'
        print("Using CPU for consistent inference")
    
    # Load bi-encoder model on chosen device
    print(f"Loading bi-encoder: {config['EMBEDDING_MODEL']['model_name']}")

    if model_name.startswith("nomic"):
        _models["bi_encoder"] = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        _models["context_embeddings"] = None
    elif model_name.startswith("jxm/cde"):
        _models["bi_encoder"] = SentenceTransformer(model_name, trust_remote_code=True, device=device)
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
        _models["bi_encoder"] = SentenceTransformer(model_name, device=device)
        _models["context_embeddings"] = None

    # Set model to eval mode for inference
    _models["bi_encoder"].eval()
    
    # Warm up the model to optimize tokenizer and inference
    print("Warming up bi-encoder model...")
    with torch.no_grad():
        _models["bi_encoder"].encode("warmup query", show_progress_bar=False, convert_to_numpy=True)
    print("Model ready")

    # Load cross-encoder model
    print(f"Loading cross-encoder: {config['EMBEDDING_MODEL']['crossencoder_model']}")
    _models["cross_encoder"] = CrossEncoder(
        config["EMBEDDING_MODEL"]["crossencoder_model"]
    )

    print("Models loaded successfully")


def make_embeddings():
    """Create embeddings for window chunks and insert into DB."""
    overall_start = time.time()
    
    print("Clearing window_vss table...")
    clear_table("window_vss")

    # Collect all window data first to avoid connection conflicts
    print("Loading window data from database...")
    load_start = time.time()
    iter_chunk = list(iter_windows())
    load_time = time.time() - load_start
    print(f"  Loaded {len(list(iter_chunk))} windows in {load_time:.2f}s")

    config = toml.load("config.toml")
    model_name = config["EMBEDDING_MODEL"]["model_name"]
    model = _models["bi_encoder"]
    
    # Verify device
    print(f"\nModel: {model_name}")
    print(f"Device: {model.device}")

    if model_name.startswith("nomic"):
        doc_formatting = "search_document: "
    else:
        doc_formatting = ""

    print("\nPreparing text for embedding...")
    prep_start = time.time()
    all_chunks = []
    all_ids = []
    for db_chunk_row in iter_chunk:
        # Build embedded text with optional location prefix
        location_prefix = ""
        if db_chunk_row.get("location_descr"):
            location_prefix = f"Location: {db_chunk_row['location_descr']}\n"

        embedded_text = f"{doc_formatting}\nEpisode: {db_chunk_row['file_name']}\n{location_prefix}Text: \n{db_chunk_row['text']}"
        all_chunks.append(embedded_text)
        all_ids.append(db_chunk_row["window_id"])
    prep_time = time.time() - prep_start
    print(f"  Prepared {len(all_chunks)} chunks in {prep_time:.2f}s")

    print(f"\nCreating embeddings for {len(all_chunks)} window chunks...")
    print(f"  Average chunk length: {sum(len(c) for c in all_chunks) / len(all_chunks):.0f} chars")
    
    # Adjust batch size based on device
    device_type = str(model.device).split(':')[0]  # Get 'cpu', 'cuda', or 'mps'
    if device_type in ['cuda', 'mps']:
        batch_size = 64  # Conservative batch size for GPU (avoids OOM)
        print(f"  Using batch_size={batch_size} for GPU ({device_type})")
    else:
        batch_size = 64  # Optimal for CPU
        print(f"  Using batch_size={batch_size} for CPU")
    
    if model_name.startswith("jxm/cde"):
        print("Train Model 1")
        # Train Model 1
        # make the mini corpus.

        # Divide chunks by season by creating a dict: {season: [chunks]}
        seasons = {}
        for chunk in all_chunks:
            season = chunk.split("Episode: ")[1].split("x")[0]
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
        print("\nTrain Model 2 (document embeddings)")
        start_time = time.time()
        all_embeddings = model.encode(
            all_chunks,  # full corpus at retrieval granularity
            prompt_name="document",
            dataset_embeddings=context_embeddings,
            convert_to_tensor=False,
            batch_size=batch_size,
            show_progress_bar=True
        )
        end_time = time.time()
        chunks_per_sec = len(all_chunks) / (end_time - start_time)
        print(f"  Encoded {len(all_chunks)} chunks in {end_time - start_time:.2f}s ({chunks_per_sec:.1f} chunks/s)")

    else:
        # Non-CDE path:
        print("\nEncoding all chunks...")
        start_time = time.time()
        all_embeddings = model.encode(
            all_chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        end_time = time.time()
        chunks_per_sec = len(all_chunks) / (end_time - start_time)
        print(f"  Encoded {len(all_chunks)} chunks in {end_time - start_time:.2f}s ({chunks_per_sec:.1f} chunks/s)")
        # TODO: encode vs encode_document https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html

    # Prepare embeddings data for batch insertion
    print("\nPreparing embeddings for database insertion...")
    prep_start = time.time()
    embeddings_data = []
    for chunk_id, embedding in zip(all_ids, all_embeddings):
        # ensure plain list for sqlite-vss adapter
        embeddings_data.append((chunk_id, embedding.tolist()))
    prep_time = time.time() - prep_start
    print(f"  Prepared {len(embeddings_data)} embeddings in {prep_time:.2f}s")

    # Actually insert the embeddings into the database
    print(f"\nInserting {len(embeddings_data)} embeddings into database...")
    insert_start = time.time()
    batch_insert_into_vss_table(embeddings_data)
    insert_time = time.time() - insert_start
    print(f"  Inserted all embeddings in {insert_time:.2f}s")
    
    total_time = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"{'='*60}")


def tag_text(input_text, generate_html=False):
    # Make sure there is an input text, otherwise error:
    if not input_text:
        raise ValueError("No input text provided")

    # Load environment variables from .env file
    load_dotenv()
    API_KEY = os.getenv("LANGEXTRACT_API_KEY")
    config = toml.load("config.toml")

    prompt = textwrap.dedent("""\
        Extract location.
        Use exact text for extractions. Do not paraphrase or overlap entities. 
        For the attribute location_descr, provide a short and concise description of the setting. If the town/city is stated and is not sunnydale in 1997 or later, that should be stated. Pre-1997 is a flashback.""")

    # 2. Provide a high-quality example to guide the model

    examples = [
        lx.data.ExampleData(
            text=example["text"],
            extractions=[
                lx.data.Extraction(
                    extraction_class="location",
                    extraction_text=example["extraction_text"],
                    attributes={
                        "location_descr": example["location_descr"],
                    },
                ),
            ],
        )
        for example in config["NER"]["location"]["examples"]
    ]

    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=examples,
        model_id="gemini-2.5-flash",
        api_key=API_KEY,
    )
    # Save the results to a JSONL file
    lx.io.save_annotated_documents(
        [result], output_name="extraction_results.jsonl", output_dir="."
    )

    # Generate the visualization from the file
    if generate_html:
        html_content = lx.visualize("extraction_results.jsonl")
        with open("visualization.html", "w") as f:
            if hasattr(html_content, "data"):
                f.write(html_content.data)
            else:
                f.write(html_content)
        # return html_content

    return result.extractions


if __name__ == "__main__":
    input_text = """
    Cut to the library. Willow has the city plans on the computer monitor.

BUFFY
There it is.

WILLOW
That runs under the graveyard.

XANDER
I don't see any access.

GILES
So, all the city plans are just, uh, open to the public?

WILLOW
Um, well, i-in a way. I sort of stumbled onto them when I accidentally decrypted the city council's security system.

XANDER
Someone's been naughty.

BUFFY
There's nothing here, this is useless!

GILES
I think you're being a bit hard on yourself.

BUFFY
You're the one that told me that I wasn't prepared enough. Understatement! (exhales) I thought I was on top of everything, and then that monster, Luke, came out of nowhere...

She flashes back to the fight in the mausoleum.

XANDER
What?

BUFFY
He didn't come out of nowhere. He came from behind me. I was facing the entrance, he came from behind me, and he didn't follow me out. The access to the tunnels is in the mausoleum! The girl must have doubled back with Jesse after I got out! God! I am so mentally challenged!

XANDER
So, what's the plan? We saddle up, right?

BUFFY
There's no 'we', okay? I'm the Slayer, and you're not.

XANDER
I knew you'd throw that back in my face.

BUFFY
Xander, this is deeply dangerous.

XANDER
I'm inadequate. That's fine. I'm less than a man.

WILLOW
Buffy, I'm not anxious to go into a dark place full of monsters. But I do want to help. I need to.

GILES
Well, then help me. I've been researching this Harvest affair. It seems to be some sort of preordained massacre. Rivers of blood, Hell on Earth, quite charmless. I'm a bit fuzzy, however, on the details. It may be that you can wrest some information from that dread machine.

Everyone stares at him. He looks back at them all.

GILES
That was a bit, um, British, wasn't it?

BUFFY
(smiles) Welcome to the New World.

GILES
(to Willow) I want you to go on the 'Net.

WILLOW
Oh, sure, I can do that. (begins to type)

BUFFY
Then I'm outta here. If Jesse's alive, I'll bring him back. (starts to leave)

GILES
Do I have to tell you to be careful?

Buffy turns back, gives Giles a look and goes. 
        
        """
    tag_text(input_text, generate_html=True)
