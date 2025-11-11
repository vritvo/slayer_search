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
    (Buffy leaning through the curtains to grab her.)

    WILLOW
    Buffy! Oh god.

    BUFFY
    Come on. (Helps her up and through the curtain. They're in a

    Sunnydale High classroom.)

    BUFFY
    Stay low. (They crouch down and creep between the desks) What did it look like?

    WILLOW
    I don't know. I-I don't know what's after me.

    BUFFY
    Well, you must have *done* something. (Frowning in disapproval)

    WILLOW
    No. I never do anything. I'm very seldom naughty. I, I just came to class, and, and the play was starting.

    BUFFY
    (straightens up) Play is long over. (Stares at Willow) Why are you still in costume?

    WILLOW
    Okay, still having to explain wherein this is just my outfit.

    (Gesturing to her clothes)

    BUFFY
    Willow, everybody already knows. Take it off.

    WILLOW
    No. No. (Looks around nervously) I need it.

    (Buffy rolls her eyes.)

    BUFFY
    Oh, for god's sake, just take it off.

    (Spins Willow around and rips her clothes off.)

    BUFFY
    That's better. It's much more realistic.

    (Suddenly all the desks have students in them. Buffy turns and goes to take her seat.)

    HARMONY
    See? Isn't everybody very clear on this now?

    (We see Anya sitting next to Harmony, giggling. The whole class is giggling.)

    (Shot of Willow in her nerdy schoolgirl outfit and long straight hair from

    BTVS first season. Holding some paper.)

    ANYA
    My god, it's like a tragedy.

    (Shot of Buffy looking at Willow.)

    OZ
    (to Tara) I tried to warn you. (Gives Willow a disgusted look)

    ANYA
    (still giggling) It's exactly like a Greek tragedy. There should only be Greeks.

    (Willow looks around the room nervously, looks down at her paper.)

    WILLOW
    (licks lips) My book report. This summer I, I read "The Lion, the

    Witch and the Wardrobe."

    XANDER
    (loudly, to ceiling) Oh, who cares?

    (Willow looks hurt. Sound of giggling. shot of Oz nuzzling Tara's cheek while she giggles.)

    WILLOW
    This book ha-has many themes...

    (Something bursts onscreen and knocks Willow down. She screams.)

    (Shot of Buffy putting her head down on her arms on the desk, looking bored. Sound of Willow screaming and the attacker growling.)

    WILLOW
    Help! Help me!

    (Shot of Xander looking bored.)

    (Shot of Oz and Tara giving each other conspiratorial smiles.)

    WILLOW
    Help me!

    (Growling noise continues as Willow struggles. The creature/person attacking Willow has dark skin and long matted dark hair, and is wrapped in rags. It bends as if to bite her neck. Closeup of Willow's face with the dark hair half-obscuring it. Her eyes widen. The skin on her face wrinkles and her eyes cloud.)
        
        
        """
    tag_text(input_text, generate_html=True)
