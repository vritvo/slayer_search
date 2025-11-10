# Slayer Search

Slayer Search is a search tool for Buffy the Vampire Slayer episode transcripts. It lets you scrape transcripts, build embeddings, and run a semantic search over scenes from the show.


# Usage

## One time setup:
Setup: 
```
uv sync
```

Run the scraper to download transcripts into ./scripts/

Run the **scraper**
```
uv run scraper.py
```
Note - Some episodes here do not have well designated scene splits. In that case, you may want to do a one-time run of the following to have an LLM identify where the scene breaks are. You'd need to have `ANTHROPIC_API_KEY` set in your .env. The output of this is already in this repo though, in `scene_splits/all_scene_splits.json`

```
uv run python -m setup.scene_splitter
```

Build the **vector database**
```
uv run python -m setup.pipeline_runner
```

## Run the semantic search
```
uv run app.py
```