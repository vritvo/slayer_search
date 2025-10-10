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

Run the **embeddings**
```
uv run python -m setup.pipeline_runner
```

## Run the semantic search
```
uv run app.py
```