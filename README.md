# Buffy Search

Buffy Search is a lightweight search tool for Buffy the Vampire Slayer episode transcripts. It lets you scrape transcripts, build embeddings, and run quick semantic or literal searches — all with minimal setup.


# What This Is

This project scrapes Buffy the Vampire Slayer episode transcripts, processes them into structured text chunks, and then generates embeddings for semantic search. With those embeddings, you can query the transcripts in natural language and get back the most relevant lines or scenes.

The emphasis here is on simplicity:

You don’t have to think too much about preprocessing or chunking — it just works.

You can choose between semantic search (vector similarity) or simple literal/fuzzy search, depending on your needs.

All outputs are plain text, so you can easily read results in the terminal.


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
(this will take a while, and you can run it either with the "line" or scene" argument)
```
uv run embeddings.py
```

## Run a semantic search
```
python semantic_search.py
```
You'll be prompted for the thing you're searching for. 

## Run a literal search

For simple/fuzzy matching (without embeddings):

```
python literal_search.py --mode literal --search_string "lie to me"
```
you can also try fuzzy search with 
`--mode = fuzzy`, restrict to a single episode with `--episode` (e.g. `--episode 2x07 Lie to Me.txt`)
