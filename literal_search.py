import os
import argparse
import rapidfuzz
import numpy as np
import toml


def search_grep(lines: str, search_string: str) -> None:
    """Execute grep search with context lines."""
    config = toml.load("config.toml")
    script_folder = config["SCRIPT_FOLDER"]
    grep_cmd = f"grep -Ri -C {lines} {search_string} {script_folder}"
    os.system(grep_cmd)


def ep_search_literal(
    search_string: str,
    episode: str = "1x01 Welcome to the Hellmouth.txt",
) -> None:
    """Search for literal string matches in a specific episode."""
    config = toml.load("config.toml")
    script_folder = config["SCRIPT_FOLDER"]
    
    with open(f"{script_folder}/{episode}") as f:
        script = f.read()

    for line in script.splitlines():
        if search_string in line:
            print(f"**************\nEPISODE: {episode}\n-------")
            print(line)
            print("\n")


def search_literal(search_string: str) -> None:
    """Search for literal string matches across all episodes."""
    config = toml.load("config.toml")
    script_folder = config["SCRIPT_FOLDER"]
    
    for episode_path in os.scandir(script_folder):
        if episode_path.is_file():
            episode_path = episode_path.path

        _, episode = episode_path.split("/")
        ep_search_literal(search_string, episode)


def ep_search_fuzz(search_string: str, episode: str = "2x07 Lie to Me.txt", N: int = 5) -> None:
    """Search for fuzzy string matches in a specific episode using rapidfuzz."""
    config = toml.load("config.toml")
    script_folder = config["SCRIPT_FOLDER"]
    
    with open(f"{script_folder}/{episode}") as f:
        script = f.read()

    script_lines = script.splitlines()

    # Calculate fuzzy match scores for all lines
    fuzzy_scores = rapidfuzz.process.cdist(
        [search_string], script_lines, scorer=rapidfuzz.fuzz.partial_ratio
    )
    sorted_indices = np.argsort(fuzzy_scores)[0]
    
    # Get top N highest scoring matches
    top_indices = sorted_indices[-N:][::-1]
    top_scores = fuzzy_scores[0][top_indices]

    # Only print if any scores meet the cutoff threshold
    if any(score >= config["SEARCH"]["fuzzy_cutoff"] for score in top_scores):
        print(f"\n**************\nEPISODE: {episode}\n-------")
        
    for idx, score in zip(top_indices, top_scores):
        if score >= config["SEARCH"]["fuzzy_cutoff"]:
            print(f"Score: {score:.2f} | {script_lines[idx]}")


def search_fuzzy(search_string: str) -> None:
    """Search for fuzzy string matches across all episodes."""
    config = toml.load("config.toml")
    script_folder = config["SCRIPT_FOLDER"]
    
    for episode_path in os.scandir(script_folder):
        if episode_path.is_file():
            episode_path = episode_path.path

        _, episode = episode_path.split("/")
        ep_search_fuzz(search_string, episode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search Buffy episode transcripts")
    parser.add_argument("--search_string", "--s", default="lie to me", help="Text to search for")
    parser.add_argument("--lns", "--l", default="3", help="Context lines for grep")
    parser.add_argument("--mode", "--m", default="fuzzy", choices=["fuzzy", "literal"], 
                       help="Search mode")
    parser.add_argument("--episode", "--e", help="Specific episode to search")
    args = parser.parse_args()

    # TODO: implement context lines for fuzzy search + literal search

    if args.mode == "fuzzy":
        if args.episode:
            ep_search_fuzz(args.search_string, args.episode)
        else:
            search_fuzzy(args.search_string)
    elif args.mode == "literal":
        if args.episode:
            ep_search_literal(args.search_string, args.episode)
        else:
            search_literal(args.search_string)
