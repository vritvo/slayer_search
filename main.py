import os
import argparse
import rapidfuzz
import numpy as np
import toml


def search_grep(lines, search_string):
    search_string = "grep -Ri -C {lns} {string} ./scripts".format(
        string=search_string, lns=lines
    )
    os.system(search_string)


def ep_search_literal(
    search_string,
    episode="1x01 Welcome to the Hellmouth.txt",
):
    with open("scripts/" + episode) as f:
        script = f.read()

    for i, line in enumerate(script.splitlines()):
        if search_string in line:
            print(f"**************\nEPISODE: {episode}\n-------")
            print(line)
            print("\n")


def search_literal(search_string):
    for episode_path in os.scandir("scripts/"):
        if episode_path.is_file():
            episode_path = episode_path.path

        _, episode = episode_path.split("/")

        ep_search_literal(search_string, episode)


def ep_search_fuzz(search_string, episode="2x07 Lie to Me.txt", N=5):
    config = toml.load("config.toml")
    with open("scripts/" + episode) as f:
        script = f.read()

    splitlines = script.splitlines()

    # TODO add check to make sure input only has 1 value
    fuzzy_sorted = rapidfuzz.process.cdist(
        [search_string], splitlines, scorer=rapidfuzz.fuzz.partial_ratio
    )
    sorted_indices = np.argsort(fuzzy_sorted)[0]
    # Take last N (highest values), reverse them
    top_indices = sorted_indices[-N:][::-1]

    # Get values and print lines with scores
    top_values = fuzzy_sorted[0][top_indices]

    if any(score >= config["SEARCH"]["fuzzy_cutoff"] for score in top_values):
        print("\n")
        print(f"**************\nEPISODE: {episode}\n-------")
    for idx, score in zip(top_indices, top_values):
        if score >= config["SEARCH"]["fuzzy_cutoff"]:
            print(f"Score: {score:.2f} | {splitlines[idx]}")


def search_fuzzy(search_string):
    for episode_path in os.scandir("scripts/"):
        if episode_path.is_file():
            episode_path = episode_path.path

        _, episode = episode_path.split("/")

        ep_search_fuzz(search_string, episode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_string", "--s", default="lie to me")
    parser.add_argument("--lns", "--l", default="3")
    parser.add_argument("--mode", "--m", default="fuzzy", choices=["fuzzy", "literal"])
    parser.add_argument("--episode", "--e")
    args = parser.parse_args()

    # TODO implement context lines for fuzzy search + literal search.

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
