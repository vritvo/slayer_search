import os
import argparse


def search_grep(lines, search_string):
    search_string = "grep -Ri -C {lns} {string} ./transcripts".format(
        string=search_string, lns=lines
    )
    os.system(search_string)


def ep_search_literal(
    search_string,
    episode="1x01 Welcome to the Hellmouth.txt",
):
    with open("transcripts/" + episode) as f:
        script = f.read()

    # print(script)

    for i, line in enumerate(script.splitlines()):
        if search_string in line:
            print(f"**************\nEPISODE: {episode}\n-------")
            print(line)
            print("\n")


def search_literal(search_string):
    for episode_path in os.scandir("transcripts/"):
        if episode_path.is_file():
            episode_path = episode_path.path

        _, episode = episode_path.split("/")

        ep_search_literal(search_string, episode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_string", "--ss")
    parser.add_argument("--lns", default="3")
    args = parser.parse_args()

    # search_grep(args.lns, args.search_string)
    search_literal(args.search_string)
