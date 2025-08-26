import os
import argparse


def search_grep(lines, search_string):
    search_string = "grep -Ri -C {lns} {string} ./transcripts".format(
        string=search_string, lns=lines
    )
    os.system(search_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_string")
    parser.add_argument("--lns", default="3")
    args = parser.parse_args()

    search_grep(args.lns, args.search_string)
