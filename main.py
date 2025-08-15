from bs4 import BeautifulSoup
import requests
from config import *

# TODO: set up config toml


def get_episode_links():
    # FOREVER_DREAMING_URL = "https://transcripts.foreverdreaming.org/viewtopic.php?t=8296&sid=943ebb32f0511dfed8eaf1a1e89614ec"
    # headers = {
    #     "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
    # }

    resp = requests.get(FOREVER_DREAMING_URL, headers=HEADERS)
    soup = BeautifulSoup(resp.text, "html.parser")

    episode_list_container = soup.find("div", class_="content")
    episode_list_container

    episode_links = {}
    episode_list_container.find_all("a")

    for link in episode_list_container.find_all("a"):
        episode_name = link.get_text(strip=True)
        if episode_name:
            episode_links[episode_name] = link.get("href")

        # TODO: need to get previous text which has episode number.

    return episode_links


def get_ep_transcript(test_ep_URL):
    pass


def main():
    episode_links = get_episode_links()

    print(episode_links)
    test_ep_URL = episode_links["Welcome to the Hellmouth"]


if __name__ == "__main__":
    main()
