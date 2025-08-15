from bs4 import BeautifulSoup
import requests
import toml
from config import *


def get_episode_links():
    config = toml.load("config.toml")

    resp = requests.get(config.FOREVER_DREAMING_URL, headers=config.HEADERS)
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


def get_ep_transcript(test_ep_url):
    pass

    test_ep_url
    resp = requests.get(test_ep_url, headers=config.HEADERS)
    soup = BeautifulSoup(resp.text, "html.parser")
    script = soup.find("div", class_="content")
    return script.get_text()


def main():
    episode_links = get_episode_links()

    print(episode_links)
    test_ep_url = episode_links["Welcome to the Hellmouth"]


if __name__ == "__main__":
    main()
