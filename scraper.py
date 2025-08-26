from bs4 import BeautifulSoup, NavigableString
import requests
import toml


def get_episode_links():
    config = toml.load("config.toml")

    resp = requests.get(config["FOREVER_DREAMING_URL"], headers=config["HEADERS"])
    soup = BeautifulSoup(resp.text, "html.parser")

    episode_list_container = soup.find("div", class_="content")
    episode_list_container

    episode_links = {}
    episode_list_container.find_all("a")

    episode_number = ""
    for node in episode_list_container.children:
        if isinstance(node, NavigableString):
            episode_number = node.strip()

        elif node.name == "a":
            episode_title = node.get_text(strip=True).replace("k*ll", "Kill")
            episode_link = node.get("href")

            if episode_title in [
                "Buffy episode scripts",
                "Read our Copyrights",
                "The Pilot Script download",
            ]:
                continue

            formatted_title = episode_number[:4] + " " + episode_title
            episode_links[formatted_title] = episode_link

    return episode_links


def get_ep_transcript(ep_url):
    str_replacements = toml.load("script_changes.toml")
    config = toml.load("config.toml")

    resp = requests.get(ep_url, headers=config["HEADERS"])
    soup = BeautifulSoup(resp.text, "html.parser")
    script = soup.find("div", class_="content").get_text()

    for k, v in str_replacements["STRINGS"].items():
        script = script.replace(k, v)

    return script


def save_transcript(episode, script):
    with open("./transcripts/" + episode + ".txt", "w") as file:
        file.write(script)


def scraper():
    episode_links = get_episode_links()

    for episode in episode_links.keys():
        print(episode)
        script = get_ep_transcript(episode_links[episode])
        save_transcript(episode, script)


if __name__ == "__main__":
    scraper()
