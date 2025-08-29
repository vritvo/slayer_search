from bs4 import BeautifulSoup
import requests
import toml
import re


def get_episode_links() -> dict[str, str]:
    """Scrape episode links from Buffy transcripts page."""
    config = toml.load("config.toml")

    resp = requests.get(config["SAD_URL"], headers=config["HEADERS"])
    soup = BeautifulSoup(resp.text, "html.parser")
    buffy_container = soup.find("div", id="buffyEpBlock")

    episode_links = {}

    for season_node in buffy_container.children:
        if season_node.name == "div":
            # Extract season number from title div
            season_num = season_node.find("div", class_="title").get_text()
            season_num = season_num[-1]
            print(season_num)

            # Get all episode links for this season
            links = season_node.find_all("a")
            for episode_link in links:
                link = episode_link.get("href")
                link = link.split("?")[1] 
                
                name = episode_link.get_text()
                ep_num, title = name.split(" - ")
                ep_num = ep_num.zfill(2)  # Pad with leading zero

                # Format: "1x01 Welcome to the Hellmouth"
                formatted_title = f"{season_num}x{ep_num} {title}"
                formatted_link = config["SAD_URL"] + "?" + link
                episode_links[formatted_title] = formatted_link

    return episode_links


def standardize_dialogue_format(script: str) -> str:
    """Convert dialogue from 'Speaker: dialogue' to standardized format."""
    lines = script.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Match pattern: "Speaker: dialogue"
        match = re.match(r'^([A-Za-z][A-Za-z\s]*?):\s*(.+)$', line.strip())
        
        if match:
            speaker = match.group(1).strip().upper()
            dialogue = match.group(2).strip()
            
            cleaned_lines.append(speaker)
            if dialogue:
                cleaned_lines.append(dialogue)
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def get_ep_transcript(ep_url: str) -> str:
    """Download and parse episode transcript from URL."""
    config = toml.load("config.toml")

    resp = requests.get(ep_url, headers=config["HEADERS"])
    soup = BeautifulSoup(resp.text, "html.parser")
    script_div = soup.find("div", id="mainpage")

    # Replace <br> tags with newlines to preserve formatting
    for br in script_div.find_all("br"):
        br.replace_with("\n")

    script = script_div.get_text()
    script = standardize_dialogue_format(script)
    return script


def save_transcript(episode: str, script: str) -> None:
    config = toml.load("config.toml")
    script_folder = config["SCRIPT_FOLDER"]
    """Save episode transcript to file."""
    with open(f"{script_folder}/{episode}.txt", "w") as file:
        file.write(script)


def scraper() -> None:
    """Main scraper function to download all episode transcripts."""
    episode_links = get_episode_links()

    for episode, url in episode_links.items():
        print(f"Downloading: {episode}")
        script = get_ep_transcript(url)
        save_transcript(episode, script)


if __name__ == "__main__":
    scraper()
