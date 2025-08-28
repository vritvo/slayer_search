from bs4 import BeautifulSoup
import requests
import toml
import re

def get_episode_links():
    config = toml.load("config.toml")

    resp = requests.get(config["SAD_URL"], headers=config["HEADERS"])
    soup = BeautifulSoup(resp.text, "html.parser")
    buffy_container = soup.find("div", id="buffyEpBlock")

    episode_links = {}

    for season_node in buffy_container.children:
        if season_node.name == "div":
            print(season_node)
            season_num = season_node.find("div", class_="title").get_text()
            season_num = season_num[-1]
            print(season_num)

            links = season_node.find_all("a")
            for e in links:
                link = e.get("href")
                print(link)
                link = link.split("?")[1]
                print(link)
                name = e.get_text()
                ep_num, title = name.split(" - ")
                ep_num = ep_num.zfill(2)

                formatted_title = "{season_num}x{ep_num} {title}".format(
                    season_num=season_num, ep_num=ep_num, title=title
                )
                formatted_link = config["SAD_URL"] + "?" + link
                episode_links[formatted_title] = formatted_link

    return episode_links


def standardize_dialogue_format(script):
    """
    Convert dialogue from 'Speaker: dialogue' format to standardized format:
    SPEAKER
    dialogue
    """
    
    lines = script.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Look for pattern like "Speaker: dialogue" where Speaker doesn't contain spaces (usually)
        # This regex matches a speaker name followed by colon and dialogue
        match = re.match(r'^([A-Za-z][A-Za-z\s]*?):\s*(.+)$', line.strip())
        
        if match:
            speaker = match.group(1).strip().upper()
            dialogue = match.group(2).strip()
            
            # Add speaker on one line, dialogue on next
            cleaned_lines.append(speaker)
            if dialogue:  # Only add dialogue line if there's actual dialogue
                cleaned_lines.append(dialogue)
        else:
            # Keep the line as-is if it doesn't match the speaker pattern
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def get_ep_transcript(ep_url):
    config = toml.load("config.toml")

    resp = requests.get(ep_url, headers=config["HEADERS"])
    soup = BeautifulSoup(resp.text, "html.parser")
    script_div = soup.find("div", id="mainpage")

    # Replace the <br>s with "/"s. We don't use the get_text separator because the scripts have a distinction
    # between a single and double break.
    for br in script_div.find_all("br"):
        _ = br.replace_with("\n")

    script = script_div.get_text()
    
    # Standardize the dialogue format
    script = standardize_dialogue_format(script)

    return script


def save_transcript(episode, script):
    with open("./scripts/" + episode + ".txt", "w") as file:
        file.write(script)


def scraper():
    episode_links = get_episode_links()

    for episode in episode_links.keys():
        print(episode)
        script = get_ep_transcript(episode_links[episode])
        save_transcript(episode, script)


if __name__ == "__main__":
    scraper()
