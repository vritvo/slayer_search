import os
import toml
import anthropic
import re
import json
from dotenv import load_dotenv


def llm_scene_split(episode_name):
    """
    Analyzes a Buffy script to identify scene breaks using an LLM.
    """
    load_dotenv()

    API_KEY = os.getenv("ANTHROPIC_API_KEY")
    config = toml.load("config.toml")

    scene_split_markers_with_text_in_list = [
        f"- Line starts with '{x}'" for x in config["WINDOW"]["scene_split_markers"]
    ]
    scene_split_markers_in_text = "\n".join(scene_split_markers_with_text_in_list)

    ""
    with open(f"scripts/{episode_name}.txt", "r") as file:
        script_content = file.read()

    prompts = toml.load("prompts.toml")
    system_prompt = (
        prompts.get("SCENE_SPLIT")
        .get("system")
        .format(scene_split_markers_in_text=scene_split_markers_in_text)
    )
    user_prompt = (
        prompts.get("SCENE_SPLIT")
        .get("user")
        .format(episode_name=episode_name, script_content=script_content)
    )

    client = anthropic.Anthropic(api_key=API_KEY)
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    )

    result = message.content[0].text

    # Strip markdown code fences if present
    result = result.strip()
    if result.startswith("```"):
        result = re.sub(r"^```(?:json)?\s*\n", "", result)
        result = re.sub(r"\n```\s*$", "", result)

    # Parse and return JSON
    return json.loads(result)


def format_scene_splits(episode_list=None, start_file_fresh=True):
    """
    For each episode in the episode list, runs the llm_scene_split function. Adds the json data to a list.
    """
    config = toml.load("config.toml")

    if episode_list is None:
        episode_list = config["EPISODES_TO_LLM_SPLIT"]

    all_json_data = []
    for episode in episode_list:
        json_data = llm_scene_split(episode)
        all_json_data.append(json_data)

    # Create the scene_breakups directory if it doesn't exist
    os.makedirs("scene_splits", exist_ok=True)

    if start_file_fresh:
        # Start fresh with only the new data
        pass  # all_json_data already has the new data
    else:
        # Append to existing data
        with open(f"scene_splits/all_scene_splits.json", "r") as file:
            existing_data = json.load(file)
        all_json_data = existing_data + all_json_data  # Combine old + new

    with open(f"scene_splits/all_scene_splits.json", "w") as file:
        json.dump(all_json_data, file, indent=2)


if __name__ == "__main__":
    format_scene_splits(start_file_fresh=True)
