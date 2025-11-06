import anthropic
from dotenv import load_dotenv
import os
import toml


def run_model():
    load_dotenv()

    API_KEY = os.getenv("ANTHROPIC_API_KEY")
    config = toml.load("config.toml")
    episode_name = "4x01 The Freshman"
    # f'{config["WINDOW"]["scene_split_markers"]}'

    scene_split_markers_with_text_in_list = [
        f"- Line starts with '{x}'" for x in config["WINDOW"]["scene_split_markers"]
    ]
    scene_split_markers_in_text = "\n".join(scene_split_markers_with_text_in_list)

    ""
    with open(f"scripts/{episode_name}.txt", "r") as file:
        script_content = file.read()
        
    prompts = toml.load("prompts.toml")
    system_prompt = prompts.get("SCENE_SPLIT").get("system").format(scene_split_markers_in_text=scene_split_markers_in_text)
    user_prompt = prompts.get("SCENE_SPLIT").get("user").format(episode_name=episode_name, script_content=script_content)

    # client = anthropic.Anthropic(api_key=API_KEY)
    # message = client.messages.create(
    #     model="claude-sonnet-4-5",
    #     max_tokens=1000,
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": "What should I search for to find the latest developments in renewable energy?",
    #         }
    #     ],
    # )
    # print(message.content)

    print(system_prompt)
    print(user_prompt)


if __name__ == "__main__":
    run_model()
