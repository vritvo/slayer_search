from setup.data_processor import llm_scene_split
import json
import re
import os

def format_scene_breakup():
    """ 
    Runs through a series of episodes. for each one, identifies the scene breaks. Adds the json formatted string output to a json file.
    
    """
    
    episode_list = [
        "4x01 The Freshman",
        "4x02 Living Conditions"
    ]

    all_json_data = []
    for episode in episode_list:
        result = llm_scene_split(episode)
        
        # Strip markdown code fences if present
        result = result.strip()
        if result.startswith("```"):
            result = re.sub(r'^```(?:json)?\s*\n', '', result)
            result = re.sub(r'\n```\s*$', '', result)
        
        json_data = json.loads(result)
        all_json_data.append(json_data)
    
    # Create the scene_breakups directory if it doesn't exist
    os.makedirs("scene_breakups", exist_ok=True)
    
    with open(f"scene_breakups/all_scene_breakups.json", "w") as file:
        json.dump(all_json_data, file, indent=2)
if __name__ == "__main__":
    format_scene_breakup()
