import hashlib
from utils.data_access import iter_scenes
import json
from utils.models import tag_text

chunk_num = 0

episode_list = [
    "1x12 Prophecy Girl",
    # "2x22 Becoming, Part 2",
    # "3x01 Anne",
    # "4x05 Beer Bad",
    "5x07 Fool For Love",
    "7x02 Beneath You",
    # "4x22 Restless",
]

# Collect all results in a list
results = []
failed_scenes = []

for scene_row in iter_scenes():
    if scene_row["file_name"] in episode_list:
        print(scene_row["file_name"])
        print(scene_row["text"])

        # Make hash
        scene_bytes = scene_row["text"].encode("utf-8")
        hash_object = hashlib.sha256(scene_bytes)
        hash_hex = hash_object.hexdigest()
        scene_row["hash"] = hash_hex
        
        # create location tagging for each scene
        try:
            result = tag_text(scene_row["text"], generate_html=False)

            extraction_list = []
            location_descr_list = []

            # Each scene has multiple location extractions, so we need to join them together
            for extraction in result:
                extraction_list.append(extraction.extraction_text)

                print(extraction.attributes)
                if extraction.attributes and "location_descr" in extraction.attributes:
                    location_descr_list.append(extraction.attributes["location_descr"])

            scene_row["location_descr"] = " | ".join(location_descr_list)
            scene_row["location_text"] = " | ".join(extraction_list)

            # Collect the result
            results.append(
                {
                    "scene_id": scene_row["scene_id"],
                    "scene_hash": scene_row["hash"],
                    "file_name": scene_row["file_name"],
                    "text": scene_row["text"],
                    "location_descr": scene_row["location_descr"],
                    "extraction_text": scene_row["location_text"],
                }
            )
            print(f"✓ Successfully processed scene {scene_row['scene_id']}\n")
            
        except Exception as e:
            print(f"\n✗ FAILED to process scene {scene_row['scene_id']}")
            print(f"  Error: {type(e).__name__}: {str(e)}")
            failed_scenes.append({
                "scene_id": scene_row["scene_id"],
                "scene_hash": scene_row["hash"],
                "file_name": scene_row["file_name"],
                "text": scene_row["text"][:500],  # First 500 chars for debugging
                "error": str(e)
            })
            print(f"  Continuing to next scene...\n")
            continue

        chunk_num += 1

# Write all results as a properly formatted JSON array
with open("lang_extract_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Write failed scenes to a separate file
if failed_scenes:
    with open("lang_extract_failed.json", "w") as f:
        json.dump(failed_scenes, f, indent=2)
