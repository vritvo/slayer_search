from utils.data_access import iter_scenes
import json
from utils.models import tag_text

chunk_num = 0

episode_list = [
    # "1x12 Prophecy Girl",
    # "2x22 Becoming, Part 2",
    # "3x01 Anne",
    # "4x05 Beer Bad",
    "5x07 Fool For Love",
    "7x02 Beneath You",
]

for scene_row in iter_scenes():
    if scene_row["file_name"] in episode_list:
        print(scene_row["file_name"])
        print(scene_row["text"])
        result = tag_text(scene_row["text"])

        extraction_list = []
        location_descr_list = []
        for extraction in result:
            extraction_list.append(extraction.extraction_text)

            print("----")
            print(extraction.attributes)
            if extraction.attributes and 'location_descr' in extraction.attributes:
                location_descr_list.append(extraction.attributes['location_descr'])

        scene_row["location_descr"] = ", ".join(location_descr_list)
        scene_row["location_text"] = ", ".join(extraction_list)

        # append to a json something with file_name, text, location_descr, and extraction_text:
        with open("lang_extract_results.json", "a") as f:
            f.write(
                json.dumps(
                    {
                        "file_name": scene_row["file_name"],
                        "text": scene_row["text"],
                        "location_descr": scene_row["location_descr"],
                        "extraction_text": scene_row["location_text"],
                    }
                )
                + "\n"
            )
        if chunk_num == 5:
            break
        chunk_num += 1
