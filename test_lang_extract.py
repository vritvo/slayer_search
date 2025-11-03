import langextract as lx
import textwrap
import os
import toml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("LANGEXTRACT_API_KEY")
config = toml.load("config.toml")

print(API_KEY)
input_text = """ A town square in Galway, Ireland, 1753. The camera looks straight down from above onto the cobblestones. A lone rider on his horse passes underneath, and the camera follows them past a well as Angelus narrates. 
"""

prompt = textwrap.dedent("""\
    Extract location.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.""")

# 2. Provide a high-quality example to guide the model

examples = [
    lx.data.ExampleData(
        text=example["text"],
        extractions=[
            lx.data.Extraction(
                extraction_class="location",
                extraction_text=example["extraction_text"],
                attributes={
                    "location_descr": example["location_descr"],
                },
            ),
        ],
    )
    for example in config["NER"]["location"]["examples"]
]
# examples = [
#     lx.data.ExampleData(
#         text="""Cut to the Summers house. Cut inside. Joyce is writing out a few bills. She hears a knocking at the door, and looks up. She goes over to the door and answers it. She is surprised at who she sees standing there. """,
#         extractions=[
#             lx.data.Extraction(
#                 extraction_class="location",
#                 extraction_text="Inside Buffy Summer's house",
#                 attributes={"location_type": "indoors"},
#             ),
#         ],
#     ),
#     lx.data.ExampleData(
#         text="""~~~~~~~~~~ Part 1 ~~~~~~~~~~""",
#         extractions=[
#             lx.data.Extraction(
#                 extraction_class="location",
#                 extraction_text="n/a",
#                 attributes={"location_type": "n/a"},
#             ),
#         ],
#     ),
#     lx.data.ExampleData(
#         text="""~~~~~~~~~~ Part 1 ~~~~~~~~~~
#     Sunnydale High School. Cut to the hall. Cordelia is demonstrating her fake laugh to another girl.
#     CORDELIA
#     (fake laughter) See? Dr. Debi says when a man is speaking you make serious eye contact, and you really, really listen, and you laugh at everything he says. (laughs again)
#     """,
#         extractions=[
#             lx.data.Extraction(
#                 extraction_class="location",
#                 extraction_text="Sunnydale High School Hall",
#                 attributes={"location_type": "indoors"},
#             ),
#         ],
#     ),
#     lx.data.ExampleData(
#         text="""Outside Angelus' mansion. The camera pans along its dark facade. Cut to the street. Buffy walks toward the mansion at a determined pace with the sword wrapped in a cloth. Suddenly Xander comes running out of the bushes on the hillside and jumps into the street in front of her. She startles and takes a reflexive step back.""",
#         extractions=[
#             lx.data.Extraction(
#                 extraction_class="location",
#                 extraction_text="Street outside Angelus's mansion",
#                 attributes={"location_type": "outdoors"},
#             ),
#         ],
#     ),
# ]


print(os.getenv("LANGEXTRACT_API_KEY"))
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    api_key=API_KEY,
)
print("---")
print(result.extractions)
print(result)

# Save the results to a JSONL file
lx.io.save_annotated_documents(
    [result], output_name="extraction_results.jsonl", output_dir="."
)

# Generate the visualization from the file
html_content = lx.visualize("extraction_results.jsonl")
with open("visualization.html", "w") as f:
    if hasattr(html_content, "data"):
        f.write(html_content.data)  # For Jupyter/Colab
    else:
        f.write(html_content)
