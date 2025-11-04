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
input_text = """Cut to Hemery High School in Los Angeles, 1996. School is over for the day, and the students come streaming out. An old, rusted Chevy Impala with its windows spray-painted black pulls up on the far side of the street. The driver's window lowers, and Angel squints out into the daylight, careful to remain in shadow. He looks over at the building and sees Buffy come down the steps with three of her friends. """

prompt = textwrap.dedent("""\
    Extract location.
    Use exact text for extractions. Do not paraphrase or overlap entities. If the location is not certain, return n/a.
    For the attribute location_descr, provide a short and concise description of the setting. If the town/city is stated and is not sunnydale in 1997 or later, that should be stated""")

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


# A town square in Galway, Ireland, 1753. The camera looks straight down from above onto the cobblestones. A lone rider on his horse passes underneath, and the camera follows them past a well as Angelus narrates.
