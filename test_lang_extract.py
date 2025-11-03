import langextract as lx
import textwrap
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("LANGEXTRACT_API_KEY")

print(API_KEY)
input_text = """Cut to the Summers house, foyer, night. Buffy enters through the front door, followed by Sam and Riley.

BUFFY
Sorry the place is such a mess. I haven't had a chance to give it a good cleaning.
"""
# input_text = """CUT TO
# close shot of the diamond sitting on a piece of black velvet.

# JONATHAN
# I didn't know it'd be so sparkly.

# ANDREW
# It's so big.

# WARREN
# Yes, gentlemen, it turns out, size is everything. (puts hand on Jonathan's leg) No offense, man.

# Jonathan smacks him. We see that they're in the basement lair, sitting and looking at the diamond on a card table.

# ANDREW
# It makes colors with the light.

# The others stare at him for a moment.

# WARREN
# All right, I think we've finished the first part, now it's time for Phase Two.
# """
# 1. Define the prompt and extraction rules
prompt = textwrap.dedent("""\
    Extract location.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.""")

# 2. Provide a high-quality example to guide the model
examples = [
    lx.data.ExampleData(
        text="""Cut to the Summers house. Cut inside. Joyce is writing out a few bills. She hears a knocking at the door, and looks up. She goes over to the door and answers it. She is surprised at who she sees standing there. """,
        extractions=[
            lx.data.Extraction(
                extraction_class="location",
                extraction_text="Inside Buffy Summer's house",
                attributes={"location_type": "indoors"},
            ),
        ],
    ),
    lx.data.ExampleData(
        text="""~~~~~~~~~~ Part 1 ~~~~~~~~~~""",
        extractions=[
            lx.data.Extraction(
                extraction_class="location",
                extraction_text="n/a",
                attributes={"location_type": "n/a"},
            ),
        ],
    ),
    lx.data.ExampleData(
        text="""~~~~~~~~~~ Part 1 ~~~~~~~~~~
    Sunnydale High School. Cut to the hall. Cordelia is demonstrating her fake laugh to another girl.
    CORDELIA
    (fake laughter) See? Dr. Debi says when a man is speaking you make serious eye contact, and you really, really listen, and you laugh at everything he says. (laughs again)
    """,
        extractions=[
            lx.data.Extraction(
                extraction_class="location",
                extraction_text="Sunnydale High School Hall",
                attributes={"location_type": "indoors"},
            ),
        ],
    ),
    lx.data.ExampleData(
        text="""Outside Angelus' mansion. The camera pans along its dark facade. Cut to the street. Buffy walks toward the mansion at a determined pace with the sword wrapped in a cloth. Suddenly Xander comes running out of the bushes on the hillside and jumps into the street in front of her. She startles and takes a reflexive step back.""",
        extractions=[
            lx.data.Extraction(
                extraction_class="location",
                extraction_text="Street outside Angelus's mansion",
                attributes={"location_type": "outdoors"},
            ),
        ],
    ),
]


# ~~~~~~~~~~ Part 2 ~~~~~~~~~~

#  A wood in Rumania, 1898. Angelus runs through the trees, panting in his desperation to reach a gypsy camp. Cut to the camp. The camera pans across the dead body of the young Kalderash Gypsy girl that Angelus has recently killed. She is on a table dressed in white and lying on an intricately patterned quilt with candles burning around the perimeter. Members of the clan are laying rose petals on her. The camera continues to pan over to the Elder Woman sitting beneath a tent canopy and chanting over an Orb of Thesulah surrounded by candles within a sacred circle. Angelus continues running through the woods as she chants.

# ELDER WOMAN
# Nici mort, nici de-al fiintei, Te invoc, spirit al trecerii. Reda trupului ce separa omul de animal!


# In every generation there is a Chosen One. She alone will stand against the vampires, the demons and the forces of darkness. She is the Slayer.

# Sunnydale High School at night. A man in a suit with a briefcase is walking past a school building at a brisk, determined pace. He stops for a moment and looks around. Behind him to his left a door opens, and a school custodian comes out with a trashcan.


# ----

# Cut to a section of floor with candles set out and symbols written on the floor in red dirt. A pair of hands throw runestones onto the symbols. The camera pulls out and we see we're in Glory's apartment. Gronx and Murk sit on the floor casting the runes.

# GRONX
# It's coming. The signs are in alignment, and soon victory will be in our grasp. (they smile) All we need do is seize the moment ... and squeeze until it bleeds.

# They both smile happily.

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
