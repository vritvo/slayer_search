import langextract as lx
import textwrap
import os
import toml
import json
from dotenv import load_dotenv

def tag_text(input_text, generate_html=False):
    
    # Make sure there is an input text, otherwise error: 
    if not input_text:
        raise ValueError("No input text provided")
    
    # Load environment variables from .env file
    load_dotenv()
    API_KEY = os.getenv("LANGEXTRACT_API_KEY")
    config = toml.load("config.toml")

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
    # Save the results to a JSONL file
    lx.io.save_annotated_documents( [result], output_name="extraction_results.jsonl", output_dir=".")
    
    
    # # Append result to JSONL file manually
    # with open("extraction_results.jsonl", "a") as f:
    #     # Convert the AnnotatedDocument to a dict-like structure for JSON serialization
    #     result_data = {
    #         "text": result.text,
    #         "extractions": [
    #             {
    #                 "extraction_class": ext.extraction_class,
    #                 "extraction_text": ext.extraction_text,
    #                 "char_interval": {
    #                     "start_pos": ext.char_interval.start_pos,
    #                     "end_pos": ext.char_interval.end_pos
    #                 },
    #                 "alignment_status": ext.alignment_status.value if hasattr(ext.alignment_status, 'value') else str(ext.alignment_status),
    #                 "extraction_index": ext.extraction_index,
    #                 "group_index": ext.group_index,
    #                 "description": ext.description,
    #                 "attributes": ext.attributes
    #             }
    #             for ext in result.extractions
    #         ]
    #     }
    #     f.write(json.dumps(result_data) + "\n")
    
    print(result.extractions)

    # Generate the visualization from the file
    if generate_html:
        html_content = lx.visualize("extraction_results.jsonl")
        with open("visualization.html", "w") as f:
            if hasattr(html_content, "data"):
                f.write(html_content.data)
            else:
                f.write(html_content)
        # return html_content

    print(result.extractions)
    return result.extractions


if __name__ == "__main__":
    input_text = """Cut to Hemery High School in Los Angeles, 1996. School is over for the day, and the students come streaming out. An old, rusted Chevy Impala with its windows spray-painted black pulls up on the far side of the street. The driver's window lowers, and Angel squints out into the daylight, careful to remain in shadow. He looks over at the building and sees Buffy come down the steps with three of her friends. """
    tag_text(input_text = input_text)