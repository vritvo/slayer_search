import anthropic
from dotenv import load_dotenv
import os


def run_model():
    load_dotenv()

    API_KEY = os.getenv("ANTHROPIC_API_KEY")

    episode_name = "4x01 The Freshman"

    # with open(f"scripts/{script}.txt", "r") as file:
    #     content = file.read

    script_content = "this is a test"

    prompt = f"""    
    You will be analyzing a Buffy the Vampire Slayer script to identify scene breaks that are not already clearly marked in the script.

Here is the script to analyze:

<episode_name>
{episode_name}
</episode_name>
<script>
{script_content}
</script>

Your task is to identify where new scenes begin in the script. A new scene occurs when there is a change in either location or time that can be inferred from stage directions or narrative text.

Important rules:

1. DO NOT identify scene breaks that are already marked by any of these existing identifiers:
   - Lines starting with "Cut to"
   - Lines starting with "~~~~" 
   - Lines starting with "=="

2. DO identify scene breaks indicated by:
   - Stage directions showing a change of location (e.g., "Later, Buffy and Willow are indoors, walking along a hallway")
   - Narrative text indicating a time jump (e.g., "The next morning", "Meanwhile, at the Bronze")
   - Descriptive text establishing a new setting (e.g., "Buffy enters the library", "At Giles' apartment")

Before providing your final answer, use the scratchpad below to work through the script systematically:

<scratchpad>
- First, identify the episode title
- Then, go through the script line by line looking for stage directions or narrative text that indicates a change in location or time
- For each potential scene break, note the exact first line of the new scene
- Double-check that none of these are already marked with the existing identifiers mentioned above
</scratchpad>

Output your response as a JSON object with this exact format:

```json
{{
        "episode_title": "[episode title here]",
  "scene_breaks": ["first line of scene 1", "first line of scene 2", etc.]
}}
```

If there are no new scene breaks to identify (because they are all already marked), return an empty array for scene_breaks.


    EXAMPLE 1: 

    
    "The Freshman TranscriptBack to Transcript LibraryReport Transcript Error


    WRITTEN BY
    Joss Whedon
    DIRECTED BY
    Joss Whedon

    This episode was originally broadcast on October 5, 1999.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    It's night in the cemetery and Buffy is pacing back and forth. Willow is seated cross-legged reading papers.

    BUFFY
    [sighing] Anything?

    WILLOW
    Ah! 'Introduction to the Modern Novel.' "A survey study of twentieth century novelists." Open to freshmen, you might like that.
    ....
    
    OUTPUT 1
    (nothing should be output)
    
    ----------------------------------------------------------------------
    EXAMPLE 2: 
    
    PAUL
    Finally matriculating with us, very cool! Tell me you're playing this week!

    OZ
    Thursday night, Alpha Delta.

    WILLOW
    Ooh! (She holds up a flier.) I have that one!

    PAUL
    I'm bringing the wrecking crew. Jello shots? Hmm? Do you know where they're distributing the work study applications?

    OZ
    (Points.) Back of Richmond Hall, next to the auditorium.

    PAUL
    Thanks. Seeya bro. (He walks off.)

    OZ
    Go get'em. (He remembers what Buffy was talking about.) My band's played here a lot. It's still all new. I don't know what the hell's going on. (He sees someone.) Hey, Doug!

    Later, Buffy and Willow are indoors, walking along a hallway.

    WILLOW
    Library... ooh! Library. C'mon. (They start climbing a flight of stairs.)

    BUFFY
    It's too bad Giles can't be librarian here. Be convenient.

    They reach a landing and turn left to continue up another flight.

    WILLOW
    Well, he says that he's enjoying being a gentleman of leisure.

    BUFFY
    Gentleman of leisure? Isn't that just british for unemployed?

    OUTPUT 2:
    
    {{"episode": "4x01 The Freshman", "scene_breaks": ["Later, Buffy and Willow are indoors, walking along a hallway."]}}

    """

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

    print(prompt)


if __name__ == "__main__":
    run_model()
