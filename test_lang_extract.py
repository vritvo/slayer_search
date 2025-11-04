from utils.models import tag_text

if __name__ == "__main__":
    input_text = """
    Cut to the Summers house. Willow walks from the kitchen, holding a glass of water.

BUFFY
(OS) I've been having these flashes. Hallucinations, I guess.

Willow comes into the living room where Buffy is sitting in the armchair. Willow gives Buffy the water.

WILLOW
Since when?

BUFFY
Uh ... night before last.

Willow sits on the sofa, where Xander and Dawn are already sitting looking at Buffy.

BUFFY
I was, uh, checking houses on that list you gave me, and looking for Warren and his pals ... and then, bam! Some kind of gross, waxy demon-thing poked me.

XANDER
And when you say poke...

BUFFY
(rolling her eyes) In the arm. (Xander and Willow exchanging a look) It stung me or something, and ... then I was like ... no. It, it wasn't "like." I *was* in an institution. There were, um ... doctors and ... nurses and, and other patients. They, they told me that I was sick. I guess crazy. And that, um, Sunnydale and, and all of this, it ... none of it ... was real.

XANDER
Oh, come on, that's ridiculous! What? You think this isn't real just because of all the vampires and demons and ex-vengeance demons and the sister that used to be a big ball of universe-destroying energy? (pauses, frowns)

BUFFY
I know how this must sound, but ... it felt so real. (softly) Mom was there.

DAWN
She was?

BUFFY
Dad, too. They were together ... (distantly) like they used to be ... before Sunnydale.

WILLOW
(stands up hastily, raises her hand) Okay! All in favor of research? (Xander raises his hand) Motion passed. All right, Xander, you hit the demon bars. Dig up any info on a new player in town.

Close on Buffy squinching up her face as if in pain.

WILLOW
(OS) Dawnie, you can help me research. We'll hop on-line, check all the-

Flash back to the asylum. CrazyBuffy is sitting in the chair with her face squinched up in the same way.

DOCTOR
(OS) -possibilities for a full recovery, (shot of the doctor sitting behind a desk) but we have to proceed cautiously. If we're not careful--

JOYCE
Wait.

Reveal Joyce and Hank sitting in chairs across from the doctor. CrazyBuffy sits in another chair a little bit separated from them, with her knees drawn up again.

JOYCE
Are you saying that Buffy could be like she was before any of this happened?

DOCTOR
(gets up, comes around the desk) Mrs. Summers, you have to understand the severity of what's happened to your daughter. (sits on the edge of his desk) For the last six years, she's been in an undifferentiated type of schizophrenia.

HANK
We know what her condition is. (Buffy frowning) That's not what we're asking.

DOCTOR
Buffy's delusions are multi-layered. (Joyce and Hank listening intently) She believes she's some type of hero.

JOYCE
The Slayer.

DOCTOR
The Slayer, right, but that's only one level. She's also created an intricate latticework to support her primary delusion. In her mind, she's the central figure in a fantastic world beyond imagination. (Buffy staring into the distance, frowning) She's surrounded herself with friends, most with their own superpowers ... who are as real to her as you or me. More so, unfortunately. Together they face ... grand overblown conflicts against an assortment of monsters both imaginary and rooted in actual myth. Every time we think we're getting through to her, more fanciful enemies magically appear-

BUFFY
(suddenly realizing) How did I miss-

DOCTOR
and she's-

BUFFY
Warren and Jonathan, they did this to me!

Buffy becomes agitated, tries to get up out of her chair. The doctor reaches over to stop her.

DOCTOR
Buffy, it's all right. They can't hurt you here. You're with your family.

Buffy looks around, upset.

BUFFY
(tearful) Dawn?

HANK
(to doctor) That's the sister, right?

DOCTOR
A magical key. Buffy inserted Dawn into her delusion, actually rewriting the entire history of it to accommodate a need for a familial bond. (to Buffy) Buffy, but that created inconsistencies, didn't it? (Buffy staring at him) Your sister, your friends, all of those people you created in Sunnydale, they aren't as comforting as they once were. Are they? They're coming apart.

Buffy whimpers, lowers her head again.

JOYCE
Buffy, listen to what the doctor's saying, it's important.

DOCTOR
Buffy, you used to create these grand villains to battle against, and now what is it? Just ordinary students you went to high school with. (Buffy staring at him) No gods or monsters ... just three pathetic little men ... who like playing with toys.

Buffy frowns anxiously.
    
    """
    result = tag_text(input_text=input_text, generate_html=True)

    # print the location_descr and extraction_text for each extraction
    for extraction in result:
        location_descr = (
            extraction.attributes.get("location_descr", "N/A")
            if extraction.attributes
            else "N/A"
        )
        print("Location Description: ", location_descr)
        print("Extraction Text: ", extraction.extraction_text)
        print("--------------------------------")
