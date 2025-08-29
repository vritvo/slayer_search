from openai import OpenAI
from dotenv import load_dotenv
import os


def make_embedding(script):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(input=script, model="text-embedding-3-small")

    return response.data[0].embedding
