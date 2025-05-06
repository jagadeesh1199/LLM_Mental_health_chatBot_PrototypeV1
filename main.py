from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import re

app = FastAPI()
model = SentenceTransformer('all-mpnet-base-v2')
chroma_client = chromadb.PersistentClient(path='data/chroma_db')
chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-mpnet-base-v2')
collection = chroma_client.get_collection(name="cbt_prompts_collection", embedding_function=chroma_ef)

# Optional: Use a local or OpenAI model
response_generator = pipeline("text-generation", model="openai-community/gpt2")

class UserInput(BaseModel):
    message: str

# Truncate text to first 2 sentences
def clean_response(text: str) -> str:
    return text.strip().strip('"').replace('\n', ' ').replace('\"', '')


@app.post("/cbt-journal")
def get_journaling_prompt(data: UserInput):
    query = data.message
    result = collection.query(query_texts=[query], n_results=1)

    if not result['documents'] or not result['documents'][0]:
        return {"error": "No matching prompts found. Please try rephrasing your input."}

    best_prompt = result['documents'][0][0]
    
    full_prompt = (
        f"You are a supportive CBT journaling coach. A user wrote:\n\"{query}\"\n\n"
        f"Based on the following journaling suggestion:\n\"{best_prompt}\"\n\n"
        "Kindly offer a reframed CBT-style journaling prompt in response:\n"
    )
    
    response_list = response_generator(full_prompt, max_length=100, do_sample=True)
    response_raw = response_list[0]['generated_text']
    response_cleaned = clean_response(response_raw.replace(full_prompt, ""))

    return {"prompt": best_prompt, "response": response_cleaned}

