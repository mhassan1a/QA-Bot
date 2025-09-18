import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

# Load API key
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def get_llm():
    """Return a Hugging Face LLM via hosted inference API."""
    return HuggingFaceEndpoint(
        repo_id="tiiuae/falcon-7b-instruct",  # you can swap this for another model on HF
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.7,
        max_new_tokens=512,
    )


def get_embedding_model():
    """Return Hugging Face embedding model (still via API if supported)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
