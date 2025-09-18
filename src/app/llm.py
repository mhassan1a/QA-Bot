# src/app/llm.py
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

def get_llm(model_name: str = "gemma3"):
    """
    Return a local Ollama LLM for generation.
    """
    return OllamaLLM(model=model_name)

def get_embedding_model():
    """
    Return a Hugging Face embedding model for RAG.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
