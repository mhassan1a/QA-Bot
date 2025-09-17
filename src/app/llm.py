
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline


def get_llm():
    """Return a Hugging Face LLM wrapped for LangChain."""
    generator = pipeline(
        "text-generation",
        model="tiiuae/falcon-7b-instruct",  # you can swap in any instruction-tuned model
        device=-1,  # -1 = CPU, or set CUDA device id (e.g., 0 for GPU)
        max_new_tokens=512,
        temperature=0.7,
    )
    return HuggingFacePipeline(pipeline=generator)


def get_embedding_model():
    """Return a Hugging Face embedding model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")