import os
from dotenv import load_dotenv
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams # type: ignore
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings

# Load environment variables from .env
load_dotenv()

IBM_API_KEY = os.getenv("IBM_API_KEY")
IBM_URL = os.getenv("IBM_URL")
PROJECT_ID = os.getenv("PROJECT_ID")


def get_llm():
    """
    Returns a configured Watsonx LLM for text generation using API key from .env.
    """
    model_id = 'mistralai/mixtral-8x7b-instruct-v01'
    
    # Generation parameters
    parameters = {
        GenParams.MAX_NEW_TOKENS: 512,
        GenParams.TEMPERATURE: 0.7,
        GenParams.TOP_P: 0.9,
        GenParams.FREQUENCY_PENALTY: 0.0,
        GenParams.PRESENCE_PENALTY: 0.0,
    }
    
    watsonx_llm = WatsonxLLM(
        api_key=IBM_API_KEY,
        model_id=model_id,
        url=IBM_URL,
        project_id=PROJECT_ID,
        params=parameters
    )
    
    return watsonx_llm


def get_embedding_model():
    """
    Returns a Watsonx embedding model using API key from .env.
    """
    model_id = "text-embedding-3-small"
    
    embedding_model = WatsonxEmbeddings(
        api_key=IBM_API_KEY,
        model_id=model_id,
        url=IBM_URL,
        project_id=PROJECT_ID,
        params={
            EmbedParams.DIMENSIONS: 1536  # adjust per model spec
        }
    )
    
    return embedding_model
