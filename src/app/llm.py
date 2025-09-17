
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18

    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
    from ibm_watsonx_ai import Credentials
    from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.chains import RetrievalQA

def get_llm():
    model_id = 'mistralai/mixtral-8x7b-instruct-v01'
    parameters = {
        .......,
        .......,
    }
    project_id = "skills-network"
    watsonx_llm = WatsonxLLM(
        model_id=......,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=.......,
        params=.....,
    )
    return watsonx_llm