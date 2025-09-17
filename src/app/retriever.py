from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm import get_embedding_model # type: ignore

def get_pdf_retriever(pdf_path: str, chunk_size: int = 400, chunk_overlap: int = 50, top_k: int = 5):
    """
    Loads a PDF, splits it into chunks, embeds it, and returns a retriever.

    Args:
        pdf_path (str): Path to the PDF file.
        chunk_size (int): Maximum size of each text chunk.
        chunk_overlap (int): Overlap between text chunks.
        top_k (int): Number of similar documents to return during retrieval.

    Returns:
        retriever: A retriever object for similarity search.
    """
    loader = PDFMinerLoader(pdf_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(documents=chunks, embedding=get_embedding_model())
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever
