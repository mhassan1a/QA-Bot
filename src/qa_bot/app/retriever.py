# src/app/retriever.py
import os
import hashlib
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qa_bot.app.llm import get_embedding_model
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

def get_pdf_retriever(pdf_path: str, chunk_size=400, chunk_overlap=50, top_k=5) -> EnsembleRetriever:
    """
    Load a PDF, split it into chunks, embed it, and return a retriever.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    # cache directory based on PDF hash
    pdf_hash = hashlib.md5(open(pdf_path, "rb").read()).hexdigest()
    persist_dir = f".chroma_db/{pdf_hash}"

    loader = PDFMinerLoader(pdf_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 2
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        persist_directory=persist_dir)

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
)
    return ensemble_retriever

if __name__ == "__main__":
    retriever = get_pdf_retriever("files/example.pdf")
    docs = retriever.get_relevant_documents("what is QLORA")
    print(docs)