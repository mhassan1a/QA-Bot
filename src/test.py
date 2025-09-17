import pytest

from app.llm import  get_embedding_model
from app.main import answer_question
class DummyLLM:
    """Mock LLM that returns a fixed answer"""
    def __call__(self, prompt):
        return "This is a test answer."

class DummyEmbedding:
    """Mock embedding model"""
    def embed_text(self, text):
        return [0.0] * 1536  # dummy vector

@pytest.fixture
def dummy_pdf(tmp_path):
    """Create a dummy PDF file"""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("This is a test document.")
    return pdf_file

def test_create_retriever(monkeypatch, dummy_pdf):
    # Patch embedding model
    monkeypatch.setattr("your_module.get_embedding_model", lambda: DummyEmbedding())
    
    retriever = get_embedding_model(dummy_pdf)
    assert retriever is not None
    # Check retriever returns documents
    results = retriever.get_relevant_documents("test")
    assert len(results) > 0
    assert "test" in results[0].page_content

def test_answer_question(monkeypatch, dummy_pdf):
    # Patch LLM and embedding model
    monkeypatch.setattr("your_module.get_llm", lambda: DummyLLM())
    monkeypatch.setattr("your_module.get_embedding_model", lambda: DummyEmbedding())
    
    answer = answer_question(dummy_pdf, "What is this document?")
    assert isinstance(answer, str)
    assert "test answer" in answer.lower()
