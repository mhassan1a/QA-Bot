import pytest
from qa_bot.app.main import answer_question
from qa_bot.app.retriever import get_pdf_retriever


class DummyLLM:
    """Mock LLM that returns a fixed answer"""
    def __call__(self, prompt):
        return "This is a test answer."


class DummyEmbedding:
    """Mock embedding model with fake vectors"""
    def embed_documents(self, texts):
        return [[0.0] * 1536 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 1536


@pytest.fixture
def dummy_pdf(tmp_path):
    """Create a dummy text file pretending to be a PDF"""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("This is a test document.")
    return pdf_file


def test_create_retriever(monkeypatch, dummy_pdf):
    # Patch embedding model inside retriever module
    monkeypatch.setattr("src.app.retriever.get_embedding_model", lambda: DummyEmbedding())
    
    retriever = get_pdf_retriever(str(dummy_pdf))
    assert retriever is not None
    
    results = retriever.get_relevant_documents("test")
    assert isinstance(results, list)


def test_answer_question(monkeypatch, dummy_pdf):
    # Patch LLM + embeddings inside main module
    monkeypatch.setattr("src.app.main.get_llm", lambda: DummyLLM())
    monkeypatch.setattr("src.app.main.get_embedding_model", lambda: DummyEmbedding())
    
    answer = answer_question(dummy_pdf, "What is this document?")
    assert isinstance(answer, str)
    assert "test answer" in answer.lower()
