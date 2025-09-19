import pytest
from qa_bot.app.retriever import get_pdf_retriever


class DummyEmbedding:
    """Mock embedding model with fake vectors"""
    def embed_documents(self, texts):
        return [[0.0] * 16 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 16


@pytest.fixture
def dummy_pdf(tmp_path):
    """Create a dummy text file pretending to be a PDF"""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("This is a test document.")
    return pdf_file


def test_create_retriever(monkeypatch, dummy_pdf):
    monkeypatch.setattr("qa_bot.app.llm.get_embedding_model", lambda: DummyEmbedding())
    retriever = get_pdf_retriever(str(dummy_pdf))

    results = retriever.get_relevant_documents("test")
    assert isinstance(results, list)
