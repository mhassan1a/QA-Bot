import pytest
from qa_bot.app.main import answer_question


class DummyLLM:
    """Mock LLM that returns a fixed answer"""
    def __call__(self, prompt):
        return "This is a test answer."


class DummyEmbedding:
    """Mock embedding model"""
    def embed_documents(self, texts):
        return [[0.0] * 16 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 16


@pytest.fixture
def dummy_pdf(tmp_path):
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("This is a test document.")
    return pdf_file


def test_answer_question(monkeypatch, dummy_pdf):
    monkeypatch.setattr("qa_bot.app.llm.get_llm", lambda: DummyLLM())
    monkeypatch.setattr("qa_bot.app.llm.get_embedding_model", lambda: DummyEmbedding())

    answer = answer_question(dummy_pdf, "What is this document?")
    assert isinstance(answer, str)
    assert "test answer" in answer.lower()
