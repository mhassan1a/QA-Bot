from qa_bot.app.retriever import get_pdf_retriever

def test_retriever_with_real_embedding(real_pdf):
    retriever = get_pdf_retriever(str(real_pdf))
    results = retriever.get_relevant_documents("AI")

    assert isinstance(results, list)
    assert len(results) > 0
    assert any("test document" in r.page_content.lower() for r in results)
