from qa_bot.app.main import answer_question

def test_answer_question_with_real_models(real_pdf):
    answer = answer_question(real_pdf, "What is this document about?")
    assert isinstance(answer, str)
    assert len(answer) > 0
