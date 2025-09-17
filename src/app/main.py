import gradio as gr # type: ignore
from llm import get_llm # type: ignore
from retriever import get_pdf_retriever # type: ignore
from langchain.chains import RetrievalQA

def answer_question(pdf_file, question):
    retriever = get_pdf_retriever(pdf_file)
    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    result = qa_chain.run(question)
    return result

iface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.File(label="Upload PDF", type="filepath"),
        gr.Textbox(label="Enter your question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Watsonx PDF Q&A",
    description="Upload a PDF and ask questions. Powered by IBM Watsonx LLM and embeddings."
)

if __name__ == "__main__":
    iface.launch()