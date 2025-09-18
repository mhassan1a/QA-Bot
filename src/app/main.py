# src/app/main.py
import gradio as gr
from llm import get_llm
from retriever import get_pdf_retriever
from langchain.chains import RetrievalQA

def answer_question(pdf_file, question):
    try:
        retriever = get_pdf_retriever(pdf_file)
        llm = get_llm()
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
        return qa_chain.run(question)
    except Exception as e:
        return f"❌ Error: {str(e)}"

iface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.File(label="Upload PDF", type="filepath"),
        gr.Textbox(label="Enter your question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Local Ollama PDF Q&A",
    description="Upload a PDF and ask questions. Works fully locally.",
    examples=[
        ["files/example.pdf", "Summarize the document."],
        ["files/example.pdf", "What are the main points?"]
    ],
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)
