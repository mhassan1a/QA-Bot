# src/qa_bot/app/main.py
import gradio as gr
from qa_bot.app.llm import get_llm
from qa_bot.app.retriever import get_pdf_retriever
from langchain.prompts import PromptTemplate

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer in detail, even if you only have partial information."
    ),
)

def answer_question(pdf_file, question):
    try:
        retriever = get_pdf_retriever(pdf_file, top_k=3)
        llm = get_llm()
        
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        chain = QA_PROMPT | llm
        
        result = chain.invoke({"context": context, "question": question})
        return result 
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
        ["files/example.pdf", "What is LoRA?"]
    ],
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8080, share=False)
