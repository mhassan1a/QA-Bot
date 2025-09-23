[![Python application CI](https://github.com/mhassan1a/QA-Bot/actions/workflows/main.yml/badge.svg)](https://github.com/mhassan1a/QA-Bot/actions/workflows/main.yml)


# Local RAG PDF Q&A App

This project implements a **Retrieval-Augmented Generation (RAG) PDF Q&A application** using:

- **Ollama**: Local LLM for text generation (`gemma3`, `mistral`, etc.)  
- **Hugging Face embeddings**: `sentence-transformers/all-MiniLM-L6-v2` for vector similarity  
- **Chroma DB**: Stores embeddings of PDF chunks  
- **Gradio**: Web interface for uploading PDFs and asking questions  

Everything runs **fully locally**. No external API keys are required.  

---

## Project Structure

```
QA-Bot/
├─ src/
│  └─ app/
│     ├─ llm.py           # Ollama LLM + embeddings
│     ├─ retriever.py     # PDF loader, splitter, and Chroma retriever
│     └─ main.py          # Gradio interface
├─ requirements.txt       # Python dependencies
├─ Makefile               # Commands for setup, run, test, lint, etc.
└─ README.md

````

## Requirements

- Python 3.10+  
- Ollama installed locally ([https://ollama.com](https://ollama.com))  
- CPU or GPU for local LLM inference  
- Virtual environment recommended  

---

## Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone <repo_url>
cd QA-Bot
````

### 2️⃣ Create virtual environment and install dependencies

```bash
make venv
make install
```

### 3️⃣ Run the Gradio App

```bash
make run
```

* The app will launch at `http://localhost:7860`
* Upload a PDF and ask questions — Ollama will generate answers using local embeddings.

---

## Makefile Commands

| Command        | Description                                |
| -------------- | ------------------------------------------ |
| `make venv`    | Create virtual environment                 |
| `make install` | Install dependencies                       |
| `make run`     | Run the Gradio Q\&A app                    |
| `make test`    | Run pytest with coverage                   |
| `make lint`    | Run ruff linting                           |
| `make format`  | Auto-format code with ruff                 |
| `make clean`   | Remove virtual environment and cache files |

---

## Notes

1. **Chroma DB caching**: Each PDF is hashed and stored in `.chroma_db/<hash>` to speed up repeat queries.
2. **LLM device**: By default, Ollama runs on CPU. If GPU is available, set `device` in the Ollama wrapper.
3. **Embedding dimensions**: Do not change the embedding model after creating a Chroma collection; otherwise, you’ll get dimension mismatch errors.

---

## Troubleshooting

* **Dimension mismatch**: Delete `.chroma_db` folder and rebuild if switching embeddings.
* **Pydantic v2 errors**: Ensure `pydantic<2` in your environment for compatibility with LangChain/Chroma.
