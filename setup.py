# install using 'pip install -e .'

from setuptools import setup, find_packages
from pathlib import Path

requirements = Path("requirements.txt").read_text().splitlines()
setup(
    name="qa-bot",
    version="0.1.0",
    description="Local RAG PDF Q&A app using Ollama + Hugging Face embeddings + Gradio",
    author="Your Name",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=requirements,
    python_requires=">=3.10",
)