import os
import json
import warnings
from importlib import resources
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFDirectoryLoader,
    CSVLoader,
)
from langchain_community.vectorstores import FAISS

PKG = resources.files("parzivai")
FILE_PATH = PKG / "data"
# Define persistent folder for FAISS index
persist_folder = FILE_PATH / "faiss_index4"
os.makedirs(persist_folder, exist_ok=True)

# Check if the FAISS index already exists
index_path = os.path.join(persist_folder, "index.faiss")


# load the documents for the vector store
def load_config(file):
    try:
        with open(FILE_PATH / file, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        raise RuntimeError(f"Configuration file not found: {FILE_PATH / file}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Error decoding configuration file: {e}")


def load_embeddings_model():
    model_name_hf = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs_hf = {"device": "cpu"}
    encode_kwargs_hf = {"normalize_embeddings": False}
    return HuggingFaceEmbeddings(
        model_name=model_name_hf,
        model_kwargs=model_kwargs_hf,
        encode_kwargs=encode_kwargs_hf,
    )


def load_documents_and_create_vectorstore(embedding_model):
    """Load documents from URLs and static files to create FAISS vector store."""
    # Load URLs
    urls_data = load_config(file="urls.json")
    # Load documents from URLs
    web_docs = []
    for url in urls_data["urls"]:
        print(f"Loading documents from URL: {url}")
        try:
            loader = WebBaseLoader(url)
            web_docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading URL {url}: {e}")

    # Load documents from static folder
    static_data_folder = "./static_data"
    os.makedirs(static_data_folder, exist_ok=True)
    static_docs = []
    for file_name in os.listdir(static_data_folder):
        file_path = os.path.join(static_data_folder, file_name)
        try:
            if file_name.endswith(".pdf"):
                loader = PyPDFDirectoryLoader(file_path)
            elif file_name.endswith(".csv"):
                loader = CSVLoader(file_path=file_path)
            else:
                continue
            static_docs.extend(loader.load())
        except (IOError, ValueError) as e:
            warnings.warn(f"Problem loading '{file_name}': {e}", UserWarning)
            continue

    # Combine and process documents
    all_docs = web_docs + static_docs
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=768, chunk_overlap=30
    )
    doc_splits = text_splitter.split_documents(all_docs)
    print("Documents loaded and split successfully.")

    # Create and save FAISS vector store
    vectorstore = FAISS.from_documents(doc_splits, embedding_model)
    vectorstore.save_local(persist_folder)
    print(f"FAISS index initialized and saved successfully in {persist_folder}.")
    return vectorstore


def get_vectorstore(embedding_model):
    vectorstore_exists = os.path.exists(index_path)
    if vectorstore_exists:
        try:
            vectorstore = FAISS.load_local(
                persist_folder,
                embedding_model,
                allow_dangerous_deserialization=True,
            )
            print(f"FAISS index loaded successfully from {persist_folder}.")
        except Exception as e:
            raise RuntimeError(f"Error loading existing FAISS index: {e}") from e
    else:
        vectorstore = load_documents_and_create_vectorstore(embedding_model)

    retriever = vectorstore.as_retriever()
    return retriever
