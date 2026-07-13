import os
import threading

from functools import lru_cache
from typing import List
from fastapi import UploadFile

from config.settings import GOOGLE_API_KEY, VECTORSTORE_DIRECTORY, MODEL_OPTIONS
from core.document_processor import save_uploaded_file, load_documents_from_paths, split_documents_to_chunks

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from utils.logger import logger

_vectorstore_cache = {}
# Guards Chroma(...) construction: the startup warmup thread and an
# in-flight request can otherwise race to construct a client for the same
# persist_directory, which trips a KeyError in chromadb's SharedSystemClient.
_vectorstore_lock = threading.Lock()


def vectorstore_exists(persist_path: str) -> bool:
    exists = os.path.exists(persist_path) and bool(os.listdir(persist_path))
    logger.debug(f"Vectorstore exists at {persist_path}: {exists}")
    return exists

@lru_cache(maxsize=None)
def get_embeddings(model_provider: str):
    logger.debug(f"Loading embeddings model for provider: {model_provider}")
    if model_provider == "groq":
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    elif model_provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
    else:
        logger.error(f"Unsupported LLM Provider: {model_provider}")
        raise ValueError(f"Unsupported LLM Provider: {model_provider}")

def initialize_empty_vectorstores():
    logger.info("Initializing empty vectorstores...")
    for provider in MODEL_OPTIONS.keys():
        persist_path = VECTORSTORE_DIRECTORY[provider]
        os.makedirs(persist_path, exist_ok=True)

        # Warm the embeddings model cache at startup so the first real request
        # isn't stuck paying the (multi-second) model load cost.
        embedding = get_embeddings(provider)

        with _vectorstore_lock:
            if not os.listdir(persist_path):
                Chroma(
                    embedding_function=embedding,
                    persist_directory=persist_path
                )
                logger.debug(f"Initialized vectorstore for {provider} at {persist_path}")

            # Also warm the vectorstore connection cache so the first real request
            # doesn't pay the cost of opening/loading the persisted collection.
            # Chroma lazily loads its on-disk index on the first query rather than
            # on construction, so run a throwaway query here to force that load now.
            vectorstore = Chroma(persist_directory=persist_path, embedding_function=embedding)
            try:
                vectorstore.similarity_search("warmup", k=1)
            except Exception:
                pass
            _vectorstore_cache[provider] = vectorstore

    logger.info("Vectorstore initialization complete.")

async def upsert_vectorstore_from_pdfs(uploaded_files: List[UploadFile], model_provider: str):
    logger.debug(f"Upserting vectorstore for {model_provider}")
    file_paths = await save_uploaded_file(uploaded_files)
    docs = load_documents_from_paths(file_paths)
    chunks = split_documents_to_chunks(docs)

    vectorstore = load_vectorstore(model_provider)
    vectorstore.add_documents(chunks)
    logger.debug(f"Added {len(chunks)} chunks to vectorstore for {model_provider}.")

    return vectorstore

def load_vectorstore(model_provider: str):
    if model_provider in _vectorstore_cache:
        return _vectorstore_cache[model_provider]

    persist_path = VECTORSTORE_DIRECTORY[model_provider]
    logger.debug(f"Loading vectorstore from {persist_path}")

    with _vectorstore_lock:
        if model_provider in _vectorstore_cache:
            return _vectorstore_cache[model_provider]

        if vectorstore_exists(persist_path):
            logger.debug(f"Loading existing vectorstore for provider: {model_provider}")
            vectorstore = Chroma(persist_directory=persist_path, embedding_function=get_embeddings(model_provider))
            _vectorstore_cache[model_provider] = vectorstore
            return vectorstore

        logger.debug(f"VectorStore not found for provider: {model_provider}")
        raise ValueError(f"VectorStore not found for provider: {model_provider}")

def get_collections_count(model_provider: str):
    logger.debug(f"Getting collection count for provider: {model_provider}")
    vectorstore = load_vectorstore(model_provider)
    return vectorstore._collection.count()

def find_similar_chunks(model_provider: str, query: str):
    logger.debug(f"Searching for similar chunks for provider: {model_provider}")
    vectorstore = load_vectorstore(model_provider)
    results = vectorstore.similarity_search_with_relevance_scores(query, k=5)
    return [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "score": max(0.0, min(1.0, score))
        }
        for doc, score in results
    ]

def get_document_page(model_provider: str, source: str, page: int):
    logger.debug(f"Fetching page {page} of {source} for provider: {model_provider}")
    vectorstore = load_vectorstore(model_provider)
    all_chunks = vectorstore.get(where={"source": source})
    documents = all_chunks.get("documents", [])
    metadatas = all_chunks.get("metadatas", [])

    pages = sorted({m.get("page", 0) for m in metadatas})
    if not pages:
        raise ValueError(f"No indexed chunks found for source: {source}")

    target_page = page if page in pages else pages[0]
    page_text = "\n\n".join(
        doc for doc, meta in zip(documents, metadatas) if meta.get("page", 0) == target_page
    )

    return {
        "text": page_text,
        "page": target_page,
        "total_pages": len(pages),
        "pages": pages
    }
