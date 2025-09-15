from pathlib import Path
from typing import List
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


def load_data(data: str) -> List[Document]:
    try:
        path = Path(data)
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Invalid directory: {data}")
        loader = DirectoryLoader(str(path), glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        if not documents:
            raise ValueError(f"No PDF documents found in {data}")
        return documents
    except Exception as e:
        raise RuntimeError(f"Failed to load documents: {e}") from e


def filter_min_doc(docs: List[Document]) -> List[Document]:
    try:
        if not docs:
            raise ValueError("Empty document list provided")
        minimal_docs = []
        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            if not doc.page_content.strip():
                continue
            minimal_docs.append(Document(page_content=doc.page_content, metadata={"source": src}))
        if not minimal_docs:
            raise ValueError("No valid documents after filtering")
        return minimal_docs
    except Exception as e:
        raise RuntimeError(f"Failed to filter documents: {e}") from e


def chunk_data(minimal_docs: List[Document]) -> List[Document]:
    try:
        if not minimal_docs:
            raise ValueError("No documents to chunk")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        chunks = splitter.split_documents(minimal_docs)
        if not chunks:
            raise ValueError("No chunks generated")
        return chunks
    except Exception as e:
        raise RuntimeError(f"Failed to chunk documents: {e}") from e


def download_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if not embeddings:
            raise ValueError("Embeddings initialization failed")
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to download embeddings: {e}") from e
