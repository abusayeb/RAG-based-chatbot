from dotenv import load_dotenv
import os
from src.utils import load_data, filter_min_doc, chunk_data, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
DIMENSION = int(os.getenv("DIMENSION"))
INDEX_NAME = os.getenv("INDEX_NAME")


extracted_data = load_data("data/")
minimal_docs = filter_min_doc(extracted_data)
docs = chunk_data(minimal_docs)
embeddings = download_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = INDEX_NAME

if not pc.has_index(index_name):
    print("Creating index...")
    pc.create_index(
        name=index_name,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    embedding=embeddings,
    index_name=index_name,
    documents=docs
)
print("Indexing completed.")



