from flask import Flask, render_template, request
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureChatOpenAI
from src.utils import *
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

load_dotenv()
app = Flask(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
DIMENSION = int(os.getenv("DIMENSION"))
INDEX_NAME = os.getenv("INDEX_NAME")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

embeddings = download_embeddings()

index_name = INDEX_NAME

doc_search = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

retriever = doc_search.as_retriever(search_type="similarity", search_kwargs={"k":3}) 
llm = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT_NAME,
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version="2025-01-01-preview"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}")
    ]
)

qa_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt)

rag_chain = create_retrieval_chain(retriever, qa_chain)


@app.route('/')

def index():
    return render_template('index.html')

@app.route('/get', methods=['GET','POST'])

def get_bot_response():
    user_text = request.form['msg']
    response = rag_chain.invoke({"input" : user_text})
    return str(response['answer'])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port= 2020, debug= True)

