from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings

from langchain_pinecone import PineconeVectorStore

from  langchain_huggingface import HuggingFaceEmbeddings      #
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
from src.prompt import *
import os

load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY            # type: ignore      
os.environ["HF_TOKEN"] = HF_TOKEN                # type: ignore


embedding = download_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)            # type: ignore


retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

model_id = "meta-llama/Llama-3.1-8B-Instruct"

llm_engine = HuggingFaceEndpoint(
    repo_id=model_id,
    task="conversational", 
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    temperature=0.3,
    max_new_tokens=512
) # type: ignore

chatModel = ChatHuggingFace(llm=llm_engine)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


@app.route("/")             # type: ignore
def index():
    return render_template('chat.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
