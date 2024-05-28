import os
from dotenv import load_dotenv
from src.data_gouv_api import get_pdf_urls
from src.vectorstore import create_vector_store, load_vector_store
from src.retriever import create_self_query_retriever
from langchain_openai import OpenAIEmbeddings
from src.helpers import pretty_print_json, pretty_print_documents

# Load the environment variables
load_dotenv()

# Set the OPEN AI API key as an environment variable
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Set the DataGouv API key and dataset ID
DATAGOUV_API_KEY = os.getenv("DATAGOUV_API_KEY")
dataset_ID = "5cb9484d9ce2e7573ebc8895"

if __name__ == "__main__":

    # Create an OpenAI embeddings object
    embedding = OpenAIEmbeddings()

    # Get the PDF URLs and metadata from DataGouv API
    pdfs = get_pdf_urls(DATAGOUV_API_KEY, dataset_ID)


    # Create a vector store if not already exists and add pdfs embedding to it
    if not os.path.exists("docs/chroma"):
        print("Creating vector store...")
        vectordb = create_vector_store(embedding, pdfs)
    else:
        print("Loading vector store...")
        # Create a vector store from the PDF URLs
        vectordb = load_vector_store(embedding)


    # create a retriever from the vector store
    retriever = create_self_query_retriever(embedding, pdfs)

    question = "Que contient le cycle 2 en histoire?"
    docs = retriever.invoke(question)
    pretty_print_documents(docs)