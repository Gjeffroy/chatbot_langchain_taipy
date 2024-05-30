import os
from dotenv import load_dotenv
from src.data_gouv_api import get_pdf_urls
from src.vectorstore import create_vector_store, load_vector_store
from src.retriever import create_self_query_retriever
from src.chatbot import create_conversational_retriever_chain
from langchain_openai import OpenAIEmbeddings
from src.helpers import pretty_print_json, pretty_print_documents


# Load the environment variables
load_dotenv()

# Set the OPEN AI API key as an environment variable
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Set the DataGouv API key and dataset ID
DATAGOUV_API_KEY = os.getenv("DATAGOUV_API_KEY")
dataset_ID = "5caaff5d06e3e73db04eac13" # https://www.data.gouv.fr/fr/datasets/programmes-denseignement-du-second-degre/#/information

if __name__ == "__main__":

    # Create an OpenAI embeddings object
    embedding = OpenAIEmbeddings()

    # Get the PDF URLs and metadata from DataGouv API
    pdfs = get_pdf_urls(DATAGOUV_API_KEY, dataset_ID)
    pretty_print_json(pdfs)


    # Create a vector store if not already exists and add pdfs embedding to it
    if not os.path.exists("docs/chroma"):
        print("Creating vector store...")
        vectordb = create_vector_store(embedding, pdfs, k=1)
    else:
        print("Loading vector store...")
        # Create a vector store from the PDF URLs
        vectordb = load_vector_store(embedding)


    # create a retriever from the vector store
    retriever = create_self_query_retriever(
        llm_name="gpt-3.5-turbo",
        embeddings=embedding,
        pdfs=pdfs
    )

    # create qa chain
    qa_chain = create_conversational_retriever_chain("gpt-3.5-turbo", "stuff", retriever)

    question = "Quels ensignement relatif à la géographie en premiere ES?"
    result = qa_chain.invoke({"question": question, 'chat_history': []})
    print(result)

    # docs = retriever.invoke(question)
    # pretty_print_documents(docs)