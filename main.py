import os
from dotenv import load_dotenv
from src.data_gouv_api import get_pdf_urls
from src.vectorstore import create_vector_store, load_vector_store, delete_documents_by_metadata
from src.retriever import create_self_query_retriever, extract_unique_values
from src.chatbot import create_conversational_retriever_chain
from langchain_openai import OpenAIEmbeddings
from src.helpers import pretty_print_json, pretty_print_documents, pretty_print_chain_results

# from langchain.globals import set_verbose, set_debug
# set_debug(True)
# set_verbose(True)


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

    # filter pdf with 'discipline' == '-'
    forbidden_disciplines = ['cycle 4', 'cycle 2', 'cycle 3', '-']
    pdfs = [pdf for pdf in pdfs if pdf['discipline'] not in forbidden_disciplines]
    rejected_pdfs = [pdf for pdf in pdfs if pdf['discipline'] in forbidden_disciplines]

    # remove rejected pdfs from the vector store if they are in it
    if rejected_pdfs:
        delete_documents_by_metadata(pdfs, 'discipline', forbidden_disciplines)

    # Create a vector store if not already exists and add pdfs embedding to it
    if not os.path.exists("docs/chroma"):
        print("Creating vector store...")
        vectordb = create_vector_store(embedding, pdfs)
    else:
        print("Loading vector store...")
        # Create a vector store from the PDF URLs
        vectordb = load_vector_store(embedding)
#
#
    # create a retriever from the vector store
    retriever = create_self_query_retriever(
        llm_name="gpt-3.5-turbo",
        embeddings=embedding,
        pdfs=pdfs,
        k=20
    )


    question = "que contient le premgramme de mathematique de terminale S?"
    # Test the retriever
    docs = retriever.invoke(question)
    pretty_print_documents(docs)

    # create qa chain
    qa_chain = create_conversational_retriever_chain("gpt-3.5-turbo", "map_reduce", retriever)
    result = qa_chain.invoke({"question": question, 'chat_history': []})
    pretty_print_chain_results(result, indent=2)
#
#
# # bug
# # question = "Qu'est ce qui est enseigné en langue vivante en CAP?"