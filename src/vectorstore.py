from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import requests


def check_url_type(url):
    try:
        response = requests.head(url)
        content_type = response.headers.get('content-type')
        print(content_type)
        if content_type:
            if 'pdf' in content_type.lower():
                return 'PDF'
            elif 'html' in content_type.lower():
                return 'HTML'
            else:
                return 'Unknown'
        else:
            return 'Unknown'
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return 'Error'


def create_vector_store(embedding, pdf_urls, persist_directory="docs/chroma/", k=None):
    """Create a vector store from the documents URLs."""

    if k is None:
        k = len(pdf_urls)

    # Load the documents from the URLs
    docs = []
    for pdf in pdf_urls[:k]:
        print(pdf['url'])
        type = check_url_type(pdf['url'])

        try:
            if type == 'PDF':
                loader = PyPDFLoader(pdf['url'])
            if type == 'HTML':
                loader = WebBaseLoader(pdf['url'])
            if type == 'Unknown':
                print('Unknown file type')
                continue
        except Exception as e:
            print("Error:", e)
            continue

        doc = loader.load()

        # Add metadata to the pages so they can be used for retrieval
        for page in doc:
            page.metadata["discipline"] = pdf["discipline"]
            page.metadata["cycle"] = pdf["cycle"]
            page.metadata["description"] = pdf["description"]

        docs.extend(doc)

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    # Create a Chroma vector store from the documents
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )

    return vectordb

def load_vector_store(embeddings, persist_directory="docs/chroma/"):
    """Load a vector store from disk."""
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)