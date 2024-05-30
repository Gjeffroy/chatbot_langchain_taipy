from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import requests


# helper function to delete document in chroma base on metadata
def delete_documents_by_metadata(vectordb, metadata_field, values):
    """
    Delete documents from the vector store based on a specific metadata field and value.

    Parameters:
        vectordb (object): The vector store object.
        metadata_field (str): The metadata field to filter on.
        value (list): The values to filter on.

    Returns:
        None
    """
    # Get the documents to delete
    coll = vectordb.get()  # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])

    ids_to_del = []

    for idx in range(len(coll['ids'])):

        id = coll['ids'][idx]
        metadata = coll['metadatas'][idx]

        if metadata[metadata_field] in values:
            ids_to_del.append(id)

    vectordb._collection.delete(ids_to_del)


# helper function to check the type of content at a given URL
def check_url_type(url):
    """
    Determines the type of content at a given URL by checking the 'Content-Type' header.

    If the 'Content-Type' header is not present, it checks the URL for common file extensions.

    Parameters:
    url (str): The URL to check.

    Returns:
    str: 'PDF' if the content is a PDF file, 'HTML' if the content is an HTML page, 'Unknown'
         if the content type is neither PDF nor HTML, or if it cannot be determined.
         'Error' if there is an issue with the URL request.
    """
    try:
        response = requests.head(url)
        content_type = response.headers.get('content-type')
        if content_type:
            if 'pdf' in content_type.lower():
                return 'PDF'
            elif 'html' in content_type.lower():
                return 'HTML'
            else:
                return 'Unknown'
        else:
            if '.pdf' in url.lower():
                return 'PDF'
            elif '.html' in url.lower() or '.htm' in url.lower():
                return 'HTML'
            else:
                return 'Unknown'
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return 'Error'


# function to create a vector store from the documents URLs
def create_vector_store(embedding, pdf_urls, persist_directory="docs/chroma/"):
    """Create a vector store from the documents URLs."""

    # Load the documents from the URLs
    docs = []
    for pdf in pdf_urls:
        print(pdf['url'])
        type = check_url_type(pdf['url'])
        print(type)

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