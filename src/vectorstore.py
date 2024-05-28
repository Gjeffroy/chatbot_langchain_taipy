from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


def create_vector_store(embedding, pdf_urls, persist_directory="docs/chroma/"):
    """Create a vector store from the PDF URLs."""

    # Load the PDFs from the URLs
    docs = []
    for pdf in pdf_urls:
        loader = PyPDFLoader(pdf['url'])
        doc = loader.load()

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