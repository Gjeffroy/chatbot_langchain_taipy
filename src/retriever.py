from langchain_community.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.vectorstores import Chroma



# helper function to extract values from a list of dictionaries
def extract_values(dict_list, key):
    """
    Extracts all values for a given key from a list of dictionaries.

    Parameters:
    dict_list (list): List of dictionaries
    key (str): The key whose values need to be extracted

    Returns:
    list: A list of values corresponding to the given key from all dictionaries
    """
    values = [d[key] for d in dict_list if key in d]
    return values


# Create a retriever from the vector store
def create_self_query_retriever(llm_name, embeddings, pdfs, persist_directory="docs/chroma/"):
    """ Create a self query retriever from the vector store and using specific metadata fields.

    Args:

    embeddings (object): The embeddings object to use for the retriever.
    pdfs (list): A list of dictionaries containing metadata for the PDFs.
    persist_directory (str): The directory to persist the retriever data.

    Returns:
    object: The retriever object created from the vector store.
    """
    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The lecture the chunk is from, should be one of " + str(extract_values(pdfs, "url")),
            type="string",
        ),
        AttributeInfo(
            name="page",
            description="The page from the lecture",
            type="integer",
        ),
        AttributeInfo(
            name="cycle",
            description="The cycle the chunk is from, should be one of " + str(extract_values(pdfs, "cycle")),
            type="string",
        ),
        AttributeInfo(
            name="description",
            description="Brief description of the overall content of the source, not specific to the chunk.",
            type="string",
        ),
        AttributeInfo(
            name="discipline",
            description="The name of the discipline the chunk is about, should be one of " + str(extract_values(pdfs, "discipline")),
            type="string",
        )
    ]

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    document_content_description = "Lecture notes"
    llm = OpenAI(temperature=0)

    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectordb,
        document_content_description,
        metadata_field_info,
        verbose=True,
        search_type="mmr",
        max_tokens_limit=4000,
        enable_limit=True,
        search_kwargs={"k": 1}
    )

    return retriever


