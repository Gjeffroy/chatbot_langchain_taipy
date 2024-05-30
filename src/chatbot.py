from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

def create_conversational_retriever_chain(llm_name, chain_type, retriever):
    """Create a conversational retriever from the given parameters."""

    llm = ChatOpenAI(model_name=llm_name, temperature=0)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
        verbose=True
    )

    return qa