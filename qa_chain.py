import os
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import Runnable

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RETRIEVAL_QA_CHAT_PROMPT_HUB = "langchain-ai/retrieval-qa-chat"


def create_qa_chain(vector_store: FAISS = None) -> Optional[Runnable]:
    """
    Creates a retrieval-based question-answering chain.

    Args:
        vector_store: An initialized FAISS vector store instance.

    Returns:
        A runnable LangChain retrieval chain object, or None if setup fails.
    """
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found in environment variables.")
        return None
    if not vector_store:
        print("Error: Vector store is not initialized.")
        return None
    try:
        # Initialize the Groq LLM
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            groq_api_key=GROQ_API_KEY,  # Use the module-level variable
            temperature=0.0,  # Keep temperature low for factual QA
            max_retries=2,
        )

        # Pull the retrieval QA chat prompt from Langchain Hub
        retrieval_qa_chat_prompt = hub.pull(RETRIEVAL_QA_CHAT_PROMPT_HUB)

        # Create the retriever from the vector store
        # You can adjust search_kwargs as needed (e.g., k for number of documents)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Create the chain to combine retrieved documents into a coherent answer
        combine_docs_chain = create_stuff_documents_chain(
            llm, retrieval_qa_chat_prompt
        )

        # Create the final retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        print("QA chain created successfully.")
        return retrieval_chain

    except Exception as e:
        print(f"Error creating QA chain: {e}")
        return None
