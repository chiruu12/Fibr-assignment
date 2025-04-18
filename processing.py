import os
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

FAISS_INDEX_PATH = "faiss_index"

class DocumentProcessor:
    """
    Handles loading, splitting, embedding, and vector store management for documents.
    """

    def __init__(self, index_path: str = FAISS_INDEX_PATH):
        """
        Initializes the DocumentProcessor.

        Args:
            index_path: The path to save/load the FAISS index.
        """
        self.EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
        self.index_path = index_path
        # Initialize embeddings (using Sentence Transformers model via HuggingFaceEmbeddings)
        # Ensure device is set appropriately if GPU is available and desired, e.g., model_kwargs={'device': 'cuda'}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'} # Explicitly use CPU
        )
        self.vector_store: Optional[FAISS] = self._load_vector_store()

    def load_and_split_pdf(self, file_path: str) -> List[Document]:
        """
        Loads a PDF, extracts text, and splits it into manageable chunks.

        Args:
            file_path: The path to the PDF file.

        Returns:
            A list of document chunks (LangChain Document objects).
        """
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load() # Loads pages as individual documents
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            # Consider raising a custom exception or returning empty list
            return []

        # Combine page text or process page by page depending on desired chunking strategy
        # For simplicity here, we join text, but page-aware splitting might be better
        # full_text = "\n".join([page.page_content for page in pages])
        # document_for_splitting = [Document(page_content=full_text)] # Wrap in Document for splitter

        # Alternatively, split each page document individually:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Adjust as needed
            chunk_overlap=150,  # Adjust as needed
            length_function=len,
        )
        # Split documents loaded by PyPDFLoader (which are already list of Document)
        docs = text_splitter.split_documents(pages)
        print(f"Split PDF {file_path} into {len(docs)} chunks.")
        return docs

    def create_and_save_vector_store(self, docs: List[Document]) -> Optional[FAISS]:
        """
        Creates a FAISS vector store from document chunks and saves it to disk.

        Args:
            docs: A list of document chunks.

        Returns:
            The created FAISS vector store object, or None if creation failed.
        """
        if not docs:
            print("No documents provided to create vector store.")
            return None
        try:
            print(f"Creating FAISS index at {self.index_path}...")
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            self.vector_store.save_local(self.index_path)
            print(f"FAISS index saved successfully to {self.index_path}.")
            return self.vector_store
        except Exception as e:
            print(f"Error creating or saving FAISS index: {e}")
            return None

    def _load_vector_store(self) -> Optional[FAISS]:
        """
        Loads the FAISS vector store from the specified path if it exists.

        Returns:
            The loaded FAISS vector store object, or None if not found or error occurs.
        """
        if os.path.exists(self.index_path):
            try:
                print(f"Loading existing FAISS index from {self.index_path}...")
                # allow_dangerous_deserialization=True is needed for FAISS loading
                vector_store = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                 )
                print("FAISS index loaded successfully.")
                return vector_store
            except Exception as e:
                print(f"Error loading FAISS index from {self.index_path}: {e}")
                return None
        else:
            print(f"FAISS index not found at {self.index_path}. Need to create one.")
            return None

    def get_vector_store(self) -> Optional[FAISS]:
        """
        Returns the current vector store object (loaded or newly created).
        """
        return self.vector_store
