import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f".env file found and loaded from: {dotenv_path}")
else:
    print(f"Warning: .env file not found at {dotenv_path}")

import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our custom modules
from processing import DocumentProcessor, FAISS_INDEX_PATH
from qa_chain import create_qa_chain

app = FastAPI(
    title="PDF Q&A RAG API",
    description="API for uploading PDFs and asking questions using LangChain, Groq, and FAISS.",
    version="0.1.0",
)

# Middleware basically connects the FastAPI backend to the Streamlit frontend would have connected database and everything
# too if we had any
origins = [
    "http://localhost",
    "http://localhost:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

document_processor = DocumentProcessor(index_path=FAISS_INDEX_PATH)
qa_runnable = None


class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str


class UploadResponse(BaseModel):
    message: str
    filename: str
    index_path: str

# --- Helper Functions ---
def initialize_qa_chain():
    """Loads vector store and initializes the QA chain."""
    global qa_runnable
    print("Attempting to initialize QA chain...")
    vector_store = document_processor.get_vector_store() # Get the currently loaded store
    if vector_store:
        qa_runnable = create_qa_chain(vector_store)
        if qa_runnable:
            print("QA chain initialized successfully.")
        else:
            print("Failed to create QA chain even though vector store exists.")
    else:
        print("Vector store not loaded. Cannot initialize QA chain.")
        # Optionally try loading again, but DocumentProcessor constructor already tried
        vs_loaded = document_processor._load_vector_store() # Try loading explicitly
        if vs_loaded:
             document_processor.vector_store = vs_loaded # Update the instance's store
             qa_runnable = create_qa_chain(vs_loaded)
             if qa_runnable:
                 print("QA chain initialized successfully after explicit load.")
             else:
                 print("Failed to create QA chain after explicit load.")
        else:
             print("Explicit load failed. QA chain remains uninitialized.")


# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    """Initialize the QA chain on startup if the index exists."""
    print("FastAPI startup: Initializing QA chain...")
    initialize_qa_chain()

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts a PDF file, processes it, creates a vector store, and saves it.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are accepted.")

    # Use a temporary directory for robust file handling
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, file.filename)

        # Save the uploaded file temporarily
        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
        finally:
            file.file.close() # Ensure file handle is closed

        print(f"Temporarily saved PDF to: {temp_file_path}")

        # Process the PDF: Load and Split
        print("Loading and splitting PDF...")
        doc_chunks = document_processor.load_and_split_pdf(temp_file_path)

        if not doc_chunks:
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {file.filename}")

        # Create and Save Vector Store
        print("Creating and saving vector store...")
        vector_store = document_processor.create_and_save_vector_store(doc_chunks)

        if not vector_store:
            raise HTTPException(status_code=500, detail="Failed to create vector store.")

        # Initialize the QA chain in the background after upload and processing
        # This ensures the chain is ready for subsequent queries
        print("Scheduling QA chain initialization in background...")
        background_tasks.add_task(initialize_qa_chain)

        return UploadResponse(
            message="File processed successfully and vector store created/updated.",
            filename=file.filename,
            index_path=document_processor.index_path
        )


@app.post("/query", response_model=QueryResponse)
async def query_pdf(query: QueryRequest):
    """
    Accepts a question and returns an answer based on the processed PDF.
    """
    global qa_runnable
    if not qa_runnable:
        # Try to initialize again if it wasn't ready at startup or after upload
        print("QA chain not ready. Trying to initialize now...")
        initialize_qa_chain()
        if not qa_runnable:
            raise HTTPException(status_code=503, detail="Vector store not ready or QA chain failed to initialize. Please upload a document first.")

    print(f"Received query: {query.question}")
    try:
        # Invoke the QA chain
        result = qa_runnable.invoke({"input": query.question})
        answer = result.get("answer", "Sorry, I couldn't find an answer to that question.")
        print(f"Generated answer: {answer}")

        # You could optionally include the context documents:
        # context = result.get("context", [])
        # return QueryResponse(answer=answer, context=context)
        return QueryResponse(answer=answer)

    except Exception as e:
        print(f"Error during QA chain invocation: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")


if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
