# PDF Q&A Chatbot - Fibr.ai AI/ML Intern Task

## Overview

This project is a simple web application built as part of the AI/ML Intern evaluation process for Fibr.ai. The application allows users to upload a PDF document and then ask questions about its content. It demonstrates the use of modern AI tools and libraries to create a Retrieval-Augmented Generation (RAG) system.

The development process heavily utilized AI-assisted coding tools (like Codeium/Cascade) for efficient code generation, debugging, and iteration, showcasing effective interaction with LLM-based development tools.

## Features

*   **PDF Upload:** Users can upload PDF files through a simple web interface.
*   **Content Processing:** Uploaded PDFs are processed, text is extracted, and split into manageable chunks.
*   **Vector Storage:** Text chunks are embedded using Hugging Face models and stored in a FAISS vector store for efficient similarity search.
*   **Question Answering:** Users can ask questions related to the uploaded PDF content.
*   **RAG Implementation:** The application retrieves relevant text chunks from the vector store based on the user's query and uses the Groq API to generate answers based on the retrieved context.

## Tech Stack

*   **Backend:** Python, FastAPI
*   **Frontend:** Streamlit
*   **Core AI/ML Libraries:**
    *   LangChain: For orchestrating the RAG pipeline (document loading, splitting, retrieval, QA chain).
    *   Groq: For accessing fast LLM inference (Llama 3.1 8B Instant).
    *   FAISS: For efficient vector storage and similarity search.
    *   `langchain-huggingface`: For using Hugging Face's sentence transformer models for embeddings.
*   **Other:** `python-dotenv`, `pypdf`, `uvicorn`

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/chiruu12/Fibr-assignment.git
    cd Fibr-assignment
    ```

2.  **Create and activate a virtual environment:**
    - #### Using venv (recommended)
    ```bash
    python -m venv venv
    ```
    - #### Windows
    ```bash
    .\venv\Scripts\activate
    ```
    - #### macOS/Linux
    ```bash
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file:**
    Create a file named `.env` in the project's root directory and add your API keys:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    HUGGINGFACE_API_KEY=your_huggingface_api_key_here
    ```
    *   Get a Groq API key from [https://console.groq.com/keys](https://console.groq.com/keys).

## Running the Application

You need to run two components in separate terminals: the FastAPI backend and the Streamlit frontend.

1.  **Run the FastAPI Backend:**
    Open a terminal in the project root directory and run:
    ```bash
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    ```
    The `--reload` flag automatically restarts the server when code changes are detected.

2.  **Run the Streamlit Frontend:**
    Open a second terminal in the project root directory and run:
    ```bash
    streamlit run ui.py
    ```
    Streamlit will typically open the application automatically in your default web browser (usually at `http://localhost:8501`).

## Code Structure and Modularity

*   `api.py`: Defines the FastAPI backend, handling API endpoints for file uploads and queries.
*   `ui.py`: Contains the Streamlit code for the user interface.
*   `processing.py`: Handles PDF loading, text splitting, embedding generation, and FAISS vector store management (`DocumentProcessor` class).
*   `qa_chain.py`: Sets up the LangChain retrieval QA chain using the vector store and the Groq LLM.
*   `requirements.txt`: Lists project dependencies.
*   `.env`: Stores sensitive API keys (should not be committed to Git).
*   `faiss_index/`: Directory created automatically to store the FAISS vector index.

This structure separates concerns, making the codebase easier to understand, maintain, and extend.


### Further improvements could include:
*   **Error Handling:** Implement more robust error handling and user feedback mechanisms.
*   **Testing:** Add unit tests for critical components to ensure reliability.
*   **Deployment:** Consider deploying the application using Docker or a cloud service for easier access and scalability.
*   **User Authentication:** Implement user authentication to manage access to the application and user data.
*   **UI Enhancements:** Improve the user interface for better usability and aesthetics.
*   **Performance Optimization:** Optimize the embedding and retrieval processes for larger documents or datasets.
*   **Documentation:** Add more detailed documentation and usage examples for developers and users.
*   **Logging and Monitoring:** Implement logging and monitoring to track application performance and user interactions.
*   **Adding more LLMs:** Integrate additional LLMs for more diverse question-answering capabilities.