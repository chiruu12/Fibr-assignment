import streamlit as st
import requests
import os
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000" # Base URL of FastAPI backend
UPLOAD_ENDPOINT = f"{API_BASE_URL}/upload"
QUERY_ENDPOINT = f"{API_BASE_URL}/query"

st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")
st.title("💬 Chat with your PDF")
st.caption("Upload a PDF and ask questions about its content.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file_processed" not in st.session_state:
    st.session_state.uploaded_file_processed = False
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

def display_chat_history():
    """Displays the chat messages stored in session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def add_message_to_history(role: str, content: str):
    """Adds a message to the chat history in session state."""
    st.session_state.messages.append({"role": role, "content": content})

def call_upload_api(uploaded_file) -> bool:
    """Sends the uploaded file to the FastAPI backend."""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    try:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=600) # Increased timeout for potentially long processing
        if response.status_code == 200:
            response_data = response.json()
            st.success(f"✅ Successfully processed: {response_data.get('filename', 'Unknown file')}")
            # print(f"Index saved at: {response_data.get('index_path')}") # Optional: log index path
            return True
        else:
            st.error(f"❌ Error processing file: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error connecting to API: {e}")
        return False
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during upload: {e}")
        return False


def call_query_api(question: str) -> str:
    """Sends the user's question to the FastAPI backend."""
    payload = {"question": question}
    try:
        with st.spinner("Thinking..."):
            response = requests.post(QUERY_ENDPOINT, json=payload, timeout=120) # Timeout for LLM response
        if response.status_code == 200:
            response_data = response.json()
            return response_data.get("answer", "Sorry, I couldn't find an answer.")
        elif response.status_code == 503: # Specific handling for "Vector store not ready"
             st.warning("The document might still be processing or wasn't uploaded successfully. Please wait or try uploading again.")
             return "Processing not complete. Please wait or re-upload."
        else:
            st.error(f"❌ Error getting answer: {response.status_code} - {response.text}")
            return "Sorry, an error occurred while fetching the answer."
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error connecting to API: {e}")
        return "Sorry, could not connect to the Q&A service."
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during query: {e}")
        return "Sorry, an unexpected error occurred."


# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        accept_multiple_files=False,
        key="pdf_uploader"
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.uploaded_filename or not st.session_state.uploaded_file_processed:
             st.write(f"Selected file: `{uploaded_file.name}`")
             if st.button(f"Process '{uploaded_file.name}'", key=f"process_{uploaded_file.name}"):
                 # Clear previous chat history on new file processing
                 st.session_state.messages = []
                 st.session_state.uploaded_file_processed = False # Reset status until success
                 st.session_state.uploaded_filename = None

                 success = call_upload_api(uploaded_file)
                 if success:
                     st.session_state.uploaded_file_processed = True
                     st.session_state.uploaded_filename = uploaded_file.name # Store filename on success
                     st.rerun() # Rerun to update main page state
                 else:
                     st.session_state.uploaded_file_processed = False
                     st.session_state.uploaded_filename = None
        elif st.session_state.uploaded_file_processed:
             st.success(f"✅ Ready to chat about `{st.session_state.uploaded_filename}`")

if not st.session_state.uploaded_file_processed:
    st.info("Please upload and process a PDF document using the sidebar to start chatting.")
else:
    st.info(f"Now chatting about: **{st.session_state.uploaded_filename}**")
    display_chat_history()

    if prompt := st.chat_input(f"Ask a question about {st.session_state.uploaded_filename}..."):
        # Add user message to chat history
        add_message_to_history("user", prompt)
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        assistant_response = call_query_api(prompt)

        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        add_message_to_history("assistant", assistant_response)
