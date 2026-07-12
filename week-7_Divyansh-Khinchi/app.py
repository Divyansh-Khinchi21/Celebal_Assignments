import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
from rag_system import DocQAEngine

# Load environment variables
load_dotenv(override=True)

# Main Title of the App
st.title("Smart PDF & Document Q&A Assistant")
st.write("A clean Retrieval-Augmented Generation (RAG) tool to ask questions over documents.")

# Read API Key from environment
api_key = os.environ.get("GEMINI_API_KEY", "")

# Single-page layout (No sidebar)
# 1. Configuration Settings in an expander
with st.expander("⚙️ Pipeline Configurations", expanded=False):
    st.write("Adjust chunk settings and retrieval parameters:")
    segment_size = st.slider("Segment Size (Chars)", min_value=100, max_value=1000, value=500, step=50)
    overlap = st.slider("Overlap (Chars)", min_value=0, max_value=200, value=100, step=10)
    k_retrieved = st.slider("Top K Segments to Retrieve", min_value=1, max_value=5, value=3, step=1)
    
    if not api_key:
        st.info("ℹ️ Running in **Offline Mode** (No GEMINI_API_KEY environment variable detected).")
    else:
        st.success("🟢 Running in **Online Mode** (Gemini API Connected).")

# Initialize Session State
if "engine" not in st.session_state:
    st.session_state.engine = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

st.markdown("---")

# 2. Upload Section
st.subheader("📁 Upload Document")
uploaded_file = st.file_uploader("Upload a PDF or TXT file to index:", type=["pdf", "txt"])

if uploaded_file:
    # Trigger indexing if it's a new file
    if st.session_state.current_file != uploaded_file.name:
        with st.spinner("Analyzing and vectorizing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = tmp.name
            
            try:
                # Initialize DocQAEngine
                current_key = api_key if api_key else "no_key"
                engine = DocQAEngine(current_key)
                engine.load_document(temp_path, segment_size, overlap)
                
                # Store in session state
                st.session_state.engine = engine
                st.session_state.current_file = uploaded_file.name
                
                st.success(f"Successfully processed: {uploaded_file.name} ({len(engine.docs_segments)} segments created)")
            except Exception as e:
                st.error(f"Error parsing document: {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

st.markdown("---")

# 3. Question Answering Section
if st.session_state.engine:
    st.subheader("💬 Ask Assistant")
    st.write(f"Querying document: `{st.session_state.current_file}`")
    
    # Sync the API key
    st.session_state.engine.api_key = api_key if api_key else "no_key"
    
    query = st.text_input("Type your question below:")
    
    if st.button("Submit Question") and query:
        with st.spinner("Generating answer..."):
            try:
                answer, retrieved_segs = st.session_state.engine.query_assistant(query, k=k_retrieved)
                
                # Display Answer in a clean block
                st.markdown("### 🤖 Answer:")
                st.success(answer)
                
                # Display Retrieved Segments
                st.markdown("### 📄 Retrieved Context Sources:")
                for i, (score, segment) in enumerate(retrieved_segs):
                    st.write(f"**Source #{i+1}** (Cosine Similarity: `{score:.4f}`):")
                    # Display segment inside a blockquote
                    st.markdown(f"> {segment}")
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
else:
    st.info("Please upload a file above to begin questioning.")
