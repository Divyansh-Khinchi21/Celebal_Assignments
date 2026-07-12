# Smart PDF & Document Q&A Assistant (RAG)

This application is an intelligent search and QA assistant that uses Retrieval-Augmented Generation (RAG) to find answers directly from uploaded documents.

By vectorizing document text segments, it uses similarity search to retrieve matching parts and generate answers using the Gemini API.

## Project Files

*   `app.py`: Streamlit frontend with single-page configurations and clean document Q&A interfaces.
*   `rag_system.py`: RAG pipeline containing:
    *   Document parser (`pypdf`)
    *   Text segmenter (slicing text into pieces)
    *   Vector creation (`models/gemini-embedding-001`)
    *   Similarity math (Cosine Similarity calculations using NumPy)
    *   Gemini generation (`models/gemini-3.5-flash`)
*   `requirements.txt`: Python package requirements.
*   `.env`: Configuration file for API keys.

## Getting Started

### 1. Installation
Install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Add API Key
Create a `.env` file in the folder (or use the template) and write your Gemini key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run the App
Launch the web interface:
```bash
streamlit run app.py
```
This will open the app in your browser at `http://localhost:8501`.
