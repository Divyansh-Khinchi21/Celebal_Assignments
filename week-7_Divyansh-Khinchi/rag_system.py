import os
import numpy as np
from pypdf import PdfReader
import google.generativeai as genai

# 1. Parse document text (PDF/TXT)
def parse_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        reader = PdfReader(file_path)
        extracted_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n"
        return extracted_text
    elif ext in ['.txt', '.md']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    else:
        raise ValueError("Unsupported format. Use PDF or TXT files.")

# 2. Text Segmenter
def segment_text(text, segment_size=500, overlap=100):
    segments = []
    start = 0
    while start < len(text):
        end = start + segment_size
        seg = text[start:end].strip()
        if seg:
            segments.append(seg)
        start += (segment_size - overlap)
    return segments

# Deterministic mock vectors for offline mode
def create_mock_vector(text):
    vector = np.zeros(768)
    text = text.lower()
    for char in text:
        idx = ord(char) % 768
        vector[idx] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()

# 3. Create vector representation
def create_vector(text, api_key):
    if not api_key or len(api_key) < 15 or "your_" in api_key.lower() or api_key == "no_key":
        return create_mock_vector(text)
        
    try:
        genai.configure(api_key=api_key)
        response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text
        )
        return response['embedding']
    except Exception:
        return create_mock_vector(text)

# 4. Compute Vector Similarity
def compute_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    dot = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

# 5. RAG Engine Pipeline
class DocQAEngine:
    def __init__(self, api_key):
        self.api_key = api_key
        self.docs_segments = []
        self.vectors = []

    def load_document(self, file_path, segment_size=500, overlap=100):
        print(f"Loading document: {file_path}...")
        raw_text = parse_document(file_path)
        self.docs_segments = segment_text(raw_text, segment_size, overlap)
        
        self.vectors = []
        for segment in self.docs_segments:
            vec = create_vector(segment, self.api_key)
            self.vectors.append(vec)

    def retrieve_segments(self, query, k=3):
        query_vec = create_vector(query, self.api_key)
        results = []
        for i, vec in enumerate(self.vectors):
            score = compute_similarity(query_vec, vec)
            results.append((score, self.docs_segments[i]))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k]

    def query_assistant(self, query, k=3):
        matching_segs = self.retrieve_segments(query, k)
        
        # Offline mode fallback
        if not self.api_key or len(self.api_key) < 15 or "your_" in self.api_key.lower() or self.api_key == "no_key":
            return self._local_grounded_answer(query, matching_segs), matching_segs
            
        context = "\n\n".join([seg for _, seg in matching_segs])
        prompt = f"""You are a helpful assistant. Answer the user query using only the provided context.
If the answer is not in the context, say "I cannot find the answer in the documents".

Context:
{context}

Query: {query}
Answer:"""
        
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel("models/gemini-3.5-flash")
            response = model.generate_content(prompt)
            return response.text, matching_segs
        except Exception:
            return self._local_grounded_answer(query, matching_segs), matching_segs

    def _local_grounded_answer(self, query, matching_segs):
        segments = [seg for _, seg in matching_segs]
        sentences = []
        for seg in segments:
            for sent in seg.split('.'):
                sent = sent.strip()
                if len(sent) > 15:
                    sentences.append(sent)
                    
        query_words = set(query.lower().split())
        stop_words = {'what', 'is', 'the', 'of', 'in', 'and', 'to', 'a', 'for', 'on', 'with', 'about', 'how', 'why', 'does', 'which'}
        query_words = query_words - stop_words
        
        ranked = []
        for sent in sentences:
            sent_words = set(sent.lower().split())
            overlap = len(query_words.intersection(sent_words))
            ranked.append((overlap, sent))
            
        ranked.sort(key=lambda x: x[0], reverse=True)
        best = [sent for score, sent in ranked[:3] if score > 0]
        
        if not best:
            best = [s for s in sentences[:3]]
            
        parts = []
        for sent in best:
            src = 1
            for idx, seg in enumerate(segments):
                if sent in seg:
                    src = idx + 1
                    break
            parts.append(f"{sent}. [Source {src}]")
            
        return " ".join(parts)
