import os, glob, pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from src.config import LOG_DIR, CACHE_DIR, EMBEDDING_MODEL

def load_logs():
    files = glob.glob(f"{LOG_DIR}/*.txt")
    docs = []
    for file in files:
        with open(file, 'r', encoding="utf-8", errors="ignore") as f:
            lines = [line for line in f if "ERROR" in line or "Exception" in line]
        content = "\n".join(lines)
        if content.strip():
            docs.append(Document(page_content=content, metadata={"source": os.path.basename(file)}))
        else:
            print(f"No errors found in {file}, skipping.")
    return docs

def chunk_logs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    return splitter.split_documents(docs)

def build_faiss_index(chunks):
    texts = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        if len(chunk.page_content.strip()) > 50:
            text = f"[Service: {source}]\n{chunk.page_content}"
            texts.append(text)

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(CACHE_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(CACHE_DIR, "faiss.index"))
    with open(os.path.join(CACHE_DIR, "texts.pkl"), "wb") as f:
        pickle.dump(texts, f)

    return index, texts, model

def load_cached_index():
    index = faiss.read_index(os.path.join(CACHE_DIR, "faiss.index"))
    with open(os.path.join(CACHE_DIR, "texts.pkl"), "rb") as f:
        texts = pickle.load(f)
    model = SentenceTransformer(EMBEDDING_MODEL)
    return index, texts, model