import glob
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings  
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import CHROMA_DB_DIR, LOG_DIR


def load_logs():
    """
    Load all .txt log files from LOG_DIR and return them as a list of Document objects.
    """
    # Find all text files inside the log directory
    files = glob.glob(f"{LOG_DIR}/*.txt")
    
    docs = []
    for file in files:
        # Open each log file and read its content
        with open(file, 'r') as f:
            content = f.read()
        
        # Wrap content inside a LangChain Document with metadata
        docs.append(Document(
            page_content=content,
            metadata={"source": file}
        ))

    return docs


def chunk_and_store(docs):
    """
    Split the documents into smaller chunks, embed them, 
    and store them in a persistent Chroma database.
    """
    # Split documents into smaller chunks for better embeddings and retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Max characters per chunk
        chunk_overlap=100,    # Overlap between chunks (for context preservation)
        separators=["\n\n", "\n", " ", ""]  # How to split text
    )
    
    # Break the documents into chunks
    chunks = splitter.split_documents(docs)

    # Load sentence-transformer embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store chunks in Chroma vector database (with persistence enabled)
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    # Save the database to disk
    db.persist()
