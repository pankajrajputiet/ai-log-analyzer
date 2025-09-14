import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from src.config import GROQ_API_KEY, CHROMA_DB_DIR, REPORT_DIR


# Instantiate Groq Large Language Model (LLM)
llm = ChatGroq(
    temperature=0,                 # Deterministic output (no randomness)
    groq_api_key=GROQ_API_KEY,     # API key from config
    model_name="openai/gpt-oss-120b"  # ✅ Choose a supported model
)


def generate_report():
    """
    Generate an error analysis report from logs stored in Chroma DB.
    Steps:
    1. Load embeddings model.
    2. Load Chroma vector database.
    3. Build a retrieval-based QA chain.
    4. Ask the LLM to analyze logs for errors and solutions.
    5. Save results into a report file.
    """

    # 1. Load Hugging Face embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. Load Chroma vector store (persistent database of embeddings)
    db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

    # 3. Create retriever and QA chain
    retriever = db.as_retriever()  # Retriever to fetch relevant log chunks
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,                     # LLM to generate answers
        retriever=retriever,         # Retriever for context
        chain_type="stuff"           # Chain type: stuff = simple context stuffing
    )

    # 4. Run query on the logs
    query = "Analyze the logs and list all errors with appropriate solutions."
    result = qa_chain.invoke(query)

    # 5. Extract response text
    if isinstance(result, dict):
        result_text = result.get("result", str(result))
    else:
        result_text = str(result)

    # 6. Save report to file
    os.makedirs(REPORT_DIR, exist_ok=True)  # Ensure directory exists
    report_path = os.path.join(REPORT_DIR, "error_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(result_text)
        
        
