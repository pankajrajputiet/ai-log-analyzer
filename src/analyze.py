import os
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from src.config import GROQ_API_KEY, REPORT_DIR
from src.ingest import load_logs

# Initialize Groq LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b"
)

# Prompt template for per-log analysis
prompt_template = PromptTemplate.from_template("""
Analyze the following log snippet and generate a single Markdown row:

| Service Name | Data Center Name | Error | Expected Solution |

Instructions:
- Extract the **Service Name** from the file name or log content.
- Extract the **Data Center Name** from patterns like 'DC=', 'datacenter='.
- Identify the **Error** from the log messages.
- Suggest an **Expected Solution**.

Log:
{context}
""")

# Combine prompt and LLM into a chain
chain = prompt_template | llm

def compress_chunk(text, max_lines=50):
    lines = text.splitlines()
    filtered = [line for line in lines if "ERROR" in line or "Exception" in line or "DC=" in line or "datacenter=" in line]
    return "\n".join(filtered[:max_lines]) if filtered else text[:2000]  # fallback to first 2000 chars

def generate_report(index=None, texts=None, model=None):
    docs = load_logs()
    rows = []

    for doc in docs:
        compressed = compress_chunk(doc.page_content)
        if not compressed.strip():
            continue

        try:
            result = chain.invoke({"context": compressed})
            row = result.strip() if isinstance(result, str) else str(result)
            if row:
                rows.append(row)
        except Exception as e:
            print(f"⚠️ Error processing {doc.metadata['source']}: {e}")

    if not rows:
        print("⚠️ No errors found in any log files.")
        return False

    # Assemble Markdown table
    header = "| Service Name | Data Center Name | Error | Expected Solution |\n|--------------|------------------|-------|-------------------|"
    table = "\n".join([header] + rows)

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "error_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(table)

    print(f"✅ Full report generated: {report_path}")
    return True