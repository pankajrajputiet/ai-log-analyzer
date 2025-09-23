import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from src.config import GROQ_API_KEY, REPORT_DIR, TOP_K_RESULTS

llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b"
)

prompt_template = PromptTemplate.from_template("""
Analyze the following log snippets and generate a Markdown table with the following columns:

| Service Name | Data Center Name | Error | Expected Solution |

Instructions:
- Extract the **Service Name** from the file name in metadata or from any mention in the log.
- Extract the **Data Center Name** from the log content (look for patterns like 'DC=', 'datacenter=', etc.).
- Identify the **Error** from the log messages.
- Suggest an **Expected Solution** based on the error.

Logs:
{context}
""")

qa_chain = LLMChain(llm=llm, prompt=prompt_template)

def search_similar_chunks(query, index, texts, model, top_k=TOP_K_RESULTS):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [texts[i] for i in indices[0]]

def compress_chunk(text):
    lines = text.splitlines()
    return "\n".join([line for line in lines if "ERROR" in line or "Exception" in line or "DC=" in line])

def generate_report(index, texts, model):
    query = "Analyze logs for errors and suggest fixes"
    relevant_chunks = search_similar_chunks(query, index, texts, model)
    compressed = [compress_chunk(c) for c in relevant_chunks]
    context = "\n\n".join(compressed)

    result = qa_chain.invoke({"context": context})
    result_text = result["text"] if isinstance(result, dict) else str(result)

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "error_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(result_text)

    print(f"✅ Report generated: {report_path}")