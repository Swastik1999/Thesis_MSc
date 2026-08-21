import json
import os
import re
from typing import Dict, List
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# =====================================================================
# 1. LLM PROMPT FOR SYNTHETIC QUERY GENERATION
# =====================================================================
SYNTHETIC_PROMPT_TEMPLATE = """You are an expert data annotator building an evaluation benchmark for a RAG system.

Read the following document entry carefully:
----------------------------------------
{doc_text}
----------------------------------------

Generate 100 distinct, highly specific questions that can ONLY be answered using the information in this document.

Requirements:
1. Make questions natural, as a real researcher or user would ask.
2. Ensure the question includes specific entity names, dates, or details mentioned in the text.
3. Output strictly valid JSON matching this structure (no markdown formatting, no extra text):

[
  {{
    "query": "Your generated question 1 here?",
    "ground_truth_answer": "Direct factual answer extracted from the content."
  }},
  {{
    "query": "Your generated question 2 here?",
    "ground_truth_answer": "Direct factual answer extracted from the content."
  }}
]

"""


# =====================================================================
# 2. HELPER FUNCTIONS
# =====================================================================
def parse_txt_corpus(txt_path: str) -> List[Dict[str, str]]:
    """Parses a unified .txt corpus separated by '========' boundaries

    and extracts document blocks and metadata.
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Text file not found at: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split documents using delimiter
    raw_blocks = content.split("========================================")
    parsed_docs = []

    for idx, block in enumerate(raw_blocks):
        block_str = block.strip()
        if not block_str or len(block_str) < 50:
            continue

        # Extract title from metadata header if available
        title_match = re.search(r"Title:\s*(.+)", block_str)
        title = (
            title_match.group(1).strip()
            if title_match
            else f"Document {idx + 1}"
        )

        # Extract volume/section to create deterministic ID, or fallback to index
        vol_match = re.search(r"Volume:\s*(\d+)", block_str)
        sec_match = re.search(r"Section:\s*(\d+)", block_str)

        vol = vol_match.group(1) if vol_match else "0"
        sec = sec_match.group(1) if sec_match else str(idx + 1)
        doc_id = f"doc_v{vol}_s{sec}"

        parsed_docs.append(
            {"doc_id": doc_id, "title": title, "full_text": block_str}
        )

    return parsed_docs


def extract_json_from_response(text: str) -> List[Dict[str, str]]:
    """Cleans LLM response and parses JSON array safely."""
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return []


# =====================================================================
# 3. SYNTHETIC DATASET GENERATOR
# =====================================================================
def generate_synthetic_dataset_from_txt(
    txt_corpus_path: str,
    output_eval_path: str,
    model_name: str = "deepseek-r1:latest",
    max_docs: int = None,
):
    """Reads .txt corpus and uses Ollama to generate evaluation pairs."""
    docs = parse_txt_corpus(txt_corpus_path)
    print(f"Parsed {len(docs)} document blocks from '{txt_corpus_path}'.")

    if max_docs:
        docs = docs[:max_docs]

    llm = ChatOllama(model=model_name, temperature=0.2)
    prompt = PromptTemplate.from_template(SYNTHETIC_PROMPT_TEMPLATE)
    chain = prompt | llm

    eval_dataset = []

    print(
        f"Generating synthetic questions using '{model_name}' across {len(docs)} documents...\n"
    )

    for idx, doc in enumerate(docs, start=1):
        print(f"[{idx}/{len(docs)}] Processing Document ID: {doc['doc_id']}...")

        try:
            response = chain.invoke(
                {
                    "doc_text": doc["full_text"][
                        :2000
                    ]  # Cap length for context limit
                }
            )

            parsed_pairs = extract_json_from_response(response.content)

            for pair in parsed_pairs:
                if "query" in pair and "ground_truth_answer" in pair:
                    eval_dataset.append(
                        {
                            "query": pair["query"],
                            "expected_doc_id": doc["doc_id"],
                            "ground_truth_answer": pair["ground_truth_answer"],
                            "source_title": doc["title"],
                        }
                    )
        except Exception as e:
            print(f"  └─ Failed for {doc['doc_id']}: {str(e)}")

    # Save output
    os.makedirs(
        os.path.dirname(os.path.abspath(output_eval_path)), exist_ok=True
    )
    with open(output_eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_dataset, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(
        f"SUCCESS: Generated {len(eval_dataset)} test cases across {len(docs)} documents."
    )
    print(f"Saved evaluation file to: {output_eval_path}")
    print("=" * 60)


# =====================================================================
# 4. EXECUTION ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    TXT_CORPUS_FILE = "Data/combined_corpus.txt"
    EVAL_OUTPUT_JSON = "Data/synthetic_eval_dataset.json"

    # Set max_docs=None to generate over all documents in your .txt file
    generate_synthetic_dataset_from_txt(
        txt_corpus_path=TXT_CORPUS_FILE,
        output_eval_path=EVAL_OUTPUT_JSON,
        model_name="deepseek-r1:latest",
        max_docs=None,
    )