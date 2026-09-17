import csv
import random
import re
import ollama
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Reuse the same chunking config the production pipeline actually indexes
# with, so ground-truth chunks share boundaries with what retrieval can
# possibly return. Falls back to pipeline.py's own defaults if these
# aren't defined in config.py.
from config import GlobalConfig

DEFAULT_CHUNK_SIZE = getattr(GlobalConfig, "CHUNK_SIZE_RANGE", [800])[0]
DEFAULT_CHUNK_OVERLAP = getattr(GlobalConfig, "CHUNK_OVERLAP_RANGE", [150])[0]


def load_and_chunk_corpus(
    file_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
):
    """Reads the .txt file and splits it using the SAME splitter the
    retrieval pipeline indexes with (RecursiveCharacterTextSplitter).

    This used to be a naive fixed-width character slice
    (text[start:end], stepping by chunk_size - chunk_overlap), which
    ignored sentence/paragraph structure entirely and -- more
    importantly -- produced different chunk boundaries than
    pipeline.py's load_txt_and_build_vectorstore() actually indexes.
    That mismatch meant a retrieval could return the semantically
    correct passage and still register as a "miss" in evaluation,
    because the exact ground_truth_chunk string was cut from different
    boundaries than what's sitting in the vectorstore. Matching the
    splitter here removes that source of measurement noise.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    raw_chunks = splitter.split_text(text)

    # Keep the same "ignore tiny trailing snippets" guard as before
    chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 100]

    return chunks


def generate_targeted_question(chunk: str, model_name: str) -> str:
    """Prompts Ollama to output a specific question directly targeting the passage

    phrased in the second person ("you").
    """
    system_prompt = (
        "You are an expert retrieval benchmark creator. Your task is to write a single, highly specific question "
        "that can ONLY be answered by reading the provided text chunk.\n\n"
        "STRICT PHRASING RULE:\n"
        "- Address the subject directly using SECOND-PERSON pronouns ('you', 'your').\n"
        "- NEVER use third-person terms like 'the narrator', 'the author', 'the speaker', 'he', or 'she'.\n"
        "- Example: Write 'Why do you feel hesitant...' instead of 'Why does the narrator feel hesitant...'."
    )

    user_prompt = (
        f"TEXT CHUNK:\n{chunk}\n\n"
        "INSTRUCTIONS:\n"
        "1. Write EXACTLY ONE specific question addressing 'you'.\n"
        "2. Do NOT mention 'the text', 'the passage', or 'the author'.\n"
        "3. Output ONLY the raw question, nothing else."
    )

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.3},
        )

        content = response["message"]["content"].strip()

        # Clean reasoning/thinking tags (e.g., <think>...</think>)
        content = re.sub(
            r"<think>.*?</think>", "", content, flags=re.DOTALL
        ).strip()

        # Extract first non-empty line
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        question = lines[0] if lines else ""

        # Post-processing regex fallback to convert remaining third-person references
        question = re.sub(
            r"\b(does|did) the (narrator|author|speaker)\b",
            "do you",
            question,
            flags=re.IGNORECASE,
        )
        question = re.sub(
            r"\bthe (narrator|author|speaker)\b",
            "you",
            question,
            flags=re.IGNORECASE,
        )

        return question

    except Exception as e:
        print(f"\n[ERROR] Failed to generate question: {e}")
        return ""


def create_rag_evaluation_dataset(
    file_path: str,
    output_csv: str = "Data/rag_evaluation_dataset.csv",
    num_samples: int = 500,
    model_name: str = "llama3.2:3b",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    print(f"[INFO] Reading corpus from '{file_path}'...")
    print(
        f"[INFO] Chunking with RecursiveCharacterTextSplitter "
        f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap}) -- "
        "matches pipeline.py's indexing splitter."
    )
    chunks = load_and_chunk_corpus(
        file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    total_chunks = len(chunks)
    print(
        f"[INFO] Created {total_chunks} total chunks from text corpus."
    )

    if total_chunks < num_samples:
        print(
            f"[WARNING] Corpus yielded {total_chunks} chunks, which is less than requested {num_samples}. "
            f"Sampling all {total_chunks} available chunks."
        )
        selected_chunks = chunks
    else:
        selected_chunks = random.sample(chunks, num_samples)

    print(
        f"[INFO] Generating {len(selected_chunks)} ground-truth question pairs using '{model_name}'...\n"
    )

    generated_count = 0
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "question", "ground_truth_chunk"])

        # Wrap iteration with tqdm progress bar
        pbar = tqdm(
            enumerate(selected_chunks, 1),
            total=len(selected_chunks),
            desc="Generating Questions",
            unit="pair",
        )

        for idx, chunk in pbar:
            question = generate_targeted_question(chunk, model_name=model_name)

            if question:
                generated_count += 1
                writer.writerow([generated_count, question, chunk])
                csv_file.flush()  # Stream disk write per iteration

                # Update progress bar status description
                pbar.set_postfix({"Saved": generated_count})

    print(
        f"\n[SUCCESS] Saved {generated_count} Question/Chunk pairs to '{output_csv}'."
    )


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # NOTE: point this at your OCR-cleaned corpus (see ocr_cleanup.py) if
    # you've run that step -- ground truth generated from noisy raw OCR
    # text will bake the same artifacts into your questions/answers.
    TEXT_FILE_PATH = "Data/combined_corpus.txt"  # Path to your .txt corpus
    OUTPUT_CSV_PATH = "Data/rag_test.csv"
    OLLAMA_MODEL = "llama3.2:3b"  # Ollama model for question generation
    TOTAL_QUESTIONS = 500

    create_rag_evaluation_dataset(
        file_path=TEXT_FILE_PATH,
        output_csv=OUTPUT_CSV_PATH,
        num_samples=TOTAL_QUESTIONS,
        model_name=OLLAMA_MODEL,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )