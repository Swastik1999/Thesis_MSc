import json
import re
import time
from pathlib import Path

import ollama


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "Data/combined_corpus.txt"
OUTPUT_FILE = "Data/rag_evaluation_dataset_2.json"

# Ollama model used to generate questions/answers
MODEL = "llama3.1:8b"

# Approximate number of characters per chunk
CHUNK_SIZE = 600

# Overlap between chunks
CHUNK_OVERLAP = 200

# Number of QA pairs to generate per chunk
QUESTIONS_PER_CHUNK = 1

# Ollama generation temperature
TEMPERATURE = 0.2

# Maximum number of questions in the final dataset
MAX_QUESTIONS = 500


# ============================================================
# TEXT PROCESSING
# ============================================================

def load_text(file_path):
    """Load the source text."""

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def normalize_text(text):
    """Normalize whitespace while preserving paragraph structure."""

    # Normalize Windows line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse more than two newlines into two
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_into_chunks(text, chunk_size=4000, overlap=500):
    """
    Split text into overlapping chunks.

    The split tries to occur at paragraph boundaries.
    """

    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # If adding the paragraph stays within the limit
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            current_chunk += paragraph + "\n\n"

        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Start a new chunk
            current_chunk = paragraph + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Add overlap
    final_chunks = []

    for i, chunk in enumerate(chunks):

        if i > 0:
            previous = chunks[i - 1]

            overlap_text = previous[-overlap:]

            chunk = overlap_text + "\n\n" + chunk

        final_chunks.append(chunk)

    return final_chunks


# ============================================================
# LLM GENERATION
# ============================================================

def generate_qa_pairs(context, num_questions=3):
    """
    Ask Ollama to generate QA pairs from a context passage.
    """

    prompt = f"""
You are creating a high-quality evaluation dataset for a Retrieval-Augmented
Generation (RAG) system.

Read the following source passage carefully.

SOURCE PASSAGE:
----------------
{context}
----------------

Generate exactly {num_questions} question-answer pairs.

Requirements:

1. Every question MUST be answerable using ONLY the source passage.
2. Write the question directly asking the narrator or writer of the passage. Example, instead of asking "How long was the narrator imprisioned?", ask "How long were you imprisoned?".
3. Questions should test factual understanding of the passage.
4. Avoid trivial questions such as "What is this passage about?"
5. Prefer questions that require identifying specific facts, relationships,
   events, people, dates, explanations, or claims.
6. The answer must be directly supported by the passage.
7. Do not invent information.
8. Answers should be concise but complete.
9. Do not use outside information.
10. If the passage does not contain enough information for a question,
    do not create that question.

Return ONLY valid JSON.

Use exactly this format:

{{
    "questions": [
        {{
            "question": "Question here",
            "answer": "Answer here"
        }}
    ]
}}
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": TEMPERATURE
        }
    )

    return response["message"]["content"]


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):
    """
    Extract JSON from an LLM response.

    Handles cases where the model accidentally adds
    markdown fences around the JSON.
    """

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)

    # Find the JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM response.")

    json_text = text[start:end + 1]

    return json.loads(json_text)


# ============================================================
# VALIDATION
# ============================================================

def validate_qa_pair(pair, context):
    """
    Basic validation of generated QA pairs.

    This does not prove that the answer is correct.
    It only checks basic structural requirements.
    """

    if not isinstance(pair, dict):
        return False

    question = pair.get("question")
    answer = pair.get("answer")

    if not question or not answer:
        return False

    if len(question.strip()) < 10:
        return False

    if len(answer.strip()) < 2:
        return False

    return True


# ============================================================
# MAIN DATASET GENERATION
# ============================================================


def generate_dataset():

    print("Loading source text...")

    text = load_text(INPUT_FILE)

    text = normalize_text(text)

    print(f"Loaded {len(text):,} characters.")

    print("Splitting text into chunks...")

    chunks = split_into_chunks(
        text,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    print(f"Created {len(chunks)} chunks.")

    dataset = []

    question_id = 1

    for chunk_index, chunk in enumerate(chunks):

        # Stop once we reach the maximum number of questions
        if len(dataset) >= MAX_QUESTIONS:
            print(
                f"\nReached maximum limit of "
                f"{MAX_QUESTIONS} questions."
            )
            break

        print(
            f"\nProcessing chunk "
            f"{chunk_index + 1}/{len(chunks)}..."
        )

        try:

            # Calculate how many questions are still needed
            remaining_questions = MAX_QUESTIONS - len(dataset)

            # Do not request more questions than necessary
            questions_to_generate = min(
                QUESTIONS_PER_CHUNK,
                remaining_questions
            )

            raw_response = generate_qa_pairs(
                chunk,
                questions_to_generate
            )

            result = extract_json(raw_response)

            questions = result.get("questions", [])

            print(
                f"Generated {len(questions)} QA pairs."
            )

            for qa in questions:

                # Stop adding questions if limit is reached
                if len(dataset) >= MAX_QUESTIONS:
                    break

                if not validate_qa_pair(qa, chunk):
                    print("Skipping invalid QA pair.")
                    continue

                dataset.append(
                    {
                        "id": f"q{question_id:04d}",

                        "question": qa["question"].strip(),

                        "ground_truth": qa["answer"].strip(),

                        # The original context containing
                        # the answer.
                        "context": [
                            chunk.strip()
                        ],

                        # Useful for retrieval evaluation.
                        "source": Path(INPUT_FILE).name,

                        "chunk_id": chunk_index
                    }
                )

                question_id += 1

        except Exception as e:

            print(
                f"ERROR processing chunk "
                f"{chunk_index}: {e}"
            )

        # Small delay to avoid overwhelming Ollama
        time.sleep(0.2)

    # ========================================================
    # SAVE DATASET
    # ========================================================

    output_path = Path(OUTPUT_FILE)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n===================================")
    print("Dataset generation complete!")
    print("===================================")

    print(f"Total questions: {len(dataset)}")
    print(f"Output file: {output_path}")



# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    generate_dataset()