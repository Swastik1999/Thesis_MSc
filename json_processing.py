import json
import os


def format_json_record_to_text(item: dict) -> str:
    """Formats a single JSON item into a structured text entry suitable

    for dense text embeddings and unified chunking pipelines.
    """
    # 1. Resolve footnotes directly into text body
    contents = item.get("contents", "")
    footnotes = item.get("footnotes", {})

    if isinstance(footnotes, dict):
        for key, val in footnotes.items():
            contents = contents.replace(f"{{{key}}}", f"[{key}]")

        if footnotes:
            footnote_lines = [f"[{k}] {v}" for k, v in footnotes.items()]
            contents += "\n\nFOOTNOTES / EXPLANATORY NOTES:\n" + "\n".join(
                footnote_lines
            )

    # 2. Render clean key-value metadata block
    metadata_block = f"""[METADATA]
Title: {item.get('title', 'N/A')}
Document Type: {item.get('document_type', 'N/A')}
Date: {item.get('document_date', 'N/A')}
Volume: {item.get('volume', 'N/A')} | Section: {item.get('section', 'N/A')}
Source: {item.get('source', 'N/A')}
Original Language: {item.get('original_language', 'N/A')}"""

    # 3. Assemble document text
    text_entry = f"""========================================
{metadata_block}

[CONTENT]
{contents.strip()}
========================================"""

    return text_entry


def append_json_to_txt_corpus(
    json_path: str, output_txt_path: str, append_mode: bool = True
):
    """Reads JSON records and writes/appends them to a single plain-text corpus file."""
    if not os.path.exists(json_path):
        print(f"[ERROR] JSON file not found at: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    formatted_entries = [format_json_record_to_text(item) for item in data]
    corpus_block = "\n\n".join(formatted_entries)

    mode = "a" if append_mode and os.path.exists(output_txt_path) else "w"

    os.makedirs(os.path.dirname(os.path.abspath(output_txt_path)), exist_ok=True)

    with open(output_txt_path, mode, encoding="utf-8") as f:
        if mode == "a":
            f.write("\n\n")  # Spacing before appended data
        f.write(corpus_block)

    print(
        f"[SUCCESS] Appended {len(data)} JSON records to text corpus at '{output_txt_path}'"
    )


if __name__ == "__main__":
    # Settings
    JSON_INPUT = "Data/CWMG.json"
    CORPUS_TXT = "Data/combined_corpus.txt"

    # Set append_mode=True to merge into an existing text file,
    # or append_mode=False to overwrite/create fresh.
    append_json_to_txt_corpus(JSON_INPUT, CORPUS_TXT, append_mode=True)