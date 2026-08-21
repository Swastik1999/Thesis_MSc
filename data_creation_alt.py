import json
import os


def append_contents_to_txt(
    json_path: str, output_txt_path: str, separator: str = "\n\n"
):
    """Extracts the 'contents' string from each object in a JSON file

    and appends them to the end of a text file.
    """
    if not os.path.exists(json_path):
        print(f"[ERROR] Could not find file: {json_path}")
        return

    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert single object to list if necessary
    if isinstance(data, dict):
        data = [data]

    # Extract non-empty contents node values
    extracted_contents = []
    for item in data:
        content_text = item.get("contents", "").strip()
        if content_text:
            extracted_contents.append(content_text)

    if not extracted_contents:
        print("[WARNING] No 'contents' field found in the provided JSON.")
        return

    # Combine extracted entries with the separator
    text_to_append = separator.join(extracted_contents)

    # Check if target text file already exists
    file_exists = os.path.exists(output_txt_path)

    # Append to file
    with open(output_txt_path, "a", encoding="utf-8") as f:
        if file_exists:
            f.write(separator)  # Ensures space between existing content and new text
        f.write(text_to_append)

    print(
        f"[SUCCESS] Appended {len(extracted_contents)} contents node(s) to '{output_txt_path}'"
    )


if __name__ == "__main__":
    # Specify your file paths here
    JSON_FILE = "Data/CWMG.json"
    TARGET_TXT_FILE = "Data/combined_corpus.txt"

    append_contents_to_txt(JSON_FILE, TARGET_TXT_FILE)