#!/usr/bin/env python3
"""
Generate answers for a QA dataset using a local Ollama model, grounded in a
ground-truth chunk.

Input CSV  : id,question,ground_truth_chunk
Output CSV : id,question,generated_answer,truth_chunk

Requires only `requests` (pip install requests) and a running Ollama server.

Edit the CONFIG block below to set the input/output paths and model, then:

    ollama serve                 # if it isn't already running
    ollama pull llama3.1:8b
    python generate_answers.py

Every setting in CONFIG can still be overridden on the command line for one-off
runs, e.g.:

    python generate_answers.py --limit 5                 # smoke test
    python generate_answers.py --resume                  # continue after a crash
    python generate_answers.py -o answers_qwen.csv --model qwen2.5:14b
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

# ---------------------------------------------------------------------------
# CONFIG - edit these
# ---------------------------------------------------------------------------

# Paths are resolved relative to this script's own location, so the script works
# no matter which directory you run it from. Use an absolute path
# (e.g. "/home/me/data/questions.csv") if your files live elsewhere.
INPUT_CSV = "Data/extracted_file.csv"
OUTPUT_CSV = "Data/answers_top_questions.csv"

MODEL = "deepseek-r1:latest"  # Ollama model for answer generation
HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

TEMPERATURE = 0.0      # 0.0 for reproducible evaluation runs
SEED = 42
NUM_CTX = 4096         # context window; raise it if your chunks get truncated
NUM_PREDICT = 512      # max tokens to generate
TIMEOUT = 300          # per-request seconds
WORKERS = 1            # parallel requests; also set OLLAMA_NUM_PARALLEL on the server
PROMPT_FILE = None     # optional path to a prompt template file

# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))


def resolve(path):
    """Make a relative path relative to this script rather than the CWD."""
    return path if os.path.isabs(path) else os.path.join(HERE, path)

SYSTEM_PROMPT = """\
You answer questions using only the context you are given. Answer directly and concisely, and never refer to "the context" or \
"the passage" in your answer."""

DEFAULT_PROMPT = """\
Context:
You are acting as Mahatma Gandhi. The ground truth context is provided below. The ground truth is written by the person you are impersonating. You should answer the question using only the context provided.
{ground_truth_chunk}
\"\"\"
Answer the question below using only the context and ground truth above.
\"\"\"

Question: {question}

Answer using only the context above. If the context does not contain enough
information to answer, reply with exactly:
"The provided context does not contain enough information to answer this question."
"""

IN_FIELDS = ["id", "question", "ground_truth_chunk"]
OUT_FIELDS = ["id", "question", "generated_answer", "truth_chunk"]

# csv fields can be long chunks of text
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in IN_FIELDS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(
                f"Input CSV is missing required column(s): {', '.join(missing)}\n"
                f"Found: {reader.fieldnames}"
            )
        return [dict(r) for r in reader]


def already_done(path):
    """Return the set of ids already present in the output file."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"] for r in csv.DictReader(f) if r.get("id")}


def check_server(host, model):
    """Fail early and loudly rather than after 500 timeouts."""
    try:
        tags = requests.get(f"{host}/api/tags", timeout=10).json()
    except requests.RequestException as e:
        raise SystemExit(f"Cannot reach Ollama at {host}: {e}\nIs `ollama serve` running?")

    names = [m["name"] for m in tags.get("models", [])]
    # a tagless name like "llama3.1" matches the pulled "llama3.1:latest"
    if model not in names and f"{model}:latest" not in names:
        raise SystemExit(
            f"Model '{model}' is not available on {host}.\n"
            f"Pull it with: ollama pull {model}\n"
            f"Available: {', '.join(names) or '(none)'}"
        )


def generate(session, host, model, prompt_template, row, options, timeout, retries=3):
    prompt = prompt_template.format(
        question=row["question"],
        ground_truth_chunk=row["ground_truth_chunk"],
    )
    payload = {
        "model": model,
        "stream": False,
        "options": options,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }

    for attempt in range(retries):
        try:
            r = session.post(f"{host}/api/chat", json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            if attempt == retries - 1:
                return f"ERROR: {type(e).__name__}: {e}"
            time.sleep(2 ** attempt)  # 1s, 2s, 4s
    return "ERROR: exhausted retries"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Every default comes from the CONFIG block at the top of this file; the
    # flags exist only so you can override one for a single run.
    ap.add_argument("-i", "--input", default=INPUT_CSV)
    ap.add_argument("-o", "--output", default=OUTPUT_CSV)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--prompt-file", default=PROMPT_FILE,
                    help="file containing the prompt template; "
                         "must use {question} and {ground_truth_chunk}")
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--num-ctx", type=int, default=NUM_CTX)
    ap.add_argument("--num-predict", type=int, default=NUM_PREDICT)
    ap.add_argument("--timeout", type=int, default=TIMEOUT)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--limit", type=int, help="only process the first N rows")
    ap.add_argument("--resume", action="store_true",
                    help="append to output, skipping ids already written")
    args = ap.parse_args()

    in_path = resolve(args.input)
    out_path = resolve(args.output)
    if not os.path.exists(in_path):
        raise SystemExit(f"Input CSV not found: {in_path}\n"
                         f"Set INPUT_CSV at the top of {os.path.basename(__file__)}.")
    print(f"Input : {in_path}\nOutput: {out_path}\nModel : {args.model}\n")

    host = args.host.rstrip("/")
    check_server(host, args.model)

    prompt_template = DEFAULT_PROMPT
    if args.prompt_file:
        with open(resolve(args.prompt_file), encoding="utf-8") as f:
            prompt_template = f.read()

    options = {
        "temperature": args.temperature,
        "seed": args.seed,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
    }

    rows = read_rows(in_path)
    if args.limit:
        rows = rows[:args.limit]

    done = already_done(out_path) if args.resume else set()
    if done:
        rows = [r for r in rows if r["id"] not in done]
        print(f"Resuming: {len(done)} rows already done, {len(rows)} remaining.")

    if not rows:
        print("Nothing to do.")
        return

    session = requests.Session()
    lock = threading.Lock()
    counter = {"n": 0}
    started = time.time()

    append = args.resume and os.path.exists(out_path)
    mode = "a" if append else "w"

    with open(out_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        if not append:
            writer.writeheader()

        def work(row):
            answer = generate(session, host, args.model, prompt_template,
                              row, options, args.timeout)
            with lock:
                writer.writerow({
                    "id": row["id"],
                    "question": row["question"],
                    "generated_answer": answer,
                    "truth_chunk": row["ground_truth_chunk"],
                })
                f.flush()  # checkpoint after every row
                counter["n"] += 1
                rate = counter["n"] / (time.time() - started)
                print(f"[{counter['n']}/{len(rows)}] id={row['id']} "
                      f"({rate:.2f} rows/s)", flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, rows))

    print(f"\nWrote {counter['n']} rows to {out_path} "
          f"in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
