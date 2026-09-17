"""
ocr_cleanup.py

Cleans up OCR artifacts in a .txt corpus before chunking/indexing.

Split into two tiers, deliberately:

  TIER 1 (SAFE, always applied automatically)
    Structural fixes with essentially zero risk of introducing new
    errors: whitespace normalization, dehyphenation across line breaks,
    stray junk-character stripping, quote/dash normalization.

  TIER 2 (FLAG FOR REVIEW, not auto-applied by default)
    Words that don't match a standard English dictionary get logged to
    a report file (word, count, example context) instead of being
    silently "corrected". OCR substitution errors (e.g. "closs" for
    "close", "n#t" for "n't") are unpredictable and proper nouns /
    archaic spellings / non-English names in a text like this WILL
    trip a naive spellchecker -- auto-correcting blindly risks quietly
    replacing a correct word with a wrong one, which is worse than
    leaving visibly-garbled text alone. Review the report, then decide
    whether to fix specific words by hand or enable --auto-correct for
    an aggressive (but logged, reversible-via-diff) pass.

Usage:
    python ocr_cleanup.py input.txt output.txt
    python ocr_cleanup.py input.txt output.txt --report review.txt
    python ocr_cleanup.py input.txt output.txt --report review.txt --auto-correct
"""

import argparse
import re
from collections import Counter
from pathlib import Path


# =====================================================================
# TIER 1: Safe structural cleanup
# =====================================================================

def dehyphenate(text: str) -> str:
    """Joins words split across a line break with a hyphen or the OCR
    soft-hyphen artifact '¬' (seen in this corpus, e.g. 'pro¬\\ngramme').
    'pro¬\\ngramme' -> 'programme', 'pro-\\ngramme' -> 'programme'.
    """
    return re.sub(r"(\w+)[¬-]\s*\n\s*(\w+)", r"\1\2", text)


def strip_junk_characters(text: str) -> str:
    """Removes characters that are near-certainly OCR noise rather than
    real punctuation/content: stray soft-hyphen remnants, guillemets
    used as paragraph-start artifacts, and other control-ish glyphs
    that showed up in this corpus. Conservative -- only characters with
    no legitimate use in this kind of English prose are touched.
    """
    # Leftover soft-hyphen / line-break artifacts that dehyphenate() didn't catch
    text = text.replace("¬", "")
    # Stray guillemets/bullet-like marks used as paragraph-start artifacts in this OCR
    # (e.g. "« The resolutions", "■who was then working")
    text = re.sub(r"[«»■□❑]\s*", "", text)
    return text


def normalize_quotes_and_dashes(text: str) -> str:
    """Standardizes curly quotes/dashes to plain ASCII equivalents so
    downstream tokenization (SPLADE, embeddings) doesn't fragment on
    inconsistent Unicode variants."""
    replacements = {
        "\u2018": "'", "\u2019": "'",   # ' '
        "\u201c": '"', "\u201d": '"',   # " "
        "\u2013": "-", "\u2014": "-",   # – —
        "\u00a0": " ",                    # non-breaking space
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapses runs of spaces/tabs to one space, and runs of 3+
    newlines down to a clean paragraph break (\\n\\n). Leaves single
    newlines and existing \\n\\n paragraph breaks alone so real
    structure (where it exists) is preserved for the text splitter."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces at end of each line
    text = re.sub(r" +\n", "\n", text)
    return text.strip()


def strip_footnote_markers(text: str) -> str:
    """Removes inline footnote reference markers like '{1}' or '{23}'.
    Optional (--strip-footnotes) because these may carry citation
    meaning you want to keep for other purposes -- but for embedding/
    retrieval they're usually just noise interrupting sentence flow."""
    return re.sub(r"\{\d+\}", "", text)


def tier1_clean(text: str, strip_footnotes: bool = False) -> str:
    text = dehyphenate(text)
    text = strip_junk_characters(text)
    text = normalize_quotes_and_dashes(text)
    if strip_footnotes:
        text = strip_footnote_markers(text)
    text = normalize_whitespace(text)
    return text


# =====================================================================
# TIER 2: Flag-for-review (and optional auto-correct)
# =====================================================================

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def build_symspell():
    """Loads symspellpy with its bundled English frequency dictionary.
    Returns None if symspellpy isn't installed -- flagging still works
    via a fallback, just without fuzzy suggestions."""
    try:
        from symspellpy import SymSpell
        import importlib.resources as importlib_resources

        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dict_path = str(
            importlib_resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
        )
        sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
        return sym_spell
    except ImportError:
        return None


def flag_unknown_words(text: str, sym_spell, context_chars: int = 40):
    """Scans for words not in the dictionary and not obviously a proper
    noun (capitalized mid-sentence words are skipped -- names like
    'Porbandar' or 'Gandhi' shouldn't be flagged as errors). Returns a
    Counter of unknown words and a dict of one example context per word.
    """
    unknown_counts = Counter()
    example_context = {}

    for m in WORD_RE.finditer(text):
        word = m.group(0)
        if len(word) < 3:
            continue
        # Skip capitalized words that aren't sentence-initial -- likely
        # proper nouns (names, places), not OCR errors.
        start = m.start()
        is_sentence_start = start == 0 or text[max(0, start - 2):start].strip() in ("", ".", "!", "?")
        if word[0].isupper() and not is_sentence_start:
            continue

        lookup_word = word.lower()
        if sym_spell is not None:
            from symspellpy import Verbosity
            suggestions = sym_spell.lookup(lookup_word, Verbosity.TOP, max_edit_distance=0)
            known = len(suggestions) > 0
        else:
            known = True  # can't check without symspellpy; skip flagging

        if not known:
            unknown_counts[word] += 1
            if word not in example_context:
                ctx_start = max(0, start - context_chars)
                ctx_end = min(len(text), start + len(word) + context_chars)
                example_context[word] = text[ctx_start:ctx_end].replace("\n", " ")

    return unknown_counts, example_context


def write_review_report(path: str, unknown_counts: Counter, example_context: dict):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"OCR REVIEW REPORT — {len(unknown_counts)} unique flagged words\n")
        f.write("=" * 70 + "\n")
        f.write(
            "These words weren't found in a standard English dictionary.\n"
            "Some are genuinely OCR errors; others are proper nouns, archaic\n"
            "spellings, or non-English terms this dictionary doesn't know\n"
            "(this corpus has plenty of both). Review before trusting any\n"
            "auto-correction of these.\n"
        )
        f.write("=" * 70 + "\n\n")
        for word, count in unknown_counts.most_common():
            f.write(f"[{count:>3}x] {word!r}\n")
            f.write(f"       ...{example_context[word]}...\n\n")


def auto_correct(text: str, sym_spell, unknown_counts: Counter) -> str:
    """Aggressively replaces flagged words with symspell's top fuzzy
    suggestion, but ONLY for words appearing more than once (single
    occurrences are more likely to be a real rare/proper word than a
    repeated OCR substitution pattern) and only within edit distance 2.
    Still risky -- read the review report first."""
    from symspellpy import Verbosity

    corrections = {}
    for word in unknown_counts:
        if unknown_counts[word] < 2:
            continue
        suggestions = sym_spell.lookup(word.lower(), Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrections[word] = suggestions[0].term

    def replace(m):
        w = m.group(0)
        return corrections.get(w, w)

    return WORD_RE.sub(replace, text)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="OCR cleanup for a .txt corpus")
    parser.add_argument("input", help="Path to the raw .txt file")
    parser.add_argument("output", help="Path to write the cleaned .txt file")
    parser.add_argument(
        "--strip-footnotes", action="store_true",
        help="Remove inline footnote markers like '{1}' -- optional since "
             "they may carry citation info you want to keep elsewhere.",
    )
    parser.add_argument(
        "--report", default=None,
        help="Path to write the flagged-words review report (Tier 2). "
             "If omitted, only Tier 1 safe cleanup runs.",
    )
    parser.add_argument(
        "--auto-correct", action="store_true",
        help="Apply aggressive fuzzy auto-correction to repeated flagged "
             "words (edit distance <=2). Requires --report. Read the "
             "report before trusting this on a real run.",
    )
    args = parser.parse_args()

    raw = Path(args.input).read_text(encoding="utf-8")
    print(f"[INFO] Read {len(raw):,} characters from {args.input}")

    cleaned = tier1_clean(raw, strip_footnotes=args.strip_footnotes)
    print(f"[INFO] Tier 1 structural cleanup done -> {len(cleaned):,} characters")

    if args.report:
        sym_spell = build_symspell()
        if sym_spell is None:
            print("[WARNING] symspellpy not installed -- skipping flag pass. "
                  "pip install symspellpy to enable.")
        else:
            unknown_counts, example_context = flag_unknown_words(cleaned, sym_spell)
            write_review_report(args.report, unknown_counts, example_context)
            print(f"[INFO] Flagged {len(unknown_counts)} unique unknown words -> {args.report}")

            if args.auto_correct:
                cleaned = auto_correct(cleaned, sym_spell, unknown_counts)
                print("[INFO] Auto-correct applied to repeated flagged words "
                      "(check a diff against the pre-auto-correct output before trusting this).")

    Path(args.output).write_text(cleaned, encoding="utf-8")
    print(f"[SUCCESS] Wrote cleaned text to {args.output}")


if __name__ == "__main__":
    main()
