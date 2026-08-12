import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import csv
import json
import argparse
from statistics import mean

from src.extraction import getTextFromPDF
from src.chunking import chunk_text
from src.ranking import rank_chunks
from src.summarization import summarize_chunks
from src.summarization import synthesize_summary
from src.aggregation import aggregate_summary
from src.evaluation import score
from src.evaluation import lead_k
from src.evaluation import extractive_baseline
from src.config import (
    AGGREGATE_MAX_LENGTH,
    AGGREGATE_MIN_LENGTH,
    TOP_N_PER_SECTION
)

# Anchored to this file so the paths don't depend on where you run it from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PAPERS_DIR = os.path.join(DATA_DIR, "papers")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
REFERENCES_PATH = os.path.join(DATA_DIR, "references.json")
RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")
RESULTS_MD = os.path.join(DATA_DIR, "results.md")

LEAD_K = 3

# Headings sometimes latch onto an author name or a stray line, producing a
# tiny "section" whose summary is pure model hallucination. Anything this
# short cannot carry real content, so it never reaches the summarizer.
MIN_SECTION_WORDS = 50

SYSTEMS = ("full", "lead3", "extractive")
METRICS = ("rouge1", "rouge2", "rougeLsum")
PARTS = ("f", "p", "r")

FIELDNAMES = (
    ["arxiv_id", "system", "status"]
    + [f"{metric}_{part}" for metric in METRICS for part in PARTS]
    + ["system_words", "reference_words"]
)


def load_references():
    with open(REFERENCES_PATH, encoding="utf-8") as f:
        return json.load(f)


def summarize_paper(pdf_path):
    sections = getTextFromPDF(pdf_path)

    # LEAKAGE GUARD -- the abstract is the reference, it must never be input.
    for name in [k for k in sections if "abstract" in k]:
        del sections[name]

    for name in [k for k, v in sections.items() if len(v.split()) < MIN_SECTION_WORDS]:
        del sections[name]

    if not sections:
        raise ValueError("no usable sections extracted")

    chunks = chunk_text(sections)
    ranked = rank_chunks(chunks)
    summaries = summarize_chunks(ranked)
    plain = aggregate_summary(summaries, section_order=list(sections.keys()), with_headers=False)
    system = synthesize_summary(plain, AGGREGATE_MAX_LENGTH, AGGREGATE_MIN_LENGTH)

    # sections feeds lead_k, ranked feeds extractive_baseline.
    return sections, ranked, system


def build_systems(arxiv_id):
    pdf_path = os.path.join(PAPERS_DIR, f"{arxiv_id}.pdf")

    sections, ranked, system = summarize_paper(pdf_path)

    # Both baselines are free - no model call - so they cost nothing to carry.
    return {
        "full": system,
        "lead3": lead_k(sections, LEAD_K),
        "extractive": extractive_baseline(ranked, TOP_N_PER_SECTION),
    }


def cached_systems(arxiv_id, use_cache=True):
    path = os.path.join(CACHE_DIR, f"{arxiv_id}.json")

    if use_cache and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    systems = build_systems(arxiv_id)

    # All three land in one file, so a file existing means a finished paper.
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(systems, f, indent=2, ensure_ascii=False)

    return systems


def flatten(result):
    # rouge-score returns {metric: Score(precision, recall, fmeasure)}. Flattening
    # in one place keeps the CSV column names defined exactly once.
    flat = {}
    for metric in METRICS:
        flat[f"{metric}_f"] = result[metric].fmeasure
        flat[f"{metric}_p"] = result[metric].precision
        flat[f"{metric}_r"] = result[metric].recall
    return flat


def evaluate_paper(arxiv_id, reference, use_cache=True):
    systems = cached_systems(arxiv_id, use_cache)

    rows = []
    for name in SYSTEMS:
        text = systems[name]

        row = {"arxiv_id": arxiv_id, "system": name, "status": "ok"}
        row.update(flatten(score(reference, text)))
        row["system_words"] = len(text.split())
        row["reference_words"] = len(reference.split())

        rows.append(row)

    return rows


def failed_row(arxiv_id, status):
    row = {"arxiv_id": arxiv_id, "system": "-", "status": status}
    row.update({f"{metric}_{part}": "" for metric in METRICS for part in PARTS})
    row["system_words"] = 0
    row["reference_words"] = 0
    return row


def run(limit=None, use_cache=True):
    references = load_references()

    arxiv_ids = list(references)
    if limit:
        arxiv_ids = arxiv_ids[:limit]

    rows = []
    succeeded = 0

    for position, arxiv_id in enumerate(arxiv_ids, start=1):
        print(f"[{position}/{len(arxiv_ids)}] {arxiv_id} ", end="", flush=True)

        try:
            paper_rows = evaluate_paper(arxiv_id, references[arxiv_id]["abstract"], use_cache)
        except Exception as error:
            # One bad paper shouldn't cost the whole run.
            print(f"failed ({type(error).__name__})")
            rows.append(failed_row(arxiv_id, type(error).__name__))
            continue

        rows.extend(paper_rows)
        succeeded += 1

        full = next(row for row in paper_rows if row["system"] == "full")
        print(f"ok   R1={full['rouge1_f']:.4f}  {full['system_words']}w "
              f"vs {full['reference_words']}w")

    return rows, succeeded, len(arxiv_ids)


def write_csv(rows):
    # newline="" keeps the csv module from doubling line endings on Windows.
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_report(rows, succeeded, total):
    lines = [
        "# ROUGE Evaluation",
        "",
        f"Reference summaries are the author-written abstracts of {total} arXiv papers. "
        "The abstract is removed from the input before summarization, so no system "
        "sees its own reference.",
        "",
        f"Extraction succeeded on **{succeeded}/{total}** papers.",
        "",
        "| System | ROUGE-1 F | ROUGE-2 F | ROUGE-Lsum F | Avg words |",
        "|---|---|---|---|---|",
    ]

    for name in SYSTEMS:
        scored = [r for r in rows if r["system"] == name and r["status"] == "ok"]
        if not scored:
            continue

        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.0f} |".format(
                name,
                mean(r["rouge1_f"] for r in scored),
                mean(r["rouge2_f"] for r in scored),
                mean(r["rougeLsum_f"] for r in scored),
                mean(r["system_words"] for r in scored),
            )
        )

    reference_words = [r["reference_words"] for r in rows if r["status"] == "ok"]
    if reference_words:
        lines += ["", f"Reference abstracts average {mean(reference_words):.0f} words."]

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Score the summarization pipeline with ROUGE against author abstracts."
    )
    parser.add_argument("--limit", type=int, help="only evaluate the first N papers")
    parser.add_argument("--no-cache", action="store_true",
                        help="regenerate summaries instead of reading data/cache")
    args = parser.parse_args()

    rows, succeeded, total = run(limit=args.limit, use_cache=not args.no_cache)

    write_csv(rows)

    report = build_report(rows, succeeded, total)
    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write(report)

    print()
    print(report)
    print(f"Wrote {RESULTS_CSV}")
    print(f"Wrote {RESULTS_MD}")


if __name__ == "__main__":
    main()
