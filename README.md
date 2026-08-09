# Research Paper Summarizer

A Python pipeline that turns a research paper PDF into a structured, readable summary. It extracts and cleans text directly from the PDF's layout, figures out which sentences actually matter using semantic embeddings, and generates an abstractive summary (with a TL;DR) using a transformer summarization model — no manual formatting or LLM API calls required.

## Features

- **Layout-aware extraction** — reads font size, weight, and orientation directly from the PDF (via `pdfplumber`) to detect section headings and strip rotated margin stamps, footnotes, and author-affiliation blocks, instead of relying on hardcoded regex patterns.
- **Text cleanup** — repairs common PDF extraction artifacts: merged words, hyphenated line breaks, garbled figure/table text, and reversed text from vector graphics.
- **Semantic chunk ranking** — embeds sentence chunks with `sentence-transformers` and ranks them by similarity to each section's centroid, surfacing the most representative content instead of just taking the first N sentences.
- **Abstractive summarization** — summarizes the top-ranked chunks per section with a HuggingFace BART model (`distilbart-cnn`), recursively reducing oversized sections so nothing is lost to truncation.
- **Synthesized TL;DR** — produces a short, top-line summary distilled from the abstract, introduction, and conclusion.
- **Modular pipeline** — each stage (extraction, preprocessing, chunking, ranking, summarization, aggregation) is an independent, swappable module.

## How It Works

```
PDF → Extraction → Preprocessing → Chunking → Ranking → Summarization → Aggregation → Final Summary
```

1. **Extraction** — pulls text and font metadata per word, groups it into lines and sections.
2. **Preprocessing** — cleans artifacts and classifies lines as headings vs. body text using font size.
3. **Chunking** — splits each section into sentence-based chunks with spaCy.
4. **Ranking** — embeds chunks and scores them by similarity to their section's centroid.
5. **Summarization** — abstractively summarizes the top-ranked chunks per section.
6. **Aggregation** — joins section summaries in paper order and synthesizes a TL;DR.

## Tech Stack

`pdfplumber` · `spaCy` · `sentence-transformers` · `HuggingFace transformers` (BART) · `scikit-learn`

## Requirements

- Python 3.7+
- pip (Python package manager)

### Python Packages

Install all required packages using:

```bash
pip install -r requirements.txt
```

#### Additional spaCy Model

The project uses spaCy's `en_core_web_sm` model for sentence segmentation. This model will be automatically downloaded the first time you run the code. If you want to install it manually, run:

```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Place your research paper PDF in a known location.
2. Update the `pdf_path` variable in `main.py` to point to your PDF.
3. Run the main script:

```bash
python main.py
```

The script prints a Markdown-formatted summary document to the console — a TL;DR followed by a per-section abstractive summary in the paper's original order.

## Configuration

All tunable parameters (chunk size, model choice, summary length caps, etc.) live in [src/config.py](src/config.py).

## Project Structure

```
research-paper-summarizer/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── src/
	├── __init__.py
	├── config.py
	├── extraction.py
	├── preprocessing.py
	├── chunking.py
	├── ranking.py
	├── summarization.py
	└── aggregation.py
```

- `main.py`: Entry point that runs the full pipeline.
- `src/`: Contains all processing modules.

## Roadmap

- Quantitative evaluation of summary quality via ROUGE scoring.

## Notes

- Make sure your PDF files are not encrypted or scanned images (text extraction works on text-based PDFs).
