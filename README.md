# Research Paper Summarizer

This is a Python-based project I created for extracting, cleaning and preparing research papers (in the forms of a PDF) for summarization and analysis. It is designed to help users who may have minimal experience in reading technical papers get a better understanding of what the paper is actually about.

## Features

- Extracts text from PDF research papers
- Cleans and preprocesses extracted text
- Splits text into sentence-based chunks
- Modular design for easy extension (summarization, ranking, aggregation, etc.)

## Requirements

- Python 3.7+
- pip (Python package manager)

### Python Packages

Install all required packages using:

```bash
pip install -r requirements.txt
```

#### Additional spaCy Model

The project uses spaCy’s `en_core_web_sm` model for sentence segmentation. This model will be automatically downloaded the first time you run the code. If you want to install it manually, run:

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

The script will extract, clean, and chunk the text, printing the results to the console.

## Project Structure

```
research-paper-summarizer/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── src/
	├── __init__.py
	├── aggregation.py
	├── chunking.py
	├── extraction.py
	├── preprocessing.py
	├── ranking.py
	└── summarization.py
```

- `main.py`: Entry point for running the pipeline.
- `src/`: Contains all processing modules.

## Notes

- Make sure your PDF files are not encrypted or scanned images (text extraction works on text-based PDFs).
