import os
import json
import time
import xml.etree.ElementTree as ET

import requests

API = "http://export.arxiv.org/api/query"
# arXiv asks automated clients to identify themselves. Set the ARXIV_CONTACT
# environment variable to your own email or repo URL; the default is a
# placeholder so no personal address is committed to the repository.
CONTACT = os.environ.get("ARXIV_CONTACT", "contact-not-set")
HEADERS = {"User-Agent": f"research-paper-summarizer/1.0 ({CONTACT})"}
NS = {"atom": "http://www.w3.org/2005/Atom"}

CATEGORY = "cs.CL"
MAX_RESULTS = 25
TARGET_PAPERS = 20
REQUEST_DELAY = 3
RETRY_ATTEMPTS = 4
RETRY_BACKOFF = 5

# Anchored to this file, not the working directory - otherwise running the
# script from a different folder creates a second data/ tree next to the CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PAPERS_DIR = os.path.join(DATA_DIR, "papers")
REFERENCES_PATH = os.path.join(DATA_DIR, "references.json")


def _collapse(text):
    return " ".join(text.split())


def _get(url, params=None, timeout=30):
    # arXiv throttles aggressively and returns 429 on even small queries, so
    # every request retries with exponential backoff. RequestException is the
    # base class: ReadTimeout is NOT an HTTPError and would escape otherwise.
    delay = RETRY_BACKOFF

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as error:
            if attempt == RETRY_ATTEMPTS:
                raise
            print(f"    {type(error).__name__}, retrying in {delay}s")
            time.sleep(delay)
            delay *= 2


def fetch_metadata(category=CATEGORY, max_results=MAX_RESULTS):
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    response = _get(API, params=params)

    root = ET.fromstring(response.content)

    papers = []
    for entry in root.findall("atom:entry", NS):
        raw_id = entry.find("atom:id", NS)
        title = entry.find("atom:title", NS)
        abstract = entry.find("atom:summary", NS)

        if raw_id is None or title is None or abstract is None:
            continue

        papers.append({
            "arxiv_id": raw_id.text.rsplit("/", 1)[-1],
            "title": _collapse(title.text),
            "abstract": _collapse(abstract.text),
        })

    return papers


def download_pdf(arxiv_id):
    # Returns True if the network was hit, False if the PDF was already on disk.
    path = os.path.join(PAPERS_DIR, f"{arxiv_id}.pdf")

    if os.path.exists(path):
        return False

    response = _get(f"https://arxiv.org/pdf/{arxiv_id}", timeout=60)

    # A throttled request can come back 200 OK with an HTML notice body, which
    # raise_for_status() will not catch. Checking the magic bytes stops that
    # HTML from being saved as a .pdf and failing later inside pdfplumber.
    if not response.content.startswith(b"%PDF"):
        raise ValueError("response body was not a PDF")

    with open(path, "wb") as f:
        f.write(response.content)

    return True


def main():
    os.makedirs(PAPERS_DIR, exist_ok=True)

    papers = fetch_metadata()
    print(f"Found {len(papers)} papers, keeping up to {TARGET_PAPERS}..\n")

    references = {}

    for paper in papers:
        if len(references) >= TARGET_PAPERS:
            break

        arxiv_id = paper["arxiv_id"]

        try:
            fetched = download_pdf(arxiv_id)
        except (requests.exceptions.RequestException, ValueError, OSError) as error:
            # One dead link shouldn't kill the whole run.
            print(f"  skipped {arxiv_id}: {type(error).__name__}")
            continue

        references[arxiv_id] = {
            "title": paper["title"],
            "abstract": paper["abstract"],
        }
        print(f"  {len(references):2d}. {arxiv_id}  {paper['title'][:60]}")

        if fetched:
            time.sleep(REQUEST_DELAY)

    # Written after the loop so references.json only lists papers whose PDF
    # actually landed - the two never disagree.
    with open(REFERENCES_PATH, "w", encoding="utf-8") as f:
        json.dump(references, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(references)} references to {REFERENCES_PATH}")


if __name__ == "__main__":
    main()
