from rouge_score import rouge_scorer

from src.chunking import get_nlp

_scorer = None

def get_scorer():
    global _scorer
    if _scorer is None:
        _scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    return _scorer


def score(reference, system):
    scorer = get_scorer()

    reference = split_into_lines(reference)
    system = split_into_lines(system)

    result = scorer.score(reference, system)

    return result

def split_into_lines(text):
    nlp = get_nlp()
    doc = nlp(text=text)

    sentences = [sent.text for sent in doc.sents]
    # rougeLsum requires \n separated sentences.
    lines = "\n".join(sentences)

    return lines

def _first_k_sentences(text, k):
    nlp = get_nlp()
    doc = nlp(text=text)

    sentences = [sent.text for sent in doc.sents]

    return " ".join(sentences[:k])

def lead_k(sections, k=3):
    for name in sections:
        if "introduction" in name:
            return _first_k_sentences(sections[name], k)

    if not sections:
        return ""

    # No introduction detected - fall back to the first section, still capped at k.
    return _first_k_sentences(next(iter(sections.values())), k)

def extractive_baseline(ranked_chunks, top_n=4):
    sorted_chunks = sorted(ranked_chunks, key=lambda x: x["score"], reverse=True)

    top_chunks = sorted_chunks[:top_n]

    return " ".join([chunk["text"] for chunk in top_chunks])

# This block runs solely when the file is executed directly
# (python -m src.evaluation) as a quick check that rouge-score is installed and
# wired up correctly.
if __name__ == "__main__":
    target = "The quick brown fox jumped over the lazy dog."
    prediction = "The agile brown fox jumped over the lazy dog."

    for metric, value in score(target, prediction).items():
        print(f"{metric}: P={value.precision:.4f} R={value.recall:.4f} F={value.fmeasure:.4f}")