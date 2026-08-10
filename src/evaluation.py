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

def lead_k(sections, k=3):
    for name in sections:
        if "introduction" in name:
            section = sections[name]
            text = section["text"]

            nlp = get_nlp()
            doc = nlp(text=text)

            sentences = [sent.text for sent in doc.sents]

            return sentences[:k]

    return next(iter(sections.values()))



def extractive_baseline(ranked_chunks, top_n=4):
    sorted_chunks = sorted(ranked_chunks, key=lambda x: x["score"], reverse=True)

    top_chunks = sorted_chunks[:top_n]

    return " ".join([chunk["text"] for chunk in top_chunks])



target = "The quick brown fox jumped over the lazy dog."
prediction = "The agile brown fox jumped over the lazy dog."

if __name__ == "__main__":
    score(target, prediction)