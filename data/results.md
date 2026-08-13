# ROUGE Evaluation

Reference summaries are the author-written abstracts of 20 arXiv papers. The abstract is removed from the input before summarization, so no system sees its own reference.

Extraction succeeded on **20/20** papers.

| System | ROUGE-1 F | ROUGE-2 F | ROUGE-Lsum F | Avg words |
|---|---|---|---|---|
| full | 0.3268 | 0.0987 | 0.3023 | 107 |
| lead3 | 0.2000 | 0.0440 | 0.1773 | 74 |
| extractive | 0.3709 | 0.1064 | 0.3390 | 466 |

Reference abstracts average 198 words.
