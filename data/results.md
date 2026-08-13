# ROUGE Evaluation

Reference summaries are the author-written abstracts of 20 arXiv papers. The abstract is removed from the input before summarization, so no system sees its own reference.

Extraction succeeded on **20/20** papers.

| System | ROUGE-1 F | 95% CI | ROUGE-2 F | ROUGE-Lsum F | Avg words |
|---|---|---|---|---|---|
| full | 0.3268 | [0.298, 0.356] | 0.0987 | 0.3023 | 107 |
| lead3 | 0.2000 | [0.181, 0.219] | 0.0440 | 0.1773 | 74 |
| extractive | 0.3709 | [0.340, 0.402] | 0.1064 | 0.3390 | 466 |

Paired per-paper comparison against the lead-3 baseline:

- **full** vs lead3: +0.1268 ROUGE-1, winning on **20/20** papers (sign test p = 1.9e-06)
- **extractive** vs lead3: +0.1710 ROUGE-1, winning on **19/20** papers (sign test p = 4e-05)

Reference abstracts average 198 words.
