# General collection bank v1 — automated audit

Date: 2026-08-08
Bank: `lexigaze-general-reading-rehearsal-v1` / `2026-08-08.v2`
Compute: CPU only; GPU not used.

## Outcome

The 12-passage A/B rehearsal bank passed the frozen automated screen, while remaining explicitly pending two independent human reviews.

- 12 unique passage families and 96 probes; every probe surface occurs exactly once in its passage.
- Each form contains two foundation, two standard, and two advanced passages.
- Mean heuristic grade increases by band: foundation 8.74, standard 13.23, advanced 14.75.
- A/B form mean heuristic grade is 12.21 versus 12.26; mean word count is 118.0 versus 115.5; mean probe Zipf frequency differs by 0.287.
- The bank spans 9 domains and 5 genres. Maximum pairwise five-word-shingle Jaccard overlap is 0.0.
- Reading length ranges from 100 to 130 words for form passages. The separate practice passage is excluded from analysis.

## Boundary

Flesch-Kincaid, word-frequency, length, and n-gram checks are inexpensive screens, not validity evidence. They cannot establish factual accuracy, naturalness, cultural accessibility, fairness, construct validity, or empirical item difficulty. The bank therefore remains `rehearsal_only_pending_two_independent_human_reviews` and cannot unlock formal recruitment or confirmation experiments.

Machine-readable evidence: `results/2026-08-08-general-collection-bank-v1-audit.json`.
