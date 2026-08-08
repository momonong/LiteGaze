# Pythia GECO L2 Replication v1 — Outcome-Blind Invalidation

- Invalidated: 2026-08-08 (Asia/Taipei)
- Outcome columns opened: **no**
- Pythia extraction: completed label-free
- GPT-2 extraction: stopped before completion

## Frozen failure

GPT-2 completed 518 of 588 label-free GECO texts, then stopped on the 519th
deterministically sorted text (`Text_ID = 4:33`). Token index 102 had character
offset `(450, 451)`, covering only the ASCII space between the display items
`words:` and `erything`. Its GPT-2 token ID was 220 (`Ġ`). It therefore
overlapped zero display-word spans and violated v1's rule that every token must
overlap exactly one word.

No reading-time column, prepared corpus feature, question, answer, or QA field
was opened. The v1 failure is technical and contains no comparative model
evidence. v1 remains invalid and its partial cache is not used.

## Permitted v2 correction

The only allowed change is a deterministic separator rule fixed before outcome
access:

1. the token must overlap zero words;
2. its source substring must contain only Unicode whitespace;
3. its complete offset must lie in the exact gap between two adjacent word
   spans;
4. it is assigned to the following word, matching the leading-space convention
   used by byte-level tokenizers;
5. leading/trailing whitespace, non-whitespace gaps, multiword overlaps,
   uncovered words, and non-monotonic mappings still fail closed.

Both GPT-2 and Pythia features must be recomputed under v2. No v1 result or
partial feature may enter evaluation.
