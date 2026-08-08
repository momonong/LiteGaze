"""Automated lexical/balance screen; it never substitutes for human review."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wordfreq import zipf_frequency

from core.participant_study.general_collection import (
    load_general_bank,
    load_general_protocol,
    validate_general_design,
)

WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z'-]*\b")
VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def _syllables(word: str) -> int:
    cleaned = re.sub(r"[^a-z]", "", word.casefold())
    if not cleaned:
        return 0
    count = len(VOWEL_GROUP_RE.findall(cleaned))
    if cleaned.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _metrics(passage: dict[str, object]) -> dict[str, object]:
    text = str(passage["text"])
    words = WORD_RE.findall(text)
    sentence_count = max(1, len(re.findall(r"[.!?]+", text)))
    syllable_count = sum(_syllables(word) for word in words)
    grade = (
        0.39 * (len(words) / sentence_count)
        + 11.8 * (syllable_count / len(words))
        - 15.59
    )
    probes = list(passage["probes"])
    normalized_words = [word.casefold() for word in words]
    shingles = {
        tuple(normalized_words[index : index + 5])
        for index in range(max(0, len(normalized_words) - 4))
    }
    return {
        "passage_id": passage["passage_id"],
        "family_id": passage["family_id"],
        "form_id": None,
        "difficulty_band": passage["difficulty_band"],
        "genre": passage["genre"],
        "domain": passage["domain"],
        "word_count": len(words),
        "sentence_count": sentence_count,
        "flesch_kincaid_grade_heuristic": grade,
        "mean_word_zipf": statistics.fmean(
            zipf_frequency(word, "en") for word in words
        ),
        "mean_probe_zipf": statistics.fmean(
            zipf_frequency(str(probe["surface"]), "en") for probe in probes
        ),
        "type_token_ratio": len(set(normalized_words)) / len(normalized_words),
        "_shingles": shingles,
    }


def audit_bank() -> dict[str, object]:
    protocol = load_general_protocol()
    bank = load_general_bank()
    design = validate_general_design(protocol, bank)
    form_lookup = {
        passage_id: form_id
        for form_id, passage_ids in bank["forms"].items()
        for passage_id in passage_ids
    }
    passages = [_metrics(dict(passage)) for passage in bank["passages"]]
    for passage in passages:
        passage["form_id"] = form_lookup[passage["passage_id"]]

    maximum_five_gram_jaccard = 0.0
    maximum_pair: list[str] = []
    for left_index, left in enumerate(passages):
        for right in passages[left_index + 1 :]:
            union = left["_shingles"] | right["_shingles"]
            overlap = len(left["_shingles"] & right["_shingles"]) / len(union) if union else 0.0
            if overlap > maximum_five_gram_jaccard:
                maximum_five_gram_jaccard = overlap
                maximum_pair = [str(left["passage_id"]), str(right["passage_id"])]

    by_band: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_form: dict[str, list[dict[str, object]]] = defaultdict(list)
    for passage in passages:
        by_band[str(passage["difficulty_band"])].append(passage)
        by_form[str(passage["form_id"])].append(passage)

    def aggregate(items: list[dict[str, object]]) -> dict[str, float]:
        return {
            "mean_grade": statistics.fmean(
                float(item["flesch_kincaid_grade_heuristic"]) for item in items
            ),
            "mean_word_count": statistics.fmean(float(item["word_count"]) for item in items),
            "mean_word_zipf": statistics.fmean(float(item["mean_word_zipf"]) for item in items),
            "mean_probe_zipf": statistics.fmean(float(item["mean_probe_zipf"]) for item in items),
            "mean_type_token_ratio": statistics.fmean(
                float(item["type_token_ratio"]) for item in items
            ),
        }

    band_summary = {band: aggregate(items) for band, items in sorted(by_band.items())}
    form_summary = {form: aggregate(items) for form, items in sorted(by_form.items())}
    form_differences = {
        metric: abs(form_summary["A"][metric] - form_summary["B"][metric])
        for metric in form_summary["A"]
    }
    band_grades = [
        band_summary[band]["mean_grade"]
        for band in ("foundation", "standard", "advanced")
    ]
    gates = {
        "difficulty_mean_grade_strictly_increases": all(
            right > left for left, right in zip(band_grades, band_grades[1:])
        ),
        "form_mean_grade_difference_le_1": form_differences["mean_grade"] <= 1.0,
        "form_mean_word_count_difference_le_5": form_differences["mean_word_count"] <= 5.0,
        "form_mean_probe_zipf_difference_le_0_5": form_differences["mean_probe_zipf"] <= 0.5,
        "maximum_five_gram_jaccard_le_0_05": maximum_five_gram_jaccard <= 0.05,
        "at_least_six_domains": len({item["domain"] for item in passages}) >= 6,
        "at_least_three_genres": len({item["genre"] for item in passages}) >= 3,
    }
    public_passages = [
        {key: value for key, value in item.items() if key != "_shingles"}
        for item in passages
    ]
    return {
        "schema_version": 1,
        "audit_id": "general-collection-bank-v1-automated-screen",
        "status": (
            "automated_screen_passed_human_review_pending"
            if all(gates.values())
            else "automated_screen_failed"
        ),
        "design": design,
        "passages": public_passages,
        "band_summary": band_summary,
        "form_summary": form_summary,
        "form_absolute_differences": form_differences,
        "diversity": {
            "domain_count": len({item["domain"] for item in passages}),
            "genre_count": len({item["genre"] for item in passages}),
            "maximum_pairwise_five_gram_jaccard": maximum_five_gram_jaccard,
            "maximum_overlap_pair": maximum_pair,
        },
        "gates": gates,
        "human_review": {
            "required_independent_reviewers": bank["review"][
                "required_independent_reviewers"
            ],
            "completed_independent_reviewers": bank["review"][
                "completed_independent_reviewers"
            ],
            "still_required": True,
            "warning": (
                "Flesch-Kincaid, Zipf frequency, and n-gram screens are descriptive "
                "heuristics. They do not establish construct validity, fairness, "
                "naturalness, factual accuracy, or difficulty calibration."
            ),
        },
        "compute": {"device": "cpu", "gpu_used": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_bank()
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] != "automated_screen_failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
