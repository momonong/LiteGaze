"""Experimental adaptive reading assessment with explicit psychometric limits.

The item parameters below are expert-seeded placeholders. They make the routing
algorithm testable, but they are not calibrated and must not be interpreted as
CEFR or general English proficiency.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from statistics import median
from typing import Any

PROTOCOL_VERSION = "reader-assessment-v2.0"
ITEM_BANK_VERSION = "pilot-en-2026-08-06"
CALIBRATION_STATUS = "expert_seed_only_uncalibrated"
MIN_ROUNDS = 4
MAX_ROUNDS = 6
STANDARD_LAYOUT = {"font_size": 16, "line_width": 650, "line_height": 1.7}


def _question(
    question_id: str,
    prompt: str,
    options: dict[str, str],
    answer: str,
    construct: str,
    difficulty: float,
    explanation: str,
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question": prompt,
        "options": options,
        "answer": answer,
        "construct": construct,
        "difficulty_b": difficulty,
        "discrimination_a": 1.0,
        "guessing_c": 0.25,
        "explanation": explanation,
        "calibration_status": CALIBRATION_STATUS,
    }


PASSAGES: tuple[dict[str, Any], ...] = (
    {
        "passage_id": "foundation-a-seed-library",
        "form_id": "A",
        "difficulty": "foundation",
        "difficulty_rank": -1,
        "domain": "community",
        "text": (
            "A neighborhood library started a seed exchange beside its usual book shelves. "
            "Residents may take packets of vegetable or flower seeds at no cost. In return, "
            "they are encouraged to save seeds from healthy plants and bring some back after "
            "the growing season. The library does not promise that every seed will grow, so "
            "volunteers label packets with the year, plant type, and any useful growing notes. "
            "The project has become more than a way to share supplies. Monthly meetings let "
            "new gardeners compare results with experienced neighbors and learn which plants "
            "cope well with the area's soil and weather."
        ),
        "questions": (
            _question(
                "F-A-01",
                "What are residents encouraged to return to the library?",
                {
                    "A": "Seeds saved from healthy plants",
                    "B": "Money for new shelves",
                    "C": "Finished vegetables",
                    "D": "Unused library cards",
                },
                "A",
                "explicit_information",
                -1.2,
                "The passage asks residents to save and return some seeds from healthy plants.",
            ),
            _question(
                "F-A-02",
                "Why do volunteers add the year and growing notes to each packet?",
                {
                    "A": "To make the packets look official",
                    "B": "To prevent people from attending meetings",
                    "C": "To give future gardeners useful context",
                    "D": "To guarantee every seed will grow",
                },
                "C",
                "inference",
                -1.0,
                "The labels help later users judge and grow seeds even though success is not guaranteed.",
            ),
            _question(
                "F-A-03",
                "In the final sentence, which plants does “which plants” refer to?",
                {
                    "A": "Plants sold by commercial farms",
                    "B": "Plants suitable for local conditions",
                    "C": "Plants displayed inside the library",
                    "D": "Plants that require no water",
                },
                "B",
                "lexical_cohesion",
                -0.8,
                "The phrase points to plants that cope with the local soil and weather.",
            ),
        ),
    },
    {
        "passage_id": "foundation-b-bus-signs",
        "form_id": "B",
        "difficulty": "foundation",
        "difficulty_rank": -1,
        "domain": "transport",
        "text": (
            "The city placed electronic arrival signs at several busy bus stops. The signs use "
            "location data from buses to estimate how many minutes passengers must wait. They "
            "are especially useful for people who do not own smartphones or who have limited "
            "mobile data. During the first month, however, riders noticed that the times were "
            "sometimes wrong when traffic was heavy. The transport office responded by showing "
            "a wider time range whenever the prediction was uncertain. It also added a symbol "
            "when a delay came from an accident or road closure. Riders said the revised signs "
            "were less precise, but more trustworthy."
        ),
        "questions": (
            _question(
                "F-B-01",
                "Who may benefit especially from the electronic signs?",
                {
                    "A": "Only bus drivers",
                    "B": "People who never use buses",
                    "C": "Road construction crews",
                    "D": "People without reliable smartphone access",
                },
                "D",
                "explicit_information",
                -1.2,
                "The passage specifically mentions people without smartphones or sufficient mobile data.",
            ),
            _question(
                "F-B-02",
                "Why did the office begin showing a wider time range?",
                {
                    "A": "A range communicates uncertainty more honestly",
                    "B": "The screens could not display single numbers",
                    "C": "Passengers asked for longer waits",
                    "D": "Buses stopped sharing location data",
                },
                "A",
                "inference",
                -1.0,
                "A wider range avoids false precision when traffic makes the prediction uncertain.",
            ),
            _question(
                "F-B-03",
                "What does “revised” most nearly mean in the final sentence?",
                {
                    "A": "Removed completely",
                    "B": "Hidden from drivers",
                    "C": "Changed after feedback",
                    "D": "Copied from another city",
                },
                "C",
                "lexical_cohesion",
                -0.8,
                "The office changed the signs after riders reported inaccurate predictions.",
            ),
        ),
    },
    {
        "passage_id": "standard-a-urban-trees",
        "form_id": "A",
        "difficulty": "standard",
        "difficulty_rank": 0,
        "domain": "environment",
        "text": (
            "Cities often promote tree planting as a simple response to extreme heat, but the "
            "benefits depend on where and how trees are maintained. A mature canopy can shade "
            "pavement and cool nearby air through evaporation. Yet young trees provide little "
            "shade, require water, and may die before reaching maturity if maintenance budgets "
            "are unstable. Placement also matters. Planting only in parks can improve pleasant "
            "areas that are already cool while leaving exposed bus stops and dense residential "
            "streets unchanged. Some cities therefore combine temperature maps with information "
            "about pedestrian activity and household vulnerability. This approach does not make "
            "planting less important; it treats limited trees, water, and labor as resources that "
            "should be directed toward locations where they can reduce risk most effectively."
        ),
        "questions": (
            _question(
                "S-A-01",
                "Which factor is presented as a risk to young trees?",
                {
                    "A": "Too much shade from buildings",
                    "B": "Unstable maintenance funding",
                    "C": "A lack of temperature maps",
                    "D": "Excess pedestrian activity",
                },
                "B",
                "explicit_information",
                -0.2,
                "The passage says young trees may die if maintenance budgets are unstable.",
            ),
            _question(
                "S-A-02",
                "What is the author's main reason for combining several kinds of city data?",
                {
                    "A": "To prove parks should receive every new tree",
                    "B": "To replace planting with air conditioning",
                    "C": "To estimate the age of existing trees",
                    "D": "To target limited resources where heat risk can be reduced",
                },
                "D",
                "inference",
                0.0,
                "The final sentence emphasizes directing limited resources to places with the greatest risk reduction.",
            ),
            _question(
                "S-A-03",
                "In the final sentence, what does “they” refer to?",
                {
                    "A": "Trees, water, and labor",
                    "B": "Temperature maps",
                    "C": "Residential streets",
                    "D": "Household vulnerabilities",
                },
                "A",
                "lexical_cohesion",
                0.2,
                "The nearest plural resources are trees, water, and labor.",
            ),
        ),
    },
    {
        "passage_id": "standard-b-bird-counts",
        "form_id": "B",
        "difficulty": "standard",
        "difficulty_rank": 0,
        "domain": "science",
        "text": (
            "A volunteer bird count can cover far more territory than a small research team, "
            "but a large sample is not automatically a representative one. Volunteers tend to "
            "visit convenient parks, and experienced observers identify more species than "
            "beginners. If researchers simply total every report, differences in access and skill "
            "may look like ecological change. Modern projects address this problem by recording "
            "how long each observer searched, how far they traveled, and whether they reported "
            "every species they could identify. Statistical models can then account for uneven "
            "effort. The corrections are imperfect, yet they make the limits of the data visible. "
            "Citizen science is most valuable not when volunteer observations are treated as "
            "error-free, but when the project design anticipates how those observations were produced."
        ),
        "questions": (
            _question(
                "S-B-01",
                "Why can a simple total of bird reports be misleading?",
                {
                    "A": "Birds are never found in parks",
                    "B": "Researchers refuse to use large samples",
                    "C": "Observer access and skill affect what gets reported",
                    "D": "Volunteers always travel the same distance",
                },
                "C",
                "explicit_information",
                -0.2,
                "The passage identifies convenience and observer skill as sources of unequal reports.",
            ),
            _question(
                "S-B-02",
                "What broader principle does the passage support?",
                {
                    "A": "More observations remove every source of bias",
                    "B": "Data should be interpreted in light of how it was collected",
                    "C": "Only professionals can contribute to science",
                    "D": "Statistical corrections make study design unnecessary",
                },
                "B",
                "inference",
                0.0,
                "The conclusion stresses anticipating the process that produced observations.",
            ),
            _question(
                "S-B-03",
                "What does “The corrections” refer to?",
                {
                    "A": "Teaching birds to enter parks",
                    "B": "Replacing beginners with experts",
                    "C": "Deleting every unusual report",
                    "D": "Statistically accounting for uneven observation effort",
                },
                "D",
                "lexical_cohesion",
                0.2,
                "The previous sentence describes models that account for uneven effort.",
            ),
        ),
    },
    {
        "passage_id": "advanced-a-feedback-loops",
        "form_id": "A",
        "difficulty": "advanced",
        "difficulty_rank": 1,
        "domain": "public_policy",
        "text": (
            "Predictive systems used to allocate inspections or patrols are often evaluated by "
            "asking whether their forecasts match later records. That comparison can be circular. "
            "Suppose an algorithm sends more inspectors to neighborhoods with historically high "
            "violation counts. Increased inspection will reveal more violations there, even if the "
            "underlying rate is identical elsewhere. When those newly observed cases become training "
            "data, the system may interpret its own allocation decision as confirmation of the original "
            "forecast. This feedback loop does not prove that every geographic prediction is biased, "
            "but it changes what counts as persuasive validation. Researchers need outcomes collected "
            "independently of the model's recommendations, randomized audits, or explicit estimates of "
            "detection probability. Otherwise, a model can appear increasingly accurate while merely "
            "becoming better at predicting where institutions will choose to look."
        ),
        "questions": (
            _question(
                "A-A-01",
                "What creates the feedback loop described in the passage?",
                {
                    "A": "Model-directed inspections generate records later used to confirm the model",
                    "B": "Inspectors refuse to record discovered violations",
                    "C": "Every neighborhood has a different underlying rate",
                    "D": "Randomized audits are included in training",
                },
                "A",
                "explicit_information",
                0.8,
                "The allocation changes where cases are detected, and those detections are then reused as training evidence.",
            ),
            _question(
                "A-A-02",
                "Why would randomized audits strengthen validation?",
                {
                    "A": "They guarantee that the algorithm has no errors",
                    "B": "They increase historical violation counts",
                    "C": "They hide inspection locations from researchers",
                    "D": "They provide outcomes less dependent on the model's own allocation choices",
                },
                "D",
                "inference",
                1.0,
                "Random allocation breaks the direct dependence between model recommendations and observed outcomes.",
            ),
            _question(
                "A-A-03",
                "In the final sentence, what contrast is established by “while merely”?",
                {
                    "A": "Accuracy is contrasted with speed",
                    "B": "Institutions are contrasted with neighborhoods",
                    "C": "True predictive improvement is contrasted with predicting observation behavior",
                    "D": "Audits are contrasted with historical records",
                },
                "C",
                "lexical_cohesion",
                1.2,
                "The system may seem to predict violations better when it actually predicts where observers will search.",
            ),
        ),
    },
    {
        "passage_id": "advanced-b-preregistration",
        "form_id": "B",
        "difficulty": "advanced",
        "difficulty_rank": 1,
        "domain": "research_methods",
        "text": (
            "Preregistration asks researchers to record hypotheses and analysis plans before seeing "
            "the relevant outcomes. Its purpose is not to forbid exploration. Rather, it preserves a "
            "distinction between tests that were planned in advance and patterns noticed after many "
            "analytical choices. Without that distinction, a surprising association can be presented "
            "as though it had been predicted all along, making ordinary chance variation look more "
            "compelling. Preregistration does not repair weak measurements, biased samples, or careless "
            "implementation, and an inflexible plan can ignore legitimate complications. For this reason, "
            "transparent deviations are often more informative than mechanical obedience. A useful record "
            "states what changed, when it changed, and why. The central benefit is therefore not purity of "
            "procedure, but a clearer audit trail for judging which conclusions are confirmatory and which "
            "remain exploratory."
        ),
        "questions": (
            _question(
                "A-B-01",
                "Which problem is preregistration primarily intended to address?",
                {
                    "A": "The cost of collecting large samples",
                    "B": "Presenting discovered patterns as prior predictions",
                    "C": "The use of any exploratory analysis",
                    "D": "Mechanical errors in measurement devices",
                },
                "B",
                "explicit_information",
                0.8,
                "The passage focuses on distinguishing advance predictions from patterns noticed after analysis choices.",
            ),
            _question(
                "A-B-02",
                "Why can a documented deviation be preferable to strict obedience?",
                {
                    "A": "It makes weak measurements valid",
                    "B": "It removes the need for an original plan",
                    "C": "It reveals how and why the analysis changed",
                    "D": "It converts exploratory results into confirmatory ones",
                },
                "C",
                "inference",
                1.0,
                "Transparent changes preserve an audit trail when legitimate complications arise.",
            ),
            _question(
                "A-B-03",
                "What does “that distinction” refer to in the third sentence?",
                {
                    "A": "The difference between planned tests and later discoveries",
                    "B": "The difference between large and small samples",
                    "C": "The difference between measurement and implementation",
                    "D": "The difference between rigid and flexible software",
                },
                "A",
                "lexical_cohesion",
                1.2,
                "It refers to the immediately preceding contrast between planned tests and patterns noticed later.",
            ),
        ),
    },
)


PASSAGE_BY_ID = {passage["passage_id"]: passage for passage in PASSAGES}
ITEM_BY_ID = {
    item["question_id"]: item for passage in PASSAGES for item in passage["questions"]
}


def _stable_fraction(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def public_passage(passage: dict[str, Any]) -> dict[str, Any]:
    """Return an assessment passage without answer-key leakage."""

    return {
        "passage_id": passage["passage_id"],
        "form_id": passage["form_id"],
        "difficulty": passage["difficulty"],
        "difficulty_rank": passage["difficulty_rank"],
        "domain": passage["domain"],
        "text": passage["text"],
        "word_count": len(passage["text"].split()),
        **STANDARD_LAYOUT,
        "quiz": [
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "options": dict(item["options"]),
                "construct": item["construct"],
            }
            for item in passage["questions"]
        ],
    }


def initial_passage(assessment_id: str) -> dict[str, Any]:
    standard = [passage for passage in PASSAGES if passage["difficulty_rank"] == 0]
    index = 0 if _stable_fraction(assessment_id, "initial") < 0.5 else 1
    return standard[index]


def score_passage(passage_id: str, responses: dict[str, str]) -> dict[str, Any]:
    passage = PASSAGE_BY_ID.get(passage_id)
    if passage is None:
        raise ValueError("unknown passage_id")
    if not isinstance(responses, dict):
        raise TypeError("responses must be an object")
    results = []
    for item in passage["questions"]:
        selected = str(responses.get(item["question_id"], "")).upper()
        if selected not in item["options"]:
            raise ValueError(f"missing or invalid response for {item['question_id']}")
        is_correct = selected == item["answer"]
        results.append(
            {
                "question_id": item["question_id"],
                "construct": item["construct"],
                "selected": selected,
                "correct": is_correct,
                "explanation": item["explanation"],
            }
        )
    correct = sum(result["correct"] for result in results)
    return {
        "passage_id": passage_id,
        "correct": correct,
        "total": len(results),
        "item_results": results,
    }


def _valid_item_results(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for record in history:
        if not isinstance(record, dict):
            continue
        passage = PASSAGE_BY_ID.get(str(record.get("passage_id", "")))
        if passage is None:
            continue
        allowed_ids = {item["question_id"] for item in passage["questions"]}
        for result in record.get("item_results", []):
            if not isinstance(result, dict):
                continue
            question_id = str(result.get("question_id", ""))
            if question_id in allowed_ids and isinstance(result.get("correct"), bool):
                results.append(
                    {"question_id": question_id, "correct": result["correct"]}
                )
    return results


def _response_probability(theta: float, item: dict[str, Any]) -> float:
    exponent = -item["discrimination_a"] * (theta - item["difficulty_b"])
    logistic = 1.0 / (1.0 + math.exp(_clip(exponent, -40.0, 40.0)))
    return item["guessing_c"] + (1.0 - item["guessing_c"]) * logistic


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def estimate_theta(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Return an EAP estimate on the uncalibrated pilot-bank scale."""

    responses = _valid_item_results(history)
    grid = [-4.0 + 0.05 * index for index in range(161)]
    log_weights: list[float] = []
    for theta in grid:
        log_weight = -0.5 * theta * theta
        for result in responses:
            item = ITEM_BY_ID[result["question_id"]]
            probability = _clip(_response_probability(theta, item), 1e-9, 1.0 - 1e-9)
            log_weight += math.log(
                probability if result["correct"] else 1.0 - probability
            )
        log_weights.append(log_weight)
    maximum = max(log_weights)
    weights = [math.exp(weight - maximum) for weight in log_weights]
    total_weight = sum(weights)
    normalized = [weight / total_weight for weight in weights]
    mean = sum(theta * weight for theta, weight in zip(grid, normalized))
    variance = sum(
        (theta - mean) ** 2 * weight for theta, weight in zip(grid, normalized)
    )

    cumulative = 0.0
    lower = grid[0]
    upper = grid[-1]
    lower_found = False
    for theta, weight in zip(grid, normalized):
        cumulative += weight
        if not lower_found and cumulative >= 0.025:
            lower = theta
            lower_found = True
        if cumulative >= 0.975:
            upper = theta
            break
    return {
        "theta": round(mean, 3),
        "posterior_sd": round(math.sqrt(max(0.0, variance)), 3),
        "credible_interval_95": [round(lower, 2), round(upper, 2)],
        "item_count": len(responses),
        "scale": "uncalibrated_pilot_bank_logit",
        "calibration_status": CALIBRATION_STATUS,
    }


def select_next_passage(
    history: list[dict[str, Any]], assessment_id: str
) -> dict[str, Any] | None:
    used = {
        str(record.get("passage_id")) for record in history if isinstance(record, dict)
    }
    candidates = [passage for passage in PASSAGES if passage["passage_id"] not in used]
    if not candidates:
        return None
    estimate = estimate_theta(history)
    theta = estimate["theta"]
    target_rank = -1 if theta < -0.55 else 1 if theta > 0.55 else 0
    prior_domains = Counter(
        PASSAGE_BY_ID[passage_id]["domain"]
        for passage_id in used
        if passage_id in PASSAGE_BY_ID
    )

    def selection_key(passage: dict[str, Any]) -> tuple[float, float, float]:
        rank_distance = abs(passage["difficulty_rank"] - target_rank)
        domain_penalty = prior_domains[passage["domain"]]
        tie_break = _stable_fraction(
            assessment_id, passage["passage_id"], str(len(history))
        )
        return (rank_distance, domain_penalty, tie_break)

    return min(candidates, key=selection_key)


def should_stop(history: list[dict[str, Any]]) -> bool:
    rounds = len(history)
    if rounds >= MAX_ROUNDS:
        return True
    if rounds < MIN_ROUNDS:
        return False
    return estimate_theta(history)["posterior_sd"] <= 0.58


def adaptive_analysis(history: list[dict[str, Any]]) -> dict[str, Any]:
    item_results = _valid_item_results(history)
    correct = sum(result["correct"] for result in item_results)
    total = len(item_results)
    passages = [
        PASSAGE_BY_ID[str(record.get("passage_id"))]
        for record in history
        if isinstance(record, dict) and str(record.get("passage_id")) in PASSAGE_BY_ID
    ]
    constructs: dict[str, list[bool]] = defaultdict(list)
    for result in item_results:
        item = ITEM_BY_ID[result["question_id"]]
        constructs[item["construct"]].append(result["correct"])

    rates = []
    for record in history:
        try:
            rate = float(record.get("wpm"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(rate) and 20.0 <= rate <= 1000.0:
            rates.append(rate)

    if len(passages) >= 4 and total >= 12 and len(constructs) >= 3:
        evidence_status = "experimental_complete"
        comprehension_status = "provisional_session_estimate"
    elif len(passages) >= 3 and total >= 9:
        evidence_status = "limited"
        comprehension_status = "limited_session_evidence"
    else:
        evidence_status = "insufficient"
        comprehension_status = "insufficient_data"

    theta = estimate_theta(history)
    construct_results = {
        construct: {
            "correct": sum(values),
            "total": len(values),
            "proportion_correct": round(sum(values) / len(values), 3),
        }
        for construct, values in sorted(constructs.items())
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "item_bank_version": ITEM_BANK_VERSION,
        "calibration_status": CALIBRATION_STATUS,
        "data_quality": {
            "status": evidence_status,
            "passage_count": len(passages),
            "item_count": total,
            "construct_count": len(constructs),
            "domain_count": len({passage["domain"] for passage in passages}),
        },
        "observations": {
            "correct": correct,
            "total": total,
            "proportion_correct": round(correct / total, 3) if total else None,
            "constructs": construct_results,
            "median_reading_rate_wpm": round(median(rates), 1) if rates else None,
            "reading_rate_range_wpm": [round(min(rates), 1), round(max(rates), 1)]
            if rates
            else None,
        },
        "experimental_model": theta,
        "claims": {
            "reading_comprehension_session": {
                "status": comprehension_status,
                "scope": "performance on this uncalibrated pilot item bank",
                "correct": correct,
                "total": total,
            },
            "english_proficiency": {
                "status": "not_estimated",
                "reason": "Pilot item parameters are not empirically calibrated or linked to CEFR/external proficiency tests.",
            },
            "general_reading_ability": {
                "status": "not_estimated",
                "reason": "The current evidence is too short and not norm referenced.",
            },
            "cognitive_ability": {
                "status": "not_estimated",
                "reason": "Reading-item performance does not identify general cognitive ability.",
            },
            "typography_preference": {
                "status": "not_estimated",
                "reason": "Typography was intentionally held constant to avoid confounding ability with layout.",
            },
        },
    }


def generate_adaptive_report(
    analysis: dict[str, Any], participant_id: str, history: list[dict[str, Any]]
) -> str:
    observations = analysis["observations"]
    model = analysis["experimental_model"]
    quality = analysis["data_quality"]
    rows = []
    for index, record in enumerate(history, start=1):
        passage = PASSAGE_BY_ID.get(str(record.get("passage_id", "")), {})
        rows.append(
            "| {round_no} | {passage_id} | {difficulty} | {score}/{total} | {wpm} |".format(
                round_no=index,
                passage_id=record.get("passage_id", "unknown"),
                difficulty=passage.get("difficulty", "unknown"),
                score=record.get("quiz_score", "—"),
                total=record.get("quiz_total", "—"),
                wpm=record.get("wpm", "—"),
            )
        )
    return f"""# LexiGaze 實驗性閱讀評量報告 v2

- 參與者：`{participant_id}`
- 協定：`{analysis["protocol_version"]}`
- 題庫：`{analysis["item_bank_version"]}`
- 題庫狀態：`{analysis["calibration_status"]}`

> [!WARNING]
> 這是未校準 pilot 題庫的 session 表現，不是 CEFR、英文熟練度、智力、注意力或疲勞診斷。實驗模型的 theta 只存在於目前題庫內，不可與真實能力等級對照。

## 證據完整度

- 狀態：`{quality["status"]}`
- 文章數：`{quality["passage_count"]}`
- 題目數：`{quality["item_count"]}`
- 構念數：`{quality["construct_count"]}`
- 主題領域數：`{quality["domain_count"]}`

## 各輪結果

| 輪次 | 文章 ID | 相對難度 | 理解題 | WPM |
| :--- | :--- | :--- | :--- | :--- |
{chr(10).join(rows)}

## 可回報的結果

- 本題庫答對：`{observations["correct"]}/{observations["total"]}`。
- 中位閱讀速率：`{observations["median_reading_rate_wpm"]} WPM`；範圍 `{observations["reading_rate_range_wpm"]}`。
- 構念分項：`{observations["constructs"]}`。
- 實驗性 theta：`{model["theta"]}`，posterior SD `{model["posterior_sd"]}`，95% credible interval `{model["credible_interval_95"]}`。

## 明確不回報的結果

- 英文能力／CEFR：`not_estimated`。
- 一般閱讀能力：`not_estimated`。
- 認知能力、注意力、疲勞：`not_estimated`。
- 最佳字體、欄寬、行高：`not_estimated`。本協定固定排版，避免把文章難度與排版效果混在一起。

## 下一階段

收集足夠 pilot 反應後，需在受試者與題目雙重 holdout 下估計題目難度、鑑別度、猜測率、DIF、公平性、重測信度與外部效度。完成前不得把 pilot theta 轉換成能力標籤。
"""


def validate_item_bank() -> dict[str, Any]:
    passage_ids = [passage["passage_id"] for passage in PASSAGES]
    question_ids = [
        item["question_id"] for passage in PASSAGES for item in passage["questions"]
    ]
    answer_counts = Counter(
        item["answer"] for passage in PASSAGES for item in passage["questions"]
    )
    constructs = Counter(
        item["construct"] for passage in PASSAGES for item in passage["questions"]
    )
    errors: list[str] = []
    if len(passage_ids) != len(set(passage_ids)):
        errors.append("duplicate_passage_id")
    if len(question_ids) != len(set(question_ids)):
        errors.append("duplicate_question_id")
    if any(len(passage["questions"]) != 3 for passage in PASSAGES):
        errors.append("passage_question_count_not_three")
    if max(answer_counts.values()) - min(answer_counts.values()) > 1:
        errors.append("answer_key_distribution_unbalanced")
    if set(constructs) != {"explicit_information", "inference", "lexical_cohesion"}:
        errors.append("construct_coverage_incomplete")
    if any(
        public.get("answer")
        for passage in PASSAGES
        for public in public_passage(passage)["quiz"]
    ):
        errors.append("answer_key_leakage")
    return {
        "ok": not errors,
        "errors": errors,
        "passage_count": len(PASSAGES),
        "question_count": len(question_ids),
        "answer_distribution": dict(sorted(answer_counts.items())),
        "construct_distribution": dict(sorted(constructs.items())),
        "difficulty_distribution": dict(
            sorted(Counter(passage["difficulty"] for passage in PASSAGES).items())
        ),
        "layout": dict(STANDARD_LAYOUT),
        "calibration_status": CALIBRATION_STATUS,
    }
