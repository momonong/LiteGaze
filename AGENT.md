# LexiGaze Developer and Agent Guidelines

## Product vision

LexiGaze combines webcam-based gaze tracking and psycholinguistic text modelling to support unobtrusive English vocabulary learning. The core product goal is to identify potentially difficult words without interrupting reading, then offer the evidence as a review aid rather than a diagnosis.

## Architecture

1. `core/unigaze_personalization/`: webcam perception, preprocessing, gaze estimation, and personalization.
2. `core/cognition/`: word-level linguistic features and text-difficulty modelling.
3. `scripts/`: fusion, experiments, evaluation, and data tooling.
4. `web/`: Flask routes and the reader UI.

## Coding constraints

### Package imports

Do not add legacy path-injection hacks. Import core modules through `core` and web modules through `web`.

### API routes

Browser code under `web/` must use relative API paths such as `/api/gaze/predict` so local and tunnel deployments behave consistently.

### Unicode and runtime

- Run Python commands containing non-ASCII output with `-X utf8`.
- Read and write text with explicit UTF-8 encoding.
- Use the repository `.venv` Python 3.11 environment. The project supports Python `>=3.11,<3.14`.

### Headless backend

Server-side computer-vision paths must not require an interactive display.

## Reader Assessment v2 (experimental)

The former Cognitive Ability Inspector heuristics are retired. A single gaze trace must not be described as a validated measure of cognitive ability, attention, fatigue, reading ability, English proficiency, or CEFR.

Required claim discipline:

1. Keep observable session behaviour separate from latent-trait claims.
2. Report data quality and uncertainty; abstain when prerequisites are missing.
3. Calculate full-text WPM only from explicit text length, completion, and elapsed time.
4. Treat fixation duration, regressions, rereads, lexical effects, and early/late changes as context-dependent signals, not direct ability labels.
5. Keep typography experiments separate from ability assessment by using matched text and randomized or counterbalanced layouts.
6. Do not tune against a fixed question set and report results on the same questions. Freeze participant and item/text holdouts before fitting.
7. Do not promote the pilot theta to an ability label until the item bank has real calibration, reliability, fairness/DIF, and external-validity evidence.
8. Keep routine assessment validation CPU-only unless a GPU experiment has a preregistered need and budget.

The complete contract, research basis, protocol, and progress log are under `docs/reader_assessment/`.
