# Generalizable Participant Collection v1

Status: standardized dress rehearsal; formal participant collection and model
promotion remain blocked.

## What “the same flow” means

Every participant receives the same consent boundary, coarse background
questions, device checks, calibration and validation targets, fixed typography,
number of passages, number and timing of word probes, data fields, retry rules,
and debrief. Passage order and A/B visit order are counterbalanced by a frozen
assignment cell; they are not chosen from participant outcomes.

The unit of generalization is deliberately larger than a webcam frame. The data
contract preserves participant, visit/capture session, device class, passage
family, passage, and probe IDs so later experiments can hold out complete
people, sessions, devices, and content families.

## Two-visit schedule

- Visit 1 and visit 2 are separated by 18–72 hours.
- Participants use the same device class and browser family for the primary
  alternate-form retest. A future device-transfer visit requires a separate
  protocol.
- Half of frozen assignment cells receive form A then B; half receive B then A.
- Each form has two foundation, two standard, and two advanced passages in one
  of six Williams-style rotated orders.
- Each passage is followed by eight hidden-until-after-reading word probes.

The current 12-family bank is researcher-authored and safe for software and
workflow rehearsal. It has not received two independent human reviews and is
therefore development-only. It cannot serve as the 48-family validation and
confirmation bank required by Reader Assessment v3.

An automated pre-screen now verifies word/probe uniqueness, A/B balance,
heuristic grade progression, word-frequency balance, domain/genre counts, and
five-word overlap. It passed, but these heuristics do not replace factual,
naturalness, accessibility, fairness, or difficulty review by two independent
people.

## Why word review replaces adaptive QA

The primary product question is whether a particular reader would benefit from
reviewing a particular word. The direct post-reading response is therefore
`no_review`, `unsure`, or `review_needed`. Comprehension correctness is not used
as a training or selection target, and the reading page never reveals gaze or
text-model scores.

This follows personalized complex-word work showing that word difficulty is
idiosyncratic across readers, while avoiding a closed QA loop. Passage-level
understanding, effort, completion, and interruption reports are retained as
secondary context and quality information.

## Gaze is optional evidence, not an exclusion rule

The browser sends transient frames to the same-origin inference endpoint. By
default it does not persist reading images; the stored observation is derived
telemetry: gaze coordinates, head pose, normalized face geometry, viewport,
monotonic timing, prediction success, and the word-layout snapshot.

In the explicit unencrypted self-only mode, the researcher may separately opt
in to a no-audio webcam recording for each timed passage. Recording starts only
with the reading timer and stops before word review. It never covers consent,
setup, calibration, validation, practice, or review. These files are marked
`self_development_only_not_confirmation` and are intended for gaze robustness,
quality-control, and synchronization development. The same recording must not
be used both to tune a model and to claim evaluation performance.

Independent five-point validation is required before and after the reading
block. Accuracy, precision, data loss, effective sampling rate, and drift are
reported separately. Poor gaze quality causes word-level gaze abstention; it
does not discard valid word-review labels or imply anything about the reader.

This choice is grounded in webcam and conventional eye-tracking studies that
recommend direct calibration verification and reporting accuracy, precision,
and data loss. Browser webcam work also shows that scrolling can increase
prediction error, so rehearsal passages use a fixed 650 px line width and must
fit without scrolling.

## What tonight’s data may and may not do

Allowed:

- test completion, resume, export, missingness, and withdrawal;
- measure calibration and reading-capture quality;
- estimate how often the gaze branch would abstain;
- verify that alternate forms and word labels are understandable;
- run development-only schema and feature smoke experiments;
- in the separately authorized self-only mode, develop gaze robustness from
  raw passage recordings, using later capture sessions as held-out checks.

Not allowed:

- tune a model or threshold and report it on the same participants or texts;
- promote a text, gaze, fusion, or participant model;
- claim cognitive ability, attention, fatigue, English proficiency, or CEFR;
- treat the 12 rehearsal passages as validation or confirmation evidence.

## Self-only development storage exception

The researcher may explicitly keep their own development data on an
unencrypted local volume without an automatic expiry. It must be disclosed on
the consent page and persisted in the session and export manifests as
`unencrypted_self_development` with
`retention_policy=manual_until_researcher_deletes`.

The exception is limited to one invite pair owned by the researcher. It never
unlocks external participants, public networking, formal collection, or
confirmation promotion. Inference snapshots remain memory-only. An additional
unchecked-by-default consent scope may save one no-audio media file per timed
passage. Raw media stays beside the source session and is not copied into the
analysis export; `reading_video_index.csv` contains integrity hashes, timing,
passage/round linkage, and the development-only role. Completed or failed
calibration images are removed, and an interrupted calibration retains images
for at most one hour.

## Evidence informing the design

- The W3C MediaStream Recording specification defines `MediaRecorder`, MIME
  selection, bitrate hints, and segmented `dataavailable` blobs. The self-only
  implementation records a video-only `MediaStream`, requests no audio track,
  and joins periodic chunks into one passage-level blob before upload.
- WebGazer's remote study began with consent and compatibility testing, and its
  authors reported greater error in a scrolling quiz task. It also motivates
  avoiding persistent webcam video when derived observations suffice.
- An independent five-point calibration-verification protocol was proposed to
  quantify accuracy and precision rather than cite ideal device specifications.
- A recent webcam-versus-EyeLink comparison evaluated accuracy, precision,
  data loss, head movement, and calibration as separate quality dimensions.
- EyeScore performed best on previously observed texts and weakened on unseen
  texts, supporting explicit passage-family holdouts rather than random rows.
- Personalized complex-word research found that individual models can beat a
  single population model, supporting direct per-reader word labels.

The machine contracts are
[`general_collection_v1.json`](../../core/participant_study/general_collection_v1.json)
and
[`general_collection_bank_v1.json`](../../core/participant_study/general_collection_bank_v1.json).

Media recording reference:
[W3C MediaStream Recording](https://www.w3.org/TR/mediastream-recording/).

## Remaining external blockers

Formal collection still requires the revised ethics determination, authorized
external anchor, a 48-family independently reviewed bank, frozen
development/validation/confirmation assignments, sample-size and subgroup
simulation, a practical-utility threshold, and a moderated end-to-end dress
rehearsal. Software must continue to fail closed until those conditions are
recorded.

A 50,000-iteration CPU-only sensitivity simulation has now been frozen and
run without participant outcomes. Under its base attrition/quality/subgroup
assumptions, 144 enrolled people were required to reach the joint yield targets
with at least 80% probability; optimistic and pessimistic scenarios required
128 and 208. These are conditional planning numbers, not recruitment
authorization. Blinded rehearsal rates must replace the assumptions, and any
tuned development cohort must remain separate from a newly recruited frozen
confirmation cohort.
