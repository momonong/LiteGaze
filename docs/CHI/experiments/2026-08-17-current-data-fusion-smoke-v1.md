# CHI Selective-Fusion Current-Data Smoke v1

Status: **`passed_pipeline_smoke`**

This is a pipeline/schema smoke test only. It does not fit a model, estimate an effect, or authorize a webcam-accuracy or user-benefit claim.

## Current data

- Participants / sessions: `1` / `1`
- Passages / families: `6` / `6`
- Word reviews: `48` (`no_review=39`, `unsure=7`, `review_needed=2`)
- Eligible reading-gaze sessions: `0`
- Unverified reading telemetry rows retained separately: `649`
- Required runtime branch: **`F1_text_person_fallback`**

The rare positive class and single self-development participant make effect fitting invalid. The useful result is that behavioral labels remain exportable while ineligible gaze stays outside the evidence table.

## Gates

- [x] `sessions_nonempty_and_completed`
- [x] `session_identity_unique_nonempty`
- [x] `review_session_foreign_key_valid`
- [x] `every_session_has_word_reviews`
- [x] `three_class_labels_allowlisted`
- [x] `review_identity_unique`
- [x] `required_review_fields_nonempty`
- [x] `eight_reviews_per_session_passage`
- [x] `formal_promotion_disabled`
- [x] `all_sessions_behavioral_only`
- [x] `ineligible_gaze_not_exported_as_evidence`
- [x] `ineligible_validation_not_exported_as_evidence`
- [x] `unavailable_uncertainty_not_exported_as_evidence`
- [x] `eligible_gaze_tables_empty_for_behavioral_only_smoke`
- [x] `unverified_reading_rows_are_non_evidence`
- [x] `manifest_gaze_counts_match_export_tables`
- [x] `manifest_gaze_separation_contract`

## Decision

Keep the export path for future dress rehearsals. Do not train F1/F2 or select an abstention threshold from this session.
