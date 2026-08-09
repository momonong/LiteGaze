"""Participant-study protocol, consent, and lifecycle controls."""

from .calibration import audit_participant_calibration
from .independent_capture import (
    audit_independent_capture_plan,
    canonical_plan_sha256,
    load_capture_plan,
)
from .general_collection import (
    assignment_for_cell,
    classify_gaze_quality,
    load_general_bank,
    load_general_protocol,
    validate_general_design,
)
from .protocol import activation_status, load_protocol, public_protocol
from .store import (
    ParticipantStudyStore,
    READING_VIDEO_MAX_BYTES,
    READING_VIDEO_SCOPE,
    StudyAuthorizationError,
    StudyError,
    StudyNotReadyError,
    StudyStateError,
    StudyValidationError,
)

__all__ = [
    "ParticipantStudyStore",
    "READING_VIDEO_MAX_BYTES",
    "READING_VIDEO_SCOPE",
    "StudyAuthorizationError",
    "StudyError",
    "StudyNotReadyError",
    "StudyStateError",
    "StudyValidationError",
    "activation_status",
    "assignment_for_cell",
    "audit_independent_capture_plan",
    "audit_participant_calibration",
    "canonical_plan_sha256",
    "classify_gaze_quality",
    "load_general_bank",
    "load_general_protocol",
    "load_protocol",
    "load_capture_plan",
    "public_protocol",
    "validate_general_design",
]
