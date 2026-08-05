"""Single source of truth for language-model selection by language."""

from __future__ import annotations

from types import MappingProxyType


DEFAULT_MODEL_BY_LANGUAGE = MappingProxyType(
    {
        "en": "gpt2",
        "zh": "bert",
        "nl": "bert",
    }
)


def default_model_for_language(language: str) -> str:
    """Return the validated default model for a normalized language code."""
    normalized = str(language).strip().lower()
    try:
        return DEFAULT_MODEL_BY_LANGUAGE[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(DEFAULT_MODEL_BY_LANGUAGE))
        raise ValueError(
            f"unsupported language {language!r}; expected one of: {supported}"
        ) from exc
