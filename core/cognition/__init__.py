"""Public cognition API with lazy model-runtime imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline import CognitiveLoadPipeline, WordResult

__all__ = ["CognitiveLoadPipeline", "WordResult"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .pipeline import CognitiveLoadPipeline, WordResult

        return {
            "CognitiveLoadPipeline": CognitiveLoadPipeline,
            "WordResult": WordResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
