from __future__ import annotations


class OneEuroFilter:
    """Backend-side filter placeholder for future server-side smoothing."""

    def __init__(self, value: float | None = None) -> None:
        self.value = value

    def reset(self) -> None:
        self.value = None
