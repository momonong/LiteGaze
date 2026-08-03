"""Validated, atomic file-backed storage for LexiGaze document sessions."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import UUID, uuid4

DEFAULT_MAX_SESSION_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_SESSION_ITEMS = 250_000
MAX_FILENAME_CHARS = 255
MAX_FILETYPE_CHARS = 64


class SessionStoreError(Exception):
    """Base class for expected session storage failures."""


class InvalidSessionId(SessionStoreError):
    """Raised when a route identifier is not a UUID."""


class InvalidSessionPayload(SessionStoreError):
    """Raised when a new session does not match the storage contract."""


class SessionTooLarge(InvalidSessionPayload):
    """Raised when a session exceeds its configured resource boundary."""


class SessionNotFound(SessionStoreError):
    """Raised when a requested session does not exist."""


class CorruptSession(SessionStoreError):
    """Raised when a stored session cannot be decoded or validated."""


class SessionWriteError(SessionStoreError):
    """Raised when an atomic session write cannot be completed."""


class SessionDeleteError(SessionStoreError):
    """Raised when a stored session cannot be deleted."""


@dataclass(frozen=True)
class SessionScan:
    """A compatible session list plus integrity metadata."""

    sessions: list[dict[str, Any]]
    corrupt_count: int
    total_files: int


class SessionStore:
    """Store session JSON safely within one configured directory."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_bytes: int = DEFAULT_MAX_SESSION_BYTES,
        max_items: int = DEFAULT_MAX_SESSION_ITEMS,
        logger: logging.Logger | None = None,
        id_factory: Callable[[], UUID] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items <= 0:
            raise ValueError("max_items must be a positive integer")

        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.max_items = max_items
        self._logger = logger or logging.getLogger(__name__)
        self._id_factory = id_factory or uuid4
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = RLock()

    @staticmethod
    def canonicalize_id(session_id: str) -> str:
        """Return a path-safe canonical UUID or reject the identifier."""

        if not isinstance(session_id, str) or not session_id:
            raise InvalidSessionId("Session id must be a UUID")
        try:
            return str(UUID(session_id))
        except (AttributeError, TypeError, ValueError) as exc:
            raise InvalidSessionId("Session id must be a UUID") from exc

    def create(self, payload: Mapping[str, Any]) -> dict[str, str]:
        """Validate and atomically persist a new session."""

        filename, filetype, items = self._validate_new_payload(payload)
        with self._lock:
            session_id = self.canonicalize_id(str(self._id_factory()))
            path = self._path_for_canonical_id(session_id)
            if path.exists():
                raise SessionWriteError("Generated session id already exists")

            created_at = self._utc_timestamp()
            session = {
                "id": session_id,
                "filename": filename,
                "filetype": filetype,
                "created_at": created_at,
                "item_count": len(items),
                "items": items,
            }
            serialized = self._serialize(session)
            self._atomic_write(path, serialized)
        return {"id": session_id, "created_at": created_at}

    def get(self, session_id: str) -> dict[str, Any]:
        """Read and validate one session by UUID."""

        canonical_id = self.canonicalize_id(session_id)
        path = self._path_for_canonical_id(canonical_id)
        with self._lock:
            return self._read_path(path)

    def delete(self, session_id: str) -> None:
        """Delete one session by UUID without allowing path traversal."""

        canonical_id = self.canonicalize_id(session_id)
        path = self._path_for_canonical_id(canonical_id)
        with self._lock:
            try:
                path.unlink()
            except FileNotFoundError as exc:
                raise SessionNotFound("Session not found") from exc
            except OSError as exc:
                raise SessionDeleteError("Unable to delete session") from exc

    def scan(self) -> SessionScan:
        """Return valid summaries while reporting corrupt files."""

        sessions: list[dict[str, Any]] = []
        corrupt_count = 0
        with self._lock:
            files = sorted(self.root.glob("*.json"))
            for path in files:
                try:
                    record = self._read_path(path)
                except SessionNotFound:
                    # A different process may remove a file after the glob.
                    continue
                except CorruptSession as exc:
                    corrupt_count += 1
                    self._logger.warning(
                        "Skipping corrupt session file %s: %s", path.name, exc
                    )
                    continue
                sessions.append(
                    {
                        "id": record["id"],
                        "filename": record["filename"],
                        "filetype": record.get("filetype", ""),
                        "created_at": record["created_at"],
                        "item_count": record["item_count"],
                    }
                )

        sessions.sort(key=lambda item: item["created_at"], reverse=True)
        return SessionScan(
            sessions=sessions,
            corrupt_count=corrupt_count,
            total_files=len(files),
        )

    def _validate_new_payload(
        self, payload: Mapping[str, Any]
    ) -> tuple[str, str, list[dict[str, Any]]]:
        if not isinstance(payload, Mapping):
            raise InvalidSessionPayload("Request body must be a JSON object")

        filename = payload.get("filename", "unknown")
        filetype = payload.get("filetype", "")
        items = payload.get("items", [])

        self._validate_text("filename", filename, MAX_FILENAME_CHARS)
        self._validate_text("filetype", filetype, MAX_FILETYPE_CHARS)
        if not isinstance(items, list):
            raise InvalidSessionPayload("items must be a JSON array")
        if len(items) > self.max_items:
            raise SessionTooLarge(f"items cannot contain more than {self.max_items} entries")
        if any(not isinstance(item, dict) for item in items):
            raise InvalidSessionPayload("Every items entry must be a JSON object")

        return filename, filetype, list(items)

    @staticmethod
    def _validate_text(name: str, value: Any, max_chars: int) -> None:
        if not isinstance(value, str):
            raise InvalidSessionPayload(f"{name} must be a string")
        if len(value) > max_chars:
            raise InvalidSessionPayload(
                f"{name} cannot contain more than {max_chars} characters"
            )
        if "\x00" in value:
            raise InvalidSessionPayload(f"{name} cannot contain a null character")

    def _serialize(self, session: dict[str, Any]) -> str:
        try:
            serialized = json.dumps(
                session,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        except (OverflowError, RecursionError, TypeError, ValueError) as exc:
            raise InvalidSessionPayload("Session contains a non-JSON value") from exc

        encoded_size = len((serialized + "\n").encode("utf-8"))
        if encoded_size > self.max_bytes:
            raise SessionTooLarge(
                f"Encoded session cannot exceed {self.max_bytes} bytes"
            )
        return serialized

    def _atomic_write(self, path: Path, serialized: str) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=self.root,
                prefix=f".{path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(serialized)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        except OSError as exc:
            raise SessionWriteError("Unable to persist session atomically") from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    self._logger.warning(
                        "Unable to clean up session temporary file %s",
                        temporary_path.name,
                    )

    def _read_path(self, path: Path) -> dict[str, Any]:
        if path.is_symlink():
            raise CorruptSession("stored session cannot be a symbolic link")
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise SessionNotFound("Session not found") from exc
        except (OSError, UnicodeError) as exc:
            raise CorruptSession("stored data cannot be read") from exc

        try:
            record = json.loads(raw, parse_constant=self._reject_nonfinite_constant)
        except (RecursionError, UnicodeError, ValueError) as exc:
            raise CorruptSession("stored data is not valid JSON") from exc
        self._validate_stored_record(path, record)
        return record

    def _validate_stored_record(self, path: Path, record: Any) -> None:
        if not isinstance(record, dict):
            raise CorruptSession("stored session must be a JSON object")

        required = ("id", "filename", "created_at", "item_count", "items")
        missing = [name for name in required if name not in record]
        if missing:
            raise CorruptSession(f"stored session is missing {', '.join(missing)}")

        try:
            record_id = self.canonicalize_id(record["id"])
            filename_id = self.canonicalize_id(path.stem)
        except InvalidSessionId as exc:
            raise CorruptSession("stored session id is invalid") from exc
        if record_id != filename_id:
            raise CorruptSession("stored session id does not match its filename")
        if not isinstance(record["filename"], str):
            raise CorruptSession("stored filename must be a string")
        if not isinstance(record.get("filetype", ""), str):
            raise CorruptSession("stored filetype must be a string")
        if not isinstance(record["created_at"], str):
            raise CorruptSession("stored created_at must be a string")
        if isinstance(record["item_count"], bool) or not isinstance(
            record["item_count"], int
        ):
            raise CorruptSession("stored item_count must be an integer")
        if not isinstance(record["items"], list):
            raise CorruptSession("stored items must be a JSON array")
        if record["item_count"] != len(record["items"]):
            raise CorruptSession("stored item_count does not match items")
        if any(not isinstance(item, dict) for item in record["items"]):
            raise CorruptSession("stored items entries must be JSON objects")

    def _path_for_canonical_id(self, canonical_id: str) -> Path:
        return self.root / f"{canonical_id}.json"

    @staticmethod
    def _reject_nonfinite_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON value: {value}")

    def _utc_timestamp(self) -> str:
        value = self._clock()
        if value.tzinfo is None:
            raise SessionWriteError("Session clock must return a timezone-aware value")
        return value.astimezone(UTC).isoformat()
