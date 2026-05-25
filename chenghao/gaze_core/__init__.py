"""Gaze integration helpers for the chenghao frontend."""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve and append sibling shengwen/src directory to sys.path
_current_file = Path(__file__).resolve()
_gaze_core_dir = _current_file.parent
_chenghao_dir = _gaze_core_dir.parent
_lexigaze_root = _chenghao_dir.parent
_shengwen_src = _lexigaze_root / "shengwen" / "src"

if _shengwen_src.exists() and str(_shengwen_src) not in sys.path:
    sys.path.append(str(_shengwen_src))
