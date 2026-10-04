"""
Bridge to the inherited IIC model package (V1).

V2 is a planner, not a controller: it simulates a water balance but ships no
learned decision model. V1 does, and its bundle is the operator-facing product
this dashboard was built around. Merging the two means V2's physics and calendar
answer *when* and *how much* water the profile has, while V1's estimators answer
*should we irrigate, and at what risk* — on the same field.

The ML package is not vendored into V2. It is imported from the sibling project
(`MP3/backend/ml`) exactly as `irrigation.physics.climate` already imports the
validated NASA POWER client and FAO-56 ET0 implementation from there. Copying a
3.6 MB artifact would create a second source of truth that could silently drift
from the trained models, which is the one thing this codebase repeatedly refuses
to do. The path is explicit and fails loudly at import time if the sibling is
missing.

V1's module layout is flat (`ml` puts itself on `sys.path` and its scripts do
`from features import ...`), so the directory is added to the path before the
imports below, matching V1's own bridge.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Repository root of THIS project (`MP3-V2/backend`). This file is
#: `backend/app/core/ml_bridge.py`, so the backend directory is two parents up.
_BACKEND_DIR = Path(__file__).resolve().parents[2]

#: Where the ML package lives. The copy inside this repository is preferred and is
#: the one a clone gets; the sibling MP3 checkout is kept only as a fallback so a
#: machine that still has both trees side by side keeps working.
#:
#: This used to list the sibling first, with an absolute path, which made the
#: project unstartable from a clean clone: the decision bundle that answers "should
#: we irrigate" resolved to a directory that existed only where two unrelated
#: checkouts happened to sit next to each other. The absolute candidate is gone for
#: the same reason - a path that is correct on exactly one machine is not a
#: dependency, it is a coincidence.
_CANDIDATES = (
    _BACKEND_DIR / "ml",
    _BACKEND_DIR.parent / "MP3" / "backend" / "ml",
)
V1_ML_DIR = next((path for path in _CANDIDATES if path.is_dir()), _CANDIDATES[0])

if str(V1_ML_DIR) not in sys.path:
    sys.path.insert(0, str(V1_ML_DIR))

from features import NUTRIENT_BANDS  # noqa: E402
from inference import (  # noqa: E402
    BUNDLE_PATH,
    CROP_PROFILES,
    SOILS,
    Controller,
    Decision,
    DeviceConfig,
    ModelNotTrainedError,
    Reading,
    get_controller,
    iso_timestamp,
    load_metrics_summary,
)

__all__ = [
    "V1_ML_DIR",
    "BUNDLE_PATH",
    "NUTRIENT_BANDS",
    "Controller",
    "Decision",
    "DeviceConfig",
    "ModelNotTrainedError",
    "Reading",
    "get_controller",
    "iso_timestamp",
    "load_metrics_summary",
    "SOILS",
    "CROP_PROFILES",
]
