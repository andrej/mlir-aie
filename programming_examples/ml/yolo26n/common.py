"""Shared device definitions for the yolo26n builders.

XCVE3858 (Telluride / AIE2PS) is guarded because it is only present in
newer mlir-aie installs. When it is absent:
  - XCVE3858 is a sentinel class so isinstance(..., XCVE3858) is always
    False, and any compute_row_start logic falls back to the non-Telluride
    default (row 2).
  - "telluride" is omitted from ``devs`` so --device telluride is not
    offered as a CLI option (the device is functionally unavailable).
"""

from __future__ import annotations

from aie.iron.device import NPU2

try:
    from aie.iron.device import XCVE3858
    devs: dict = {"strix": NPU2(), "telluride": XCVE3858()}
except ImportError:
    XCVE3858 = type("XCVE3858", (), {})  # sentinel: isinstance() always False
    devs: dict = {"strix": NPU2()}
