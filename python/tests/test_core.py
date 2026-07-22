"""
Tests for the vendor-neutral core: building a StrainSequence from raw points
with no ECHOPAC/DICOM concepts, exactly as an embedding program (e.g. ramireport)
would. Proves the strain library is source-agnostic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Deliberately import ONLY the neutral core surface — no reader/adapter.
from gls_analysis import StrainSequence, assess, compute_gls  # noqa: E402


def _open_wall(scale: float, n: int = 50) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, n)
    wall = np.column_stack([40.0 * t ** 2, t * 20.0])  # U-shape, apex at top
    c = wall.mean(axis=0)
    return (wall - c) * scale + c


def test_from_points_open_topology():
    scales = [1.0, 0.94, 0.88, 0.85, 0.9, 0.98]
    pts = np.stack([_open_wall(s) for s in scales])
    seq = StrainSequence.from_points(
        pts, frame_rate=50.0, topology="open",
        reference_frame=0, end_systole_frame=3, name="lv-a4c",
    )
    assert seq.num_frames == 6
    assert seq.strain_kind == "longitudinal"
    result = compute_gls(seq, correct_drift=False)
    assert result.metric_name == "GLS"
    assert abs(result.gls_percent - (-15.0)) < 0.6
    assert assess(result).quality_score > 0


def test_from_points_closed_topology():
    # A shrinking ring -> circumferential strain (GCS).
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    ring = np.column_stack([np.cos(theta), np.sin(theta)]) * 30.0
    scales = [1.0, 0.95, 0.90, 0.95, 1.0]
    pts = np.stack([ring * s for s in scales])
    seq = StrainSequence.from_points(
        pts, frame_rate=60.0, topology="closed",
        reference_frame=0, end_systole_frame=2,
    )
    assert seq.strain_kind == "circumferential"
    result = compute_gls(seq, correct_drift=False)
    assert result.metric_name == "GCS"
    assert abs(result.gls_percent - (-10.0)) < 0.6


def test_from_events_maps_markers():
    pts = np.stack([_open_wall(s) for s in [1.0, 0.9, 0.85, 0.9, 0.95, 1.0]])
    seq = StrainSequence.from_events(
        pts, 50.0, "open", events=[0, 1, 1, 2, 4, 5],
    )
    assert seq.reference_frame == 0
    assert seq.end_systole_frame == 2       # AVC
    assert seq.systole_search_end == 4      # MVO
    assert seq.drift_end_frame == 5         # Q2
    assert seq.ecg_events == [0, 1, 1, 2, 4, 5]


def test_validation_rejects_bad_shape():
    try:
        StrainSequence.from_points(np.zeros((5, 10)), 50.0)  # missing coord axis
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad points shape")


def test_marker_out_of_range_rejected():
    pts = np.stack([_open_wall(1.0) for _ in range(4)])
    try:
        StrainSequence.from_points(pts, 50.0, end_systole_frame=99)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out-of-range marker")


def _main():
    tests = [v for k, v in globals().items()
             if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
