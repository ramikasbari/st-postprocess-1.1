"""
Tests for the image-based strain glue (Route 2), using synthetic contours so no
segmentation model, torch, or cv2 is required. These validate the part that is
this repo's responsibility: turning a sequence of per-frame endocardial contours
into strain via the already-validated core.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gls_analysis import (  # noqa: E402
    CineLoop,
    analyze_cine,
    compute_gls,
    contours_to_sequence,
    detect_ed_es_from_areas,
)


def _u_wall(scale: float, n: int = 60) -> np.ndarray:
    """A U-shaped endocardial wall (hinge -> apex -> hinge), scaled by ``scale``.

    Scaling the geometry by ``s`` scales every length by ``s``, so the strain
    relative to the unscaled reference must be exactly ``(s - 1) * 100`` percent.
    """
    t = np.linspace(-1.0, 1.0, n)
    col = t * 20.0                 # lateral extent
    row = 40.0 - 40.0 * (1 - t ** 2)  # parabola: apex at top (row small)
    wall = np.column_stack([row, col])
    centroid = wall.mean(axis=0)
    return (wall - centroid) * scale + centroid


def _synthetic_sequence(scales):
    return [_u_wall(s) for s in scales]


def test_known_scaling_gives_known_strain():
    # ED (scale 1.0) -> ES (scale 0.85) should give -15% strain.
    scales = [1.0, 0.95, 0.90, 0.85, 0.90, 0.97]
    seq = contours_to_sequence(_synthetic_sequence(scales), frame_rate=50.0,
                               ed_frame=0, es_frame=3, num_points=100)
    result = compute_gls(seq, correct_drift=False)
    assert result.geometry == "4CH"
    assert abs(result.gls_percent - (-15.0)) < 0.5, result.gls_percent
    assert abs(result.end_systolic_percent - (-15.0)) < 0.5


def test_peak_systolic_is_most_negative_in_window():
    scales = [1.0, 0.9, 0.8, 0.82, 0.95, 1.0]  # peak shortening at frame 2
    seq = contours_to_sequence(_synthetic_sequence(scales), frame_rate=50.0,
                               ed_frame=0, es_frame=4, num_points=80)
    result = compute_gls(seq, correct_drift=False)
    assert result.peak_frame == 2
    assert abs(result.gls_percent - (-20.0)) < 0.7


def test_segments_present_and_finite():
    scales = [1.0, 0.9, 0.85, 0.9, 1.0]
    seq = contours_to_sequence(_synthetic_sequence(scales), frame_rate=40.0,
                               ed_frame=0, es_frame=2)
    result = compute_gls(seq, correct_drift=False)
    assert len(result.segments) == 6
    for s in result.segments:
        assert np.isfinite(s.peak_systolic_percent)


def test_detect_ed_es_from_areas():
    areas = np.array([100, 90, 70, 55, 68, 95, 100.0])
    ed, es = detect_ed_es_from_areas(areas)
    assert ed == 0
    assert es == 3


def test_analyze_cine_with_stub_segmenter():
    # A stub "segmenter" that renders each synthetic contour into a filled mask,
    # exercising the full analyze_cine path (incl. mask->contour) without any
    # real model. Needs cv2, which analyze_cine requires anyway; skip if absent.
    try:
        import cv2
    except ImportError:
        print("SKIP test_analyze_cine_with_stub_segmenter (cv2 not installed)")
        return

    scales = [1.0, 0.93, 0.86, 0.9, 0.98]
    walls = _synthetic_sequence(scales)
    H = W = 96

    def rasterize(wall):
        mask = np.zeros((H, W), dtype=np.uint8)
        pts = wall.copy()
        pts[:, 0] += 50  # shift into frame (row)
        pts[:, 1] += 48  # col
        poly = pts[:, ::-1].astype(np.int32)  # (row,col) -> (x,y) for cv2
        cv2.fillPoly(mask, [poly], 1)
        return mask

    class StubSegmenter:
        def __init__(self):
            self._i = 0

        def segment(self, frame):
            m = rasterize(walls[self._i % len(walls)])
            self._i += 1
            return m

    frames = np.zeros((len(walls), H, W), dtype=np.float32)
    cine = CineLoop(frames=frames, frame_rate=45.0, view="A4C",
                    ed_frame=0, es_frame=2)
    result = analyze_cine(cine, StubSegmenter())
    assert result.gls_percent < 0
    assert result.geometry == "4CH"


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
