"""
Image-based strain: cine loop -> per-frame contours -> validated strain core.

This is the "Route 2" (segment-every-frame) path. Per-frame endocardial
contours are resampled to a common point count by arc length — which gives an
approximate point correspondence anchored at the mitral hinges and apex — and
packed into the exact ``(frames, points, 2)`` array the CSV pipeline uses. The
strain itself is then computed by the same tested code in
:mod:`gls_analysis.strain` / :mod:`gls_analysis.gls`.

What is and isn't trustworthy:
    * The strain computation (length-change GLS) is validated (see the CSV tests).
    * The *contours* it is fed come from a segmentation model and the mask->wall
      heuristic, neither of which is validated here. So a number out of this
      path is only as good as those upstream links — and for an apical-4-chamber
      model it is single-plane longitudinal strain, NOT full 3-view GLS.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .cine import CineLoop
from .core import OPEN, StrainSequence
from .gls import GLSResult, compute_gls
from .segmentation import Segmenter, mask_to_endocardial_contour


def contours_to_sequence(
    contours: Sequence[np.ndarray],
    frame_rate: float,
    ed_frame: int,
    es_frame: int,
    name: str = "cine",
    num_points: int = 100,
    topology: str = OPEN,
    view: Optional[str] = None,
) -> StrainSequence:
    """Pack per-frame endocardial contours into a :class:`StrainSequence`.

    Args:
        contours: One ordered ``(N_f, 2)`` contour per frame (hinge -> apex ->
            hinge for an open wall). Point counts may differ per frame; each is
            resampled to ``num_points`` by arc length for correspondence.
        frame_rate: Hz.
        ed_frame: End-diastole frame index (strain reference).
        es_frame: End-systole frame index.
        name: Sequence label.
        num_points: Common resampled point count.
        topology: ``"open"`` (longitudinal wall) or ``"closed"`` (ring).
        view: Optional acquisition-view label for reporting (e.g. ``"A4C"``).

    Returns:
        A :class:`StrainSequence` ready for :func:`compute_gls`.
    """
    from .segmentation import _resample_open_contour

    num_frames = len(contours)
    if num_frames < 3:
        raise ValueError("need at least 3 frames of contours")
    xy = np.stack([_resample_open_contour(np.asarray(c, float), num_points) for c in contours])

    return StrainSequence.from_points(
        points=xy,
        frame_rate=frame_rate,
        topology=topology,
        reference_frame=int(np.clip(ed_frame, 0, num_frames - 2)),
        end_systole_frame=int(np.clip(es_frame, 1, num_frames - 1)),
        name=name,
        view=view,
    )


def detect_ed_es_from_areas(areas: np.ndarray) -> tuple[int, int]:
    """Pick ED (max LV area) and ES (min area after ED) from an area curve."""
    areas = np.asarray(areas, dtype=float)
    ed = int(np.argmax(areas))
    tail = areas[ed:]
    es = ed + int(np.argmin(tail)) if len(tail) > 1 else int(np.argmin(areas))
    if es <= ed:
        es = min(len(areas) - 1, ed + 1)
    return ed, es


def analyze_cine(
    cine: CineLoop,
    segmenter: Segmenter,
    num_points: int = 100,
) -> GLSResult:
    """Full image path: segment every frame, extract wall, compute strain.

    Args:
        cine: The vendor-agnostic cine loop (from your DICOM reader adapter).
        segmenter: Any :class:`Segmenter` (e.g. :class:`EchoNetSegmenter`).
        num_points: Wall resampling density.

    Returns:
        A :class:`GLSResult`. For an A4C model this is single-plane 4-chamber
        longitudinal strain — report it as such, not as full GLS.

    Raises:
        RuntimeError: If too few frames yield a usable contour.
    """
    frames = cine.grayscale()
    contours: List[Optional[np.ndarray]] = []
    areas: List[float] = []
    for t in range(cine.num_frames):
        mask = segmenter.segment(frames[t])
        area = float(np.asarray(mask).sum())
        areas.append(area)
        contour = mask_to_endocardial_contour(mask, n_points=num_points) if area > 0 else None
        contours.append(contour)

    valid = [c for c in contours if c is not None]
    if len(valid) < 0.8 * cine.num_frames:
        raise RuntimeError(
            f"only {len(valid)}/{cine.num_frames} frames produced a usable "
            "contour; segmentation likely failed"
        )

    # Fill any gaps by repeating the previous valid contour (rare).
    filled: List[np.ndarray] = []
    last = valid[0]
    for c in contours:
        last = c if c is not None else last
        filled.append(last)

    if cine.ecg_events is not None:
        ed, es = cine.ecg_events[0], cine.ecg_events[3]
    elif cine.ed_frame is not None and cine.es_frame is not None:
        ed, es = cine.ed_frame, cine.es_frame
    else:
        ed, es = detect_ed_es_from_areas(np.asarray(areas))

    seq = contours_to_sequence(
        filled, cine.frame_rate, ed, es,
        name=cine.patient_id or cine.view, num_points=num_points,
        view=cine.view,
    )
    # No drift correction: image contours have no guaranteed cyclic closure.
    return compute_gls(seq, n_segments=6, correct_drift=False)
