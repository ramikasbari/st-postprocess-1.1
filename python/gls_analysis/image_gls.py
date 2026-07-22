"""
Image-based strain: masks / cine loop -> per-frame contours -> validated core.

Entry points, cheapest first:
    * :func:`analyze_masks` / :func:`masks_to_sequence` — you already have
      per-frame LV masks (e.g. from your own model); hand them straight in.
    * :func:`analyze_cine` — you have a :class:`~gls_analysis.cine.CineLoop` and
      a :class:`~gls_analysis.segmentation.Segmenter`; it segments then delegates.

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


def _contours_from_masks(
    masks: Sequence[np.ndarray], num_points: int
) -> tuple[List[Optional[np.ndarray]], np.ndarray]:
    """Extract an endocardial contour (or None) and the area of each mask."""
    contours: List[Optional[np.ndarray]] = []
    areas: List[float] = []
    for mask in masks:
        m = np.asarray(mask)
        area = float((m > 0).sum())
        areas.append(area)
        contours.append(
            mask_to_endocardial_contour(m, n_points=num_points) if area > 0 else None
        )
    return contours, np.asarray(areas)


def _fill_contour_gaps(contours: List[Optional[np.ndarray]]) -> List[np.ndarray]:
    """Replace any missing frame's contour with the previous valid one."""
    valid = [c for c in contours if c is not None]
    filled: List[np.ndarray] = []
    last = valid[0]
    for c in contours:
        last = c if c is not None else last
        filled.append(last)
    return filled


def masks_to_sequence(
    masks: Sequence[np.ndarray],
    frame_rate: float,
    ed_frame: Optional[int] = None,
    es_frame: Optional[int] = None,
    topology: str = OPEN,
    num_points: int = 100,
    name: str = "cine",
    view: Optional[str] = None,
    min_valid_fraction: float = 0.8,
) -> StrainSequence:
    """Turn a sequence of per-frame LV masks into a :class:`StrainSequence`.

    This is the direct hand-off for a program that already has segmentation
    masks (e.g. from its own model) and just wants strain — no ``CineLoop`` or
    ``Segmenter`` object required.

    Args:
        masks: One ``(H, W)`` binary LV blood-pool mask per frame.
        frame_rate: Hz.
        ed_frame: End-diastole frame (reference). Auto-detected from mask areas
            (largest cavity) when ``None``.
        es_frame: End-systole frame. Auto-detected (smallest cavity after ED)
            when ``None``.
        topology: ``"open"`` (longitudinal wall / GLS) or ``"closed"`` (ring / GCS).
        num_points: Wall resampling density.
        name: Sequence label.
        view: Optional acquisition-view label for reporting.
        min_valid_fraction: Minimum fraction of frames that must yield a usable
            contour, else :class:`RuntimeError`.

    Returns:
        A :class:`StrainSequence` ready for :func:`compute_gls`.

    Raises:
        ValueError: If fewer than 3 masks are given.
        RuntimeError: If too few masks produce a usable contour.
        ImportError: If OpenCV (needed for mask→contour) is not installed.
    """
    masks = list(masks)
    n = len(masks)
    if n < 3:
        raise ValueError("need at least 3 masks")

    contours, areas = _contours_from_masks(masks, num_points)
    n_valid = sum(c is not None for c in contours)
    if n_valid < min_valid_fraction * n:
        raise RuntimeError(
            f"only {n_valid}/{n} masks produced a usable contour; "
            "segmentation likely failed"
        )

    filled = _fill_contour_gaps(contours)

    if ed_frame is None or es_frame is None:
        auto_ed, auto_es = detect_ed_es_from_areas(areas)
        ed_frame = auto_ed if ed_frame is None else ed_frame
        es_frame = auto_es if es_frame is None else es_frame

    return contours_to_sequence(
        filled, frame_rate, ed_frame, es_frame,
        name=name, num_points=num_points, topology=topology, view=view,
    )


def analyze_masks(
    masks: Sequence[np.ndarray],
    frame_rate: float,
    ed_frame: Optional[int] = None,
    es_frame: Optional[int] = None,
    topology: str = OPEN,
    num_points: int = 100,
    name: str = "cine",
    view: Optional[str] = None,
) -> GLSResult:
    """Masks -> strain in one call (the ramireport hand-off).

    Equivalent to ``compute_gls(masks_to_sequence(...), correct_drift=False)``.
    Drift correction is off because image contours have no guaranteed cyclic
    closure. See :func:`masks_to_sequence` for arguments.
    """
    seq = masks_to_sequence(
        masks, frame_rate, ed_frame, es_frame,
        topology=topology, num_points=num_points, name=name, view=view,
    )
    return compute_gls(seq, n_segments=6, correct_drift=False)


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
    masks = [segmenter.segment(frames[t]) for t in range(cine.num_frames)]

    if cine.ecg_events is not None:
        ed, es = cine.ecg_events[0], cine.ecg_events[3]
    else:
        ed, es = cine.ed_frame, cine.es_frame  # may be None -> auto-detected

    return analyze_masks(
        masks, cine.frame_rate, ed_frame=ed, es_frame=es,
        topology=cine.topology, num_points=num_points,
        name=cine.patient_id or cine.view, view=cine.view,
    )
