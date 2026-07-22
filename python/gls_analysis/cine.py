"""
Cine-loop container: the seam between an (external) DICOM reader and the
image-based strain pipeline.

This module deliberately does NOT read DICOM. You already have a DICOM reader;
adapt its output into a :class:`CineLoop` and the rest of the pipeline
(segmentation -> contour -> strain) consumes it. Keeping this boundary thin and
explicit is what lets any vendor's export flow through unchanged.

A minimal adapter looks like::

    from gls_analysis.cine import CineLoop

    def cineloop_from_my_reader(study) -> CineLoop:
        return CineLoop(
            frames=study.pixels,            # (T, H, W) grayscale, float or uint8
            frame_rate=study.fps,           # Hz
            pixel_spacing=study.spacing_mm,  # (row_mm, col_mm) or None
            view="A4C",
            ed_frame=study.ed_index,        # int or None
            es_frame=study.es_index,        # int or None
        )

Note on calibration: GLS is a *ratio* of lengths, so it is invariant to
isotropic pixel scaling — ``pixel_spacing`` is optional and only needed if you
also want absolute chamber lengths. Anisotropic resizing (e.g. squashing a
non-square frame to 112x112) *does* distort geometry, so resize preserving
aspect ratio or work in original-frame coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CineLoop:
    """A single 2D echo cine loop, vendor-agnostic.

    Attributes:
        frames: ``(T, H, W)`` grayscale or ``(T, H, W, 3)`` frames. Any dtype;
            segmenters typically normalise internally.
        frame_rate: Acquisition frame rate in Hz.
        pixel_spacing: ``(row_mm, col_mm)`` physical spacing, or ``None``
            (strain is scale-invariant, so this is optional).
        view: Acoustic view label, e.g. ``"A4C"`` (EchoNet-Dynamic is A4C only).
        topology: ``"open"`` for apical/longitudinal walls (GLS), ``"closed"``
            for short-axis rings (GCS).
        ed_frame: End-diastole frame index, or ``None`` to auto-detect from the
            segmentation area curve.
        es_frame: End-systole frame index, or ``None`` to auto-detect.
        ecg_events: Optional six 0-based indices (Q1, MVC, AVO, AVC, MVO, Q2);
            when present these override ed/es.
        patient_id: Optional identifier for reporting.
        vendor: Optional acquisition vendor, for per-vendor QC/analysis.
        metadata: Free-form extra fields.
    """

    frames: np.ndarray
    frame_rate: float
    pixel_spacing: Optional[Tuple[float, float]] = None
    view: str = "A4C"
    topology: str = "open"  # "open" (longitudinal, apical) | "closed" (SAX ring)
    ed_frame: Optional[int] = None
    es_frame: Optional[int] = None
    ecg_events: Optional[List[int]] = None
    patient_id: Optional[str] = None
    vendor: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames)
        if self.frames.ndim not in (3, 4):
            raise ValueError(
                f"frames must be (T,H,W) or (T,H,W,3); got shape {self.frames.shape}"
            )
        if self.frame_rate <= 0:
            raise ValueError(f"frame_rate must be positive; got {self.frame_rate}")

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    def grayscale(self) -> np.ndarray:
        """Return frames as ``(T, H, W)`` grayscale float32 in [0, 1]."""
        f = self.frames.astype(np.float32)
        if f.ndim == 4:  # (T, H, W, C) -> luminance
            f = f[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        fmin = f.min()
        fmax = f.max()
        if fmax > fmin:
            f = (f - fmin) / (fmax - fmin)
        return f
