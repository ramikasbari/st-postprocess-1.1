"""
Vendor-neutral core data type for strain analysis.

:class:`StrainSequence` is the single structure the strain math operates on. It
knows nothing about ECHOPAC, DICOM, or any vendor — it is just tracked
myocardial points over time plus the frame markers strain needs. Every input
(ECHOPAC CSV, DICOM image segmentation, or your own program) builds one of
these and hands it to :func:`gls_analysis.compute_gls`.

Build one directly from points::

    from gls_analysis import StrainSequence

    seq = StrainSequence.from_points(
        points,                 # (num_frames, num_points, 2) array, any units
        frame_rate=50.0,
        topology="open",        # "open" = longitudinal wall, "closed" = ring
        reference_frame=0,      # end-diastole (strain reference)
        end_systole_frame=12,   # end-systole
    )

or from a full six-event ECG list (Q1, MVC, AVO, AVC, MVO, Q2)::

    seq = StrainSequence.from_events(points, 50.0, "open",
                                     events=[13, 14, 17, 28, 31, 44])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Topology of the tracked wall:
#   "open"   -> an open chain (e.g. apical longitudinal wall: base -> apex -> base)
#   "closed" -> a closed ring  (e.g. short-axis circumferential wall)
OPEN = "open"
CLOSED = "closed"

_STRAIN_KIND = {OPEN: "longitudinal", CLOSED: "circumferential"}


@dataclass
class StrainSequence:
    """Tracked myocardial contour points over a sequence of frames.

    Attributes:
        points: ``(num_frames, num_points, 2)`` coordinates. Point index must be
            consistent across frames (same material point / arc-length position).
        frame_rate: Frames per second (Hz).
        topology: ``"open"`` or ``"closed"`` (see module docstring).
        reference_frame: Frame used as the strain reference (end-diastole).
        end_systole_frame: End-systole frame (where the end-systolic value is read).
        systole_search_end: Last frame of the peak-systolic search window
            (e.g. mitral-valve opening); defaults to ``end_systole_frame``.
        drift_end_frame: Frame that should coincide with ``reference_frame`` over a
            full cycle (drift-correction span end); defaults to the last frame.
        name: Human-readable label.
        flip: Wall-tracing orientation flag (only affects the signed radial
            direction in the pointwise MATLAB-port strain, not global strain).
        events: Optional full six-event ECG list (Q1, MVC, AVO, AVC, MVO, Q2);
            when given it populates the four marker frames above.
        begin_time / end_time / es_time: Optional physical timing (seconds) used
            only for velocity in the pointwise port; derived from frame_rate if absent.
        metadata: Free-form extra fields (vendor, view label, patient id, ...).
        source_path: Optional provenance string.
    """

    points: np.ndarray
    frame_rate: float
    topology: str = OPEN
    reference_frame: int = 0
    end_systole_frame: Optional[int] = None
    systole_search_end: Optional[int] = None
    drift_end_frame: Optional[int] = None
    name: str = "sequence"
    flip: bool = False
    events: Optional[List[int]] = None
    begin_time: Optional[float] = None
    end_time: Optional[float] = None
    es_time: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    source_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float)
        if self.points.ndim != 3 or self.points.shape[2] != 2:
            raise ValueError(
                f"points must be (num_frames, num_points, 2); got {self.points.shape}"
            )
        if self.topology not in (OPEN, CLOSED):
            raise ValueError(f"topology must be '{OPEN}' or '{CLOSED}'; got {self.topology!r}")
        if self.frame_rate <= 0:
            raise ValueError(f"frame_rate must be positive; got {self.frame_rate}")

        n = self.num_frames
        if self.events is not None:
            if len(self.events) != 6:
                raise ValueError(f"events must have 6 entries; got {len(self.events)}")
            self.reference_frame = int(self.events[0])
            self.end_systole_frame = int(self.events[3])
            self.systole_search_end = int(self.events[4])
            self.drift_end_frame = int(self.events[5])

        if self.end_systole_frame is None:
            self.end_systole_frame = n - 1
        if self.systole_search_end is None:
            self.systole_search_end = self.end_systole_frame
        if self.drift_end_frame is None:
            self.drift_end_frame = n - 1

        for label, idx in (
            ("reference_frame", self.reference_frame),
            ("end_systole_frame", self.end_systole_frame),
            ("systole_search_end", self.systole_search_end),
            ("drift_end_frame", self.drift_end_frame),
        ):
            if not 0 <= idx < n:
                raise ValueError(f"{label}={idx} out of range [0, {n})")

    # --- convenience constructors ---------------------------------------- #

    @classmethod
    def from_points(
        cls,
        points: np.ndarray,
        frame_rate: float,
        topology: str = OPEN,
        reference_frame: int = 0,
        end_systole_frame: Optional[int] = None,
        systole_search_end: Optional[int] = None,
        name: str = "sequence",
        **metadata,
    ) -> "StrainSequence":
        """Build a sequence from raw tracked points (the general entry point)."""
        return cls(
            points=points,
            frame_rate=frame_rate,
            topology=topology,
            reference_frame=reference_frame,
            end_systole_frame=end_systole_frame,
            systole_search_end=systole_search_end,
            name=name,
            metadata=dict(metadata),
        )

    @classmethod
    def from_events(
        cls,
        points: np.ndarray,
        frame_rate: float,
        topology: str,
        events: List[int],
        name: str = "sequence",
        flip: bool = False,
        **kwargs,
    ) -> "StrainSequence":
        """Build a sequence from a six-event ECG list (Q1..Q2)."""
        return cls(
            points=points,
            frame_rate=frame_rate,
            topology=topology,
            events=list(events),
            name=name,
            flip=flip,
            **kwargs,
        )

    # --- derived / back-compatible views --------------------------------- #

    @property
    def num_frames(self) -> int:
        return int(self.points.shape[0])

    @property
    def num_points(self) -> int:
        return int(self.points.shape[1])

    # Aliases kept so existing code / adapters keep working.
    @property
    def num_cp(self) -> int:
        return self.num_points

    @property
    def xy(self) -> np.ndarray:
        return self.points

    @property
    def is_open(self) -> bool:
        return self.topology == OPEN

    @property
    def is_4ch(self) -> bool:  # legacy name: apical/open == longitudinal
        return self.topology == OPEN

    @property
    def geometry(self) -> str:
        """Legacy display label: '4CH' (open) or 'SAX' (closed)."""
        return "4CH" if self.topology == OPEN else "SAX"

    @property
    def strain_kind(self) -> str:
        return _STRAIN_KIND[self.topology]

    @property
    def ecg_events(self) -> Optional[List[int]]:
        return self.events
