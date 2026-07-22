"""
Global and segmental strain (GLS / GCS) from speckle-tracking sequences.

The headline metric is *peak systolic* strain — the most negative value of the
global endocardial-length strain curve within the systolic window (Q1 to mitral
valve opening). The value at aortic valve closure (end-systole) is also reported,
as is a per-segment breakdown along the wall.

For apical (4CH) geometry this is Global Longitudinal Strain (GLS); for
short-axis geometry it is Global Circumferential Strain (GCS). The computation is
identical — only the wall topology (open vs closed) and the labels differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .echopac_reader import STSequence
from .strain import (
    apply_drift_correction,
    endocardial_length,
    global_strain_curve,
)

# Wall-segment labels, in tracing order (basal-septal first for 4CH per the
# README protocol; anteroseptal first for SAX).
SEGMENT_LABELS: Dict[str, List[str]] = {
    "4CH": [
        "basal septal",
        "mid septal",
        "apical septal",
        "apical lateral",
        "mid lateral",
        "basal lateral",
    ],
    "SAX": [
        "anteroseptal",
        "inferoseptal",
        "inferior",
        "inferolateral",
        "anterolateral",
        "anterior",
    ],
}


@dataclass
class SegmentStrain:
    """Peak-systolic strain for a single wall segment."""

    index: int
    label: str
    peak_systolic_percent: float
    end_systolic_percent: float


@dataclass
class GLSResult:
    """Complete global/segmental strain result for one sequence."""

    name: str
    geometry: str                 # '4CH' or 'SAX'
    strain_kind: str              # 'longitudinal' or 'circumferential'
    frame_rate: float
    num_frames: int
    gls_percent: float            # peak-systolic global strain (headline metric)
    end_systolic_percent: float   # global strain at aortic valve closure
    peak_frame: int               # frame index of the peak-systolic value
    curve_percent: np.ndarray     # (num_frames,) full global strain curve
    segments: List[SegmentStrain] = field(default_factory=list)
    ecg_events: Optional[List[int]] = None

    @property
    def metric_name(self) -> str:
        return "GLS" if self.geometry == "4CH" else "GCS"

    def as_dict(self) -> Dict:
        """JSON-serialisable summary (drops the full curve array)."""
        return {
            "name": self.name,
            "geometry": self.geometry,
            "metric": self.metric_name,
            "strain_kind": self.strain_kind,
            "frame_rate": self.frame_rate,
            "num_frames": self.num_frames,
            f"{self.metric_name.lower()}_percent": round(self.gls_percent, 2),
            "end_systolic_percent": round(self.end_systolic_percent, 2),
            "peak_frame": self.peak_frame,
            "segments": [
                {
                    "index": s.index,
                    "label": s.label,
                    "peak_systolic_percent": round(s.peak_systolic_percent, 2),
                    "end_systolic_percent": round(s.end_systolic_percent, 2),
                }
                for s in self.segments
            ],
        }


def _systolic_window(seq: STSequence) -> tuple[int, int]:
    """Frame range [start, end] to search for the peak systolic value.

    Uses Q1 (cycle start) to MVO (mitral valve opening) when ECG events are
    available; otherwise brackets the exporter's ES-time marker.
    """
    if seq.ecg_events is not None:
        q1, mvo = seq.ecg_events[0], seq.ecg_events[4]
        return q1, max(q1 + 1, mvo)
    # Fallback: use the ES-time marker relative to the left marker.
    es_frame = int(round((seq.es_time - seq.begin_time) * seq.frame_rate))
    es_frame = int(np.clip(es_frame, 1, seq.num_frames - 1))
    return 0, es_frame


def _end_systole_frame(seq: STSequence) -> int:
    """Frame index of end-systole (aortic valve closure, or ES-time fallback)."""
    if seq.ecg_events is not None:
        return seq.ecg_events[3]  # AVC
    es_frame = int(round((seq.es_time - seq.begin_time) * seq.frame_rate))
    return int(np.clip(es_frame, 0, seq.num_frames - 1))


def _segment_curves(seq: STSequence, n_segments: int, correct_drift: bool) -> np.ndarray:
    """Per-segment strain curves via arc length, shape ``(n_segments, num_frames)``.

    The wall is partitioned into ``n_segments`` contiguous pieces of equal
    reference (Q1) arc length. Each piece's length change over the cycle gives a
    regional strain curve.
    """
    q1 = seq.ecg_events[0] if seq.ecg_events is not None else 0
    q2 = seq.ecg_events[-1] if seq.ecg_events is not None else seq.num_frames - 1
    xy = seq.xy.astype(float)
    if correct_drift:
        xy = apply_drift_correction(xy, q1, q2)

    # Cumulative reference arc length along the wall (open chain).
    ref = xy[q1]
    seg_lengths = np.sqrt((np.diff(ref, axis=0) ** 2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumulative[-1]

    # Boundaries in arc-length space, mapped to nearest control-point indices.
    edges = np.linspace(0.0, total, n_segments + 1)
    bounds = [int(np.argmin(np.abs(cumulative - e))) for e in edges]
    bounds[0], bounds[-1] = 0, seq.num_cp - 1

    curves = np.full((n_segments, seq.num_frames), np.nan)
    for s in range(n_segments):
        lo, hi = bounds[s], bounds[s + 1]
        if hi <= lo:
            continue
        ref_len = endocardial_length_open(xy[q1][lo : hi + 1])
        if ref_len == 0:
            continue
        for f in range(seq.num_frames):
            length = endocardial_length_open(xy[f][lo : hi + 1])
            curves[s, f] = (length - ref_len) / ref_len * 100.0
    return curves


def endocardial_length_open(points: np.ndarray) -> float:
    """Arc length of an open poly-line of control points (millimetres)."""
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())


def compute_gls(
    seq: STSequence,
    n_segments: int = 6,
    correct_drift: bool = True,
) -> GLSResult:
    """Compute global and segmental strain for one sequence.

    Args:
        seq: A parsed :class:`STSequence`.
        n_segments: Number of wall segments for the regional breakdown
            (6 by convention for a single apical or short-axis view).
        correct_drift: Apply drift correction before measuring lengths.

    Returns:
        A populated :class:`GLSResult`. The headline ``gls_percent`` is the
        peak systolic value of the global strain curve.
    """
    curve = global_strain_curve(seq, correct_drift=correct_drift)

    lo, hi = _systolic_window(seq)
    window = curve[lo : hi + 1]
    peak_offset = int(np.nanargmin(window))
    peak_frame = lo + peak_offset
    gls = float(curve[peak_frame])

    es_frame = _end_systole_frame(seq)
    end_systolic = float(curve[es_frame])

    labels = SEGMENT_LABELS.get(seq.geometry, [f"segment {i+1}" for i in range(n_segments)])
    seg_curves = _segment_curves(seq, n_segments, correct_drift)
    segments: List[SegmentStrain] = []
    for s in range(n_segments):
        sc = seg_curves[s]
        if np.all(np.isnan(sc)):
            continue
        seg_window = sc[lo : hi + 1]
        seg_peak = float(np.nanmin(seg_window))
        seg_es = float(sc[es_frame])
        label = labels[s] if s < len(labels) else f"segment {s + 1}"
        segments.append(
            SegmentStrain(
                index=s + 1,
                label=label,
                peak_systolic_percent=seg_peak,
                end_systolic_percent=seg_es,
            )
        )

    return GLSResult(
        name=seq.name,
        geometry=seq.geometry,
        strain_kind=seq.strain_kind,
        frame_rate=seq.frame_rate,
        num_frames=seq.num_frames,
        gls_percent=gls,
        end_systolic_percent=end_systolic,
        peak_frame=peak_frame,
        curve_percent=curve,
        segments=segments,
        ecg_events=seq.ecg_events,
    )
