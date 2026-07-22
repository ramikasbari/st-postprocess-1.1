"""
Strain computation from speckle-tracking control-point trajectories.

Two complementary routes are provided:

1. :func:`compute_local_strain` is a faithful NumPy port of the MATLAB
   ``getSTdataXY.m`` function. It builds a per-point local coordinate frame
   (radial / longitudinal-or-circumferential), applies drift correction, and
   returns displacement, velocity and the pointwise Lagrangian strain ratio in
   local coordinates. This reproduces the reference implementation exactly and
   is what per-segment / per-wall curves are derived from.

2. :func:`endocardial_length` / :func:`global_strain_curve` compute a
   numerically-stable *global* strain from the change in total endocardial wall
   length. This avoids the pointwise ratio ``tmp_f / tmp_1`` blowing up near the
   apex (where the reference longitudinal segment component approaches zero),
   which makes a naive spatial mean of the pointwise strain unreliable. The
   global length change is the definition clinical GLS/GCS is built on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .core import StrainSequence

# 90-degree rotation used to obtain the radial direction from the wall tangent.
_ROT90 = np.array([[0.0, -1.0], [1.0, 0.0]])


def apply_drift_correction(xy: np.ndarray, q1: int, q2: int) -> np.ndarray:
    """Remove linear drift so that each point returns to its start over a cycle.

    Mirrors the drift correction in ``getSTdataXY.m``: for every control point a
    linear ramp between the cycle-start (``q1``) and cycle-end (``q2``) residual
    is subtracted across all frames.

    Args:
        xy: ``(num_frames, num_cp, 2)`` positions in millimetres.
        q1: 0-based frame index of cycle start (Q1).
        q2: 0-based frame index of cycle end (Q2).

    Returns:
        A new drift-corrected array of the same shape.
    """
    num_frames = xy.shape[0]
    residual = xy[q2] - xy[q1]                       # (num_cp, 2)
    span = float(q2 - q1)
    if span == 0:
        return xy.copy()
    frames = np.arange(num_frames, dtype=float)      # (num_frames,)
    ramp = (frames - q1) / span                      # (num_frames,)
    # Outer product over frames x points x coords.
    return xy - ramp[:, None, None] * residual[None, :, :]


def _wall_tangent(xy_f: np.ndarray, p: int, num_cp: int, is_4ch: bool) -> np.ndarray:
    """Central-difference wall tangent at point ``p`` in a single frame.

    Matches the endpoint handling in ``getSTdataXY.m`` for both geometries.
    """
    if not is_4ch:  # SAX: closed ring, wrap the endpoints
        if p == 0:
            return (xy_f[1] - xy_f[num_cp - 1]) / 2.0
        if p == num_cp - 1:
            return (xy_f[0] - xy_f[p - 1]) / 2.0
        return (xy_f[p + 1] - xy_f[p - 1]) / 2.0
    # 4CH: open wall, one-sided differences at the ends
    if p == 0:
        return xy_f[1] - xy_f[0]
    if p == num_cp - 1:
        return xy_f[p] - xy_f[p - 1]
    return (xy_f[p + 1] - xy_f[p - 1]) / 2.0


def _strain_segment(xy_f: np.ndarray, p: int, num_cp: int, is_4ch: bool) -> np.ndarray:
    """Half-segment vector used for the pointwise strain ratio.

    Identical to the ``tmp_f`` / ``tmp_1`` construction in ``getSTdataXY.m``
    (note the 4CH endpoints are halved here, unlike the tangent above).
    """
    if not is_4ch:
        if p == 0:
            return (xy_f[1] - xy_f[num_cp - 1]) / 2.0
        if p == num_cp - 1:
            return (xy_f[0] - xy_f[p - 1]) / 2.0
        return (xy_f[p + 1] - xy_f[p - 1]) / 2.0
    if p == 0:
        return (xy_f[1] - xy_f[0]) / 2.0
    if p == num_cp - 1:
        return (xy_f[p] - xy_f[p - 1]) / 2.0
    return (xy_f[p + 1] - xy_f[p - 1]) / 2.0


@dataclass
class LocalStrainResult:
    """Pointwise fields in the local radial/longitudinal frame."""

    xy: np.ndarray            # (num_frames, num_cp, 2) drift-corrected positions
    displacement: np.ndarray  # (num_frames, num_cp, 2) [radial, long] vs Q1
    velocity: np.ndarray      # (num_frames, num_cp, 2) [radial, long]
    strain_ratio: np.ndarray  # (num_frames, num_cp) longitudinal stretch ratio (L/L0)
    time_interval: float      # seconds per frame

    def longitudinal_strain_percent(self) -> np.ndarray:
        """Pointwise Lagrangian strain in percent: ``(ratio - 1) * 100``."""
        return (self.strain_ratio - 1.0) * 100.0


def compute_local_strain(
    seq: StrainSequence,
    correct_drift: bool = True,
) -> LocalStrainResult:
    """Port of ``getSTdataXY.m`` returning pointwise local-frame quantities.

    Args:
        seq: A :class:`StrainSequence`. ``reference_frame`` / ``drift_end_frame``
            define the reference frame and drift span.
        correct_drift: Apply the linear drift correction (recommended, matches
            the MATLAB default ``options.correctDrift = 1``).

    Returns:
        A :class:`LocalStrainResult`.
    """
    num_frames, num_cp = seq.num_frames, seq.num_cp
    q1, q2 = seq.reference_frame, seq.drift_end_frame

    xy = seq.xy.astype(float)
    if correct_drift:
        xy = apply_drift_correction(xy, q1, q2)

    if seq.begin_time is not None and seq.end_time is not None and q2 != q1:
        time_interval = (seq.end_time - seq.begin_time) / (q2 - q1)
    else:
        time_interval = 1.0 / seq.frame_rate

    displacement = np.full((num_frames, num_cp, 2), np.nan)
    velocity = np.full((num_frames, num_cp, 2), np.nan)
    strain_ratio = np.full((num_frames, num_cp), np.nan)

    for f in range(num_frames):
        for p in range(num_cp):
            tangent = _wall_tangent(xy[f], p, num_cp, seq.is_4ch)
            norm = np.linalg.norm(tangent)
            if norm == 0:
                continue
            uo = tangent / norm
            ur = _ROT90 @ uo
            if seq.flip is False and not seq.is_4ch:
                ur = -ur
            if seq.is_4ch and p > num_cp // 2:
                uo = -uo
            frame_mat = np.column_stack([ur, uo])         # local -> Cartesian
            inv = np.linalg.inv(frame_mat)                 # Cartesian -> local

            # Displacement vs reference frame Q1.
            displacement[f, p] = inv @ (xy[f, p] - xy[q1, p])

            # Velocity (forward difference; last frame left as NaN).
            if f < num_frames - 1:
                velocity[f, p] = inv @ ((xy[f + 1, p] - xy[f, p]) / time_interval)

            # Pointwise Lagrangian strain ratio in local coordinates.
            seg_f = inv @ _strain_segment(xy[f], p, num_cp, seq.is_4ch)
            seg_1 = inv @ _strain_segment(xy[q1], p, num_cp, seq.is_4ch)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = seg_f / seg_1
            strain_ratio[f, p] = ratio[1]                  # longitudinal component

    return LocalStrainResult(
        xy=xy,
        displacement=displacement,
        velocity=velocity,
        strain_ratio=strain_ratio,
        time_interval=time_interval,
    )


def endocardial_length(xy_f: np.ndarray, is_4ch: bool) -> float:
    """Total endocardial wall length for one frame (millimetres).

    For 4CH the wall is an open chain (basal wall -> apex -> basal wall). For
    SAX the wall is a closed ring, so the closing segment is included.
    """
    diffs = np.diff(xy_f, axis=0)
    length = float(np.sqrt((diffs ** 2).sum(axis=1)).sum())
    if not is_4ch:
        length += float(np.linalg.norm(xy_f[0] - xy_f[-1]))
    return length


def global_strain_curve(
    seq: StrainSequence,
    correct_drift: bool = True,
) -> np.ndarray:
    """Numerically-stable global strain curve over the whole sequence.

    Computed as the relative change in total endocardial length with respect to
    the reference frame: ``(L(f) - L(ref)) / L(ref) * 100`` for every frame.
    Negative values indicate shortening (the physiological direction in
    systole).

    Args:
        seq: A :class:`StrainSequence`.
        correct_drift: Apply drift correction before measuring lengths.

    Returns:
        ``(num_frames,)`` array of strain in percent.
    """
    q1, q2 = seq.reference_frame, seq.drift_end_frame
    xy = seq.xy.astype(float)
    if correct_drift:
        xy = apply_drift_correction(xy, q1, q2)

    ref_length = endocardial_length(xy[q1], seq.is_4ch)
    if ref_length == 0:
        raise ValueError("Reference endocardial length is zero; cannot compute strain")

    lengths = np.array(
        [endocardial_length(xy[f], seq.is_4ch) for f in range(seq.num_frames)]
    )
    return (lengths - ref_length) / ref_length * 100.0
