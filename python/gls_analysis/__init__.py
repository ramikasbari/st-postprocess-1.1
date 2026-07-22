"""
gls_analysis
============

A source-agnostic strain (GLS / GCS) analysis library. The core operates on a
vendor-neutral :class:`StrainSequence` — tracked myocardial points over time —
so it can be embedded in any program. Input adapters (ECHOPAC CSV, DICOM image
segmentation) are optional and independent of the core.

Minimal, source-independent usage::

    from gls_analysis import StrainSequence, compute_gls, assess

    seq = StrainSequence.from_points(
        points,               # (num_frames, num_points, 2)
        frame_rate=50.0,
        topology="open",      # "open" = longitudinal, "closed" = circumferential
        reference_frame=0,    # end-diastole
        end_systole_frame=12,
    )
    result = compute_gls(seq)
    report = assess(result)
    print(result.metric_name, result.gls_percent)   # e.g. GLS -18.3

Optional input adapters:

* :func:`parse_csv` / :func:`read_registry` — ECHOPAC "store full trace" CSV.
* :func:`analyze_cine` (+ :class:`CineLoop`, :class:`EchoNetSegmenter`) — the
  DICOM/image path: segment each frame, extract the wall, then feed the same core.
"""

# --- vendor-neutral core (the public heart of the library) --------------- #
from .core import CLOSED, OPEN, StrainSequence
from .gls import GLSResult, SegmentStrain, compute_gls
from .quality import QCReport, assess
from .strain import (
    LocalStrainResult,
    compute_local_strain,
    endocardial_length,
    global_strain_curve,
)

# --- optional input adapters --------------------------------------------- #
from .cine import CineLoop
from .echopac_reader import (
    ECG_EVENT_NAMES,
    STSequence,  # backward-compatible alias of StrainSequence
    SubjectEntry,
    load_sequence,
    parse_csv,
    read_registry,
)
from .image_gls import (
    analyze_cine,
    analyze_masks,
    contours_to_sequence,
    detect_ed_es_from_areas,
    masks_to_sequence,
)
from .segmentation import (
    EchoNetSegmenter,
    Segmenter,
    mask_to_endocardial_contour,
)

__all__ = [
    # Neutral core
    "StrainSequence",
    "OPEN",
    "CLOSED",
    "compute_gls",
    "GLSResult",
    "SegmentStrain",
    "assess",
    "QCReport",
    "global_strain_curve",
    "endocardial_length",
    "compute_local_strain",
    "LocalStrainResult",
    # ECHOPAC adapter
    "parse_csv",
    "read_registry",
    "load_sequence",
    "SubjectEntry",
    "ECG_EVENT_NAMES",
    "STSequence",
    # Image (DICOM) adapter
    "CineLoop",
    "Segmenter",
    "EchoNetSegmenter",
    "mask_to_endocardial_contour",
    "analyze_masks",
    "masks_to_sequence",
    "analyze_cine",
    "contours_to_sequence",
    "detect_ed_es_from_areas",
]

__version__ = "2.0.0"
