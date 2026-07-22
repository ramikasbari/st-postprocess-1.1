"""
gls_analysis
============

A pure-Python/NumPy GLS (Global Longitudinal Strain) analysis system for the
speckle-tracking post-processing toolkit, operating on ECHOPAC "store full
trace" ``.CSV`` exports.

Pipeline::

    from gls_analysis import parse_csv, compute_gls, assess

    seq = parse_csv("Data/VOL_0001/VOL_0001_OFF_4CH.CSV",
                    is_4ch=True, ecg_events=[13, 14, 17, 28, 31, 44])
    result = compute_gls(seq)
    report = assess(result)
    print(result.metric_name, result.gls_percent)

The strain math is a faithful port of the toolkit's MATLAB ``getSTdataXY.m``
(see :mod:`gls_analysis.strain`), with a numerically-stable global strain built
on endocardial-length change.
"""

from .echopac_reader import (
    ECG_EVENT_NAMES,
    STSequence,
    SubjectEntry,
    load_sequence,
    parse_csv,
    read_registry,
)
from .gls import GLSResult, SegmentStrain, compute_gls
from .quality import QCReport, assess
from .strain import (
    LocalStrainResult,
    compute_local_strain,
    endocardial_length,
    global_strain_curve,
)

__all__ = [
    "ECG_EVENT_NAMES",
    "STSequence",
    "SubjectEntry",
    "parse_csv",
    "read_registry",
    "load_sequence",
    "compute_local_strain",
    "LocalStrainResult",
    "global_strain_curve",
    "endocardial_length",
    "compute_gls",
    "GLSResult",
    "SegmentStrain",
    "assess",
    "QCReport",
]

__version__ = "1.0.0"
