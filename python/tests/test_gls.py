"""
Tests for the GLS analysis system, run against the repository's real sample
ECHOPAC exports in ``Data/``.

These assert *physiological* behaviour (correct sign, plausible magnitude,
stability) rather than exact numbers, so they act as regression + sanity gates.
Run with: ``python -m pytest python/tests`` or ``python python/tests/test_gls.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gls_analysis import (  # noqa: E402
    assess,
    compute_gls,
    compute_local_strain,
    global_strain_curve,
    parse_csv,
)

DATA = Path(__file__).resolve().parents[2] / "Data"

# (folder, csv, is_4ch, flip, ecg events [Q1 MVC AVO AVC MVO Q2], 0-based)
CASES = [
    ("VOL_0001", "VOL_0001_OFF_4CH.CSV", True, False, [13, 14, 17, 28, 31, 44]),
    ("VOL_0002", "VOL_0002_OFF_4CH.CSV", True, False, [27, 29, 33, 53, 57, 109]),
    ("SUB_0014", "SUB_0014_POST_SAX.CSV", False, True, [36, 39, 44, 74, 89, 121]),
    ("SUB_0015", "SUB_0015_POST_SAX.CSV", False, True, [31, 34, 38, 65, 74, 105]),
]


def _load(case):
    folder, csv, is_4ch, flip, events = case
    return parse_csv(DATA / folder / csv, is_4ch=is_4ch, flip=flip, ecg_events=events)


def test_parsing_shapes():
    for case in CASES:
        seq = _load(case)
        assert seq.xy.shape == (seq.num_frames, seq.num_cp, 2)
        assert seq.frame_rate > 0
        assert len(seq.ecg_events) == 6
        assert np.isfinite(seq.xy).all()


def test_global_strain_is_negative_and_stable():
    for case in CASES:
        seq = _load(case)
        curve = global_strain_curve(seq)
        assert np.isfinite(curve).all()
        # Reference frame (Q1) strain is ~0.
        assert abs(curve[seq.ecg_events[0]]) < 1e-6
        # Peak systolic strain is negative (shortening) and physiologically bounded.
        peak = curve.min()
        assert -40.0 < peak < 0.0, f"{seq.name}: implausible peak {peak:.1f}%"


def test_compute_gls_ranges():
    for case in CASES:
        seq = _load(case)
        result = compute_gls(seq)
        assert result.gls_percent < 0
        # Peak systolic magnitude should be >= end-systolic magnitude.
        assert result.gls_percent <= result.end_systolic_percent + 1e-6
        assert 0 <= result.peak_frame < result.num_frames
        assert len(result.segments) == 6
        # Every segment strain must be finite.
        for s in result.segments:
            assert np.isfinite(s.peak_systolic_percent)


def test_volunteer_gls_physiological():
    # Healthy volunteer apical views should give clearly negative GLS.
    seq = _load(CASES[0])
    result = compute_gls(seq)
    assert result.metric_name == "GLS"
    assert result.gls_percent < -8.0
    assert result.gls_percent > -30.0


def test_local_strain_port_matches_reference_frame():
    # The MATLAB port: at Q1 the displacement is zero and the strain ratio is 1.
    seq = _load(CASES[0])
    local = compute_local_strain(seq)
    q1 = seq.ecg_events[0]
    disp_q1 = local.displacement[q1]
    assert np.allclose(disp_q1[np.isfinite(disp_q1)], 0.0, atol=1e-6)
    ratio_q1 = local.strain_ratio[q1]
    assert np.allclose(ratio_q1[np.isfinite(ratio_q1)], 1.0, atol=1e-6)


def test_qc_flags_are_consistent():
    for case in CASES:
        seq = _load(case)
        result = compute_gls(seq)
        report = assess(result)
        assert 0 <= report.quality_score <= 100
        # No sample sequence should trip a 'critical' finding.
        assert not any(f.severity == "critical" for f in report.findings), (
            f"{seq.name}: unexpected critical finding"
        )


def _main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
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
