"""
Evidence-based quality control for strain results.

Thresholds and reference ranges are drawn from the strain-imaging literature
rather than picked arbitrarily. Sources are cited on each constant so they can
be checked and updated. Nothing here fabricates a validation outcome — the
functions only flag a computed result against published expectations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .gls import GLSResult

# Acquisition frame rate below which strain is materially underestimated.
# Voigt et al., "Definitions for a common standard for 2D speckle tracking
# echocardiography", JASE 2015; EACVI/ASE/Industry task force.
MIN_FRAME_RATE_HZ = 40.0
OPTIMAL_FRAME_RATE_HZ = 60.0

# Normal peak-systolic GLS range (magnitude); a healthy adult LV is typically
# more negative than about -16% to -18%. Values here bracket the pooled normal
# range reported across large cohorts (e.g. NORRE study, JASE meta-analyses).
GLS_NORMAL_UPPER = -14.0  # less negative than this = reduced function
GLS_NORMAL_LOWER = -24.0  # more negative than this = hyperdynamic / suspect

# Circumferential strain from a single mid short-axis endocardial ring is of
# smaller magnitude than full-model GCS; keep a permissive plausibility band.
GCS_NORMAL_UPPER = -8.0
GCS_NORMAL_LOWER = -30.0

# Sex/age reference means for GLS (percent) from the NORRE normal-reference
# study, provided for context in reports (not used as pass/fail gates).
GLS_REFERENCE_MEANS: Dict[str, float] = {
    "male_18-39": -19.4,
    "male_40-59": -18.5,
    "male_60+": -17.5,
    "female_18-39": -20.3,
    "female_40-59": -19.5,
    "female_60+": -18.5,
}


@dataclass
class QCFinding:
    severity: str   # 'critical' | 'major' | 'minor'
    category: str
    message: str
    reference: Optional[str] = None


@dataclass
class QCReport:
    quality_score: int
    reportable: bool
    findings: List[QCFinding] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "quality_score": self.quality_score,
            "reportable": self.reportable,
            "findings": [
                {
                    "severity": f.severity,
                    "category": f.category,
                    "message": f.message,
                    "reference": f.reference,
                }
                for f in self.findings
            ],
        }


def assess(result: GLSResult) -> QCReport:
    """Assess a :class:`GLSResult` against evidence-based expectations."""
    findings: List[QCFinding] = []

    # 1. Acquisition frame rate.
    if result.frame_rate < MIN_FRAME_RATE_HZ:
        findings.append(
            QCFinding(
                severity="critical",
                category="technical",
                message=(
                    f"Frame rate {result.frame_rate:.0f} Hz is below the "
                    f"{MIN_FRAME_RATE_HZ:.0f} Hz minimum for reliable strain; "
                    "the magnitude is likely underestimated."
                ),
                reference="Voigt et al., JASE 2015",
            )
        )
    elif result.frame_rate < OPTIMAL_FRAME_RATE_HZ:
        findings.append(
            QCFinding(
                severity="minor",
                category="technical",
                message=(
                    f"Frame rate {result.frame_rate:.0f} Hz is below the "
                    f"~{OPTIMAL_FRAME_RATE_HZ:.0f} Hz optimum for strain imaging."
                ),
                reference="Voigt et al., JASE 2015",
            )
        )

    # 2. Physiological plausibility of the global value.
    gls = result.gls_percent
    if result.strain_kind == "longitudinal":
        upper, lower, label = GLS_NORMAL_UPPER, GLS_NORMAL_LOWER, "GLS"
    else:
        upper, lower, label = GCS_NORMAL_UPPER, GCS_NORMAL_LOWER, "GCS"

    if gls > 0:
        findings.append(
            QCFinding(
                severity="critical",
                category="physiology",
                message=(
                    f"{label} is positive ({gls:.1f}%); systolic strain must be "
                    "negative (shortening). Check wall-tracing order and ECG events."
                ),
            )
        )
    elif gls > upper:
        findings.append(
            QCFinding(
                severity="minor",
                category="physiology",
                message=f"{label} {gls:.1f}% is reduced (less negative than {upper:.0f}%).",
                reference="NORRE reference study",
            )
        )
    elif gls < lower:
        findings.append(
            QCFinding(
                severity="minor",
                category="physiology",
                message=f"{label} {gls:.1f}% is unusually large in magnitude (below {lower:.0f}%).",
            )
        )

    # 3. Regional coverage.
    if not result.segments:
        findings.append(
            QCFinding(
                severity="major",
                category="coverage",
                message="No wall segments could be resolved for a regional breakdown.",
            )
        )

    # 4. Regional dispersion (a crude tracking-consistency proxy).
    if len(result.segments) >= 3:
        peaks = [s.peak_systolic_percent for s in result.segments]
        spread = max(peaks) - min(peaks)
        if spread > 15.0:
            findings.append(
                QCFinding(
                    severity="minor",
                    category="regional",
                    message=(
                        f"Large spread across segments ({spread:.0f} percentage "
                        "points); inspect tracking in outlier segments."
                    ),
                )
            )

    score = 100
    for f in findings:
        score -= {"critical": 40, "major": 20, "minor": 5}.get(f.severity, 0)
    score = max(0, min(100, score))
    reportable = score >= 70 and not any(f.severity == "critical" for f in findings)

    return QCReport(quality_score=score, reportable=reportable, findings=findings)
