"""
Reader for ECHOPAC (GE Healthcare) "store full trace" speckle-tracking exports.

This mirrors the parsing logic of ``Code/a1_ReadExportedData.m`` in the MATLAB
toolkit, but in pure Python/NumPy. The exported ``.CSV`` file contains, in order:

    * 7 free-text header lines (FName/LName/ID/Exam.Date/View/2DS Date/Knots note)
    * one line with ``FR=`` (frame rate, Hz) and the marker/ES times (seconds)
    * two more header lines (``Num Frames:  Knots:`` label + separators)
    * two integers: number of frames and number of knots (control points)
    * a flat comma-separated block of knot X/Y positions in millimetres,
      ordered frame-by-frame, that MATLAB reshapes as ``(2, numCP, numFrames)``.

For short-axis (SAX) acquisitions the exporter repeats the first knot as a
closing point, so the effective control-point count is ``numCP - 1`` (this is
what ``a1_ReadExportedData.m`` does when ``is4CH == 0``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from .core import CLOSED, OPEN, StrainSequence

# Six ECG events expected per sequence (see README, Recommendation C.2):
# Q1  = onset of QRS / start of cycle (reference / end-diastole)
# MVC = mitral valve closure
# AVO = aortic valve opening
# AVC = aortic valve closure (end-systole)
# MVO = mitral valve opening
# Q2  = onset of next QRS / end of cycle
ECG_EVENT_NAMES = ("Q1", "MVC", "AVO", "AVC", "MVO", "Q2")

_FLOAT_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")
_HEADER_RE = re.compile(
    r"FR=\s*([0-9.]+).*?"
    r"Left Marker Time=([0-9.]+).*?"
    r"Right Marker Time=([0-9.]+).*?"
    r"ES Time=([0-9.]+)",
    re.DOTALL,
)
_KNOTS_RE = re.compile(r"Num Frames:\s*Knots:[^\d]*([0-9]+)\s+([0-9]+)")


# ``STSequence`` used to be defined here; it is now the vendor-neutral
# :class:`~gls_analysis.core.StrainSequence`. The alias is kept so existing
# imports (``from gls_analysis.echopac_reader import STSequence``) keep working.
STSequence = StrainSequence


def parse_csv(
    path: str | Path,
    is_4ch: bool,
    flip: bool = False,
    ecg_events: Optional[List[int]] = None,
    name: Optional[str] = None,
) -> StrainSequence:
    """Parse one ECHOPAC ``.CSV`` export into a :class:`StrainSequence`.

    Args:
        path: Path to the ``.CSV`` file.
        is_4ch: ``True`` for apical (4CH-style, open wall) acquisitions,
            ``False`` for short-axis (closed ring) acquisitions.
        flip: Wall-tracing orientation flag from the subject registry. Only
            affects the signed radial direction in the full strain computation.
        ecg_events: Optional list of six 0-based frame indices in the order
            ``(Q1, MVC, AVO, AVC, MVO, Q2)``. When omitted, phase handling
            falls back to the exporter's ES-time marker.
        name: Optional human-readable label; defaults to the file stem.

    Returns:
        A populated :class:`StrainSequence`.

    Raises:
        ValueError: If the header, knot counts, or coordinate block cannot be
            parsed, or if the coordinate block is shorter than expected.
    """
    path = Path(path)
    raw = path.read_text(errors="replace")

    header = _HEADER_RE.search(raw)
    if header is None:
        raise ValueError(f"{path}: could not locate 'FR=/... ES Time=' header line")
    frame_rate, begin_time, end_time, es_time = (float(g) for g in header.groups())

    knots = _KNOTS_RE.search(raw)
    if knots is None:
        raise ValueError(f"{path}: could not locate 'Num Frames: Knots:' counts")
    num_frames, num_cp = int(knots.group(1)), int(knots.group(2))

    tail = raw[knots.end():]
    values = np.array([float(v) for v in _FLOAT_RE.findall(tail)], dtype=float)
    expected = 2 * num_cp * num_frames
    if values.size < expected:
        raise ValueError(
            f"{path}: expected {expected} coordinate values "
            f"({num_frames} frames x {num_cp} knots x 2), found {values.size}"
        )
    # Stream is ordered frame -> knot -> (x, y); reshape accordingly.
    xy = values[:expected].reshape(num_frames, num_cp, 2)

    if not is_4ch:
        # SAX export repeats the first knot as a closing point (see MATLAB).
        num_cp -= 1
        xy = xy[:, :num_cp, :]

    if ecg_events is not None:
        ecg_events = _validate_events(list(ecg_events), num_frames, path)

    topology = OPEN if is_4ch else CLOSED
    return StrainSequence(
        points=xy,
        frame_rate=frame_rate,
        topology=topology,
        events=ecg_events,
        name=name or path.stem,
        flip=bool(flip),
        begin_time=begin_time,
        end_time=end_time,
        es_time=es_time,
        metadata={"vendor": "ECHOPAC", "view": ("4CH" if is_4ch else "SAX")},
        source_path=str(path),
    )


def _validate_events(events: List[int], num_frames: int, path: Path) -> List[int]:
    if len(events) != len(ECG_EVENT_NAMES):
        raise ValueError(
            f"{path}: expected {len(ECG_EVENT_NAMES)} ECG events "
            f"{ECG_EVENT_NAMES}, got {len(events)}"
        )
    for name, idx in zip(ECG_EVENT_NAMES, events):
        if not 0 <= idx < num_frames:
            raise ValueError(
                f"{path}: ECG event {name}={idx} outside frame range "
                f"[0, {num_frames})"
            )
    return events


# --------------------------------------------------------------------------- #
# Subject registry (DEMO_DATA.xls) reader
# --------------------------------------------------------------------------- #

@dataclass
class SubjectEntry:
    """One row of the subject registry spreadsheet."""

    sheet: str
    population: str          # e.g. 'VOL', 'SUB'
    series_number: int       # e.g. 1, 14
    status: str              # e.g. 'OFF', 'POST'
    view: str                # e.g. '4CH', 'SAX'
    ecg_events: List[int]    # 0-based frame indices, len == 6
    is_4ch: bool
    flip: bool

    @property
    def folder_name(self) -> str:
        """Subject folder, e.g. 'VOL_0001' or 'SUB_0014'."""
        return f"{self.population}_{self.series_number:04d}"

    @property
    def csv_name(self) -> str:
        """Expected CSV file name, e.g. 'VOL_0001_OFF_4CH.CSV'."""
        return f"{self.folder_name}_{self.status}_{self.view}.CSV"


def read_registry(xls_path: str | Path) -> List[SubjectEntry]:
    """Read ``DEMO_DATA.xls`` (or a compatible registry) into subject entries.

    Column order mirrors ``a1_ReadExportedData.m``: population, series number,
    status, view, six ECG events, an ``is4CH`` flag and a ``Flip`` flag.

    The MATLAB reader adds ``+1`` to each ECG value to convert to 1-based
    frame indices; here we keep the raw spreadsheet values, which are already
    the correct 0-based Python frame indices.

    Requires the optional ``xlrd`` package for legacy ``.xls`` files.
    """
    try:
        import xlrd  # noqa: WPS433 (optional dependency)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading the .xls registry requires the 'xlrd' package "
            "(pip install xlrd). Alternatively, pass ECG events explicitly."
        ) from exc

    book = xlrd.open_workbook(str(xls_path))
    entries: List[SubjectEntry] = []
    for sheet in book.sheets():
        for r in range(1, sheet.nrows):  # row 0 is the header
            row = [sheet.cell_value(r, c) for c in range(sheet.ncols)]
            if not str(row[0]).strip():
                continue
            events = [int(float(v)) for v in row[4:10]]
            entries.append(
                SubjectEntry(
                    sheet=sheet.name,
                    population=str(row[0]).strip(),
                    series_number=int(float(row[1])),
                    status=str(row[2]).strip(),
                    view=str(row[3]).strip(),
                    ecg_events=events,
                    is_4ch=bool(int(float(row[10]))),
                    flip=bool(int(float(row[11]))),
                )
            )
    return entries


def load_sequence(data_root: str | Path, entry: SubjectEntry) -> StrainSequence:
    """Load the CSV referenced by a registry entry, wiring in its ECG events."""
    data_root = Path(data_root)
    csv_path = data_root / entry.folder_name / entry.csv_name
    return parse_csv(
        csv_path,
        is_4ch=entry.is_4ch,
        flip=entry.flip,
        ecg_events=entry.ecg_events,
        name=f"{entry.folder_name}_{entry.status}_{entry.view}",
    )
