"""
Command-line interface for the GLS analysis system.

Examples::

    # Process the whole demo registry (DEMO_DATA.xls) under Data/
    python -m gls_analysis.cli --data ../Data

    # Process a single CSV with explicit ECG events (Q1 MVC AVO AVC MVO Q2)
    python -m gls_analysis.cli --csv ../Data/VOL_0001/VOL_0001_OFF_4CH.CSV \
        --4ch --events 13 14 17 28 31 44

    # Emit machine-readable JSON
    python -m gls_analysis.cli --data ../Data --json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .echopac_reader import load_sequence, parse_csv, read_registry
from .gls import GLSResult, compute_gls
from .quality import assess


def _process_sequence(seq) -> dict:
    result = compute_gls(seq)
    report = assess(result)
    return {"result": result, "qc": report}


def _print_row(result: GLSResult, qc) -> None:
    segs = ", ".join(
        f"{s.label}={s.peak_systolic_percent:+.1f}" for s in result.segments
    )
    print(f"  {result.name}  [{result.geometry}, {result.frame_rate:.0f} Hz]")
    print(
        f"    {result.metric_name} (peak systolic): {result.gls_percent:+.2f}%   "
        f"end-systolic: {result.end_systolic_percent:+.2f}%   "
        f"QC: {qc.quality_score}/100 "
        f"({'reportable' if qc.reportable else 'review'})"
    )
    if segs:
        print(f"    segments: {segs}")
    for f in qc.findings:
        print(f"    [{f.severity}] {f.message}")


def _run_registry(data_root: Path) -> List[dict]:
    xls = data_root / "DEMO_DATA.xls"
    if not xls.exists():
        print(f"Registry {xls} not found.", file=sys.stderr)
        return []
    entries = read_registry(xls)
    outputs: List[dict] = []
    print(f"Processing {len(entries)} sequence(s) from {xls.name}:\n")
    for entry in entries:
        csv_path = data_root / entry.folder_name / entry.csv_name
        if not csv_path.exists():
            print(f"  {entry.folder_name}: CSV not found ({csv_path.name}), skipping")
            continue
        try:
            seq = load_sequence(data_root, entry)
            out = _process_sequence(seq)
            _print_row(out["result"], out["qc"])
            print()
            outputs.append(out)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  {entry.folder_name}: failed - {exc}\n")
    return outputs


def _run_single(csv: Path, is_4ch: bool, flip: bool, events: Optional[List[int]]) -> List[dict]:
    seq = parse_csv(csv, is_4ch=is_4ch, flip=flip, ecg_events=events)
    out = _process_sequence(seq)
    _print_row(out["result"], out["qc"])
    return [out]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gls_analysis",
        description="Compute GLS/GCS from ECHOPAC speckle-tracking CSV exports.",
    )
    parser.add_argument("--data", type=Path, help="Data root containing DEMO_DATA.xls")
    parser.add_argument("--csv", type=Path, help="Single CSV file to process")
    parser.add_argument("--4ch", dest="is_4ch", action="store_true", help="Sequence is apical (4CH)")
    parser.add_argument("--sax", dest="is_sax", action="store_true", help="Sequence is short-axis")
    parser.add_argument("--flip", action="store_true", help="Wall-tracing flip flag")
    parser.add_argument(
        "--events",
        type=int,
        nargs=6,
        metavar=("Q1", "MVC", "AVO", "AVC", "MVO", "Q2"),
        help="Six 0-based ECG-event frame indices",
    )
    parser.add_argument("--json", type=Path, help="Write results to this JSON file")
    args = parser.parse_args(argv)

    outputs: List[dict] = []
    if args.data:
        outputs = _run_registry(args.data)
    elif args.csv:
        if not (args.is_4ch or args.is_sax):
            parser.error("specify --4ch or --sax when using --csv")
        outputs = _run_single(args.csv, args.is_4ch and not args.is_sax, args.flip, args.events)
    else:
        parser.error("provide either --data <root> or --csv <file>")

    if args.json and outputs:
        payload = [
            {**o["result"].as_dict(), "qc": o["qc"].as_dict()} for o in outputs
        ]
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {len(payload)} result(s) to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
