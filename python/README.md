# Strain Analysis Library (Python)

A pure-Python / NumPy **strain (GLS / GCS)** analysis library. It is
**source-agnostic**: the core works on a vendor-neutral `StrainSequence`
(tracked myocardial points over time), so it drops into any program. Reading
ECHOPAC CSV or running a DICOM segmentation model are **optional adapters**, not
part of the core.

The strain math is a faithful port of this toolkit's MATLAB `getSTdataXY.m`
(`../Code`), with a numerically-stable global metric added on top.

## The neutral core (embed this)

Everything flows through one type. Build it from raw points — no ECHOPAC, no
DICOM, no view jargon required:

```python
from gls_analysis import StrainSequence, compute_gls, assess

seq = StrainSequence.from_points(
    points,               # (num_frames, num_points, 2) — your tracked contour
    frame_rate=50.0,
    topology="open",      # "open" = longitudinal wall, "closed" = circumferential ring
    reference_frame=0,    # end-diastole (strain reference)
    end_systole_frame=12, # end-systole
)
result = compute_gls(seq)          # -> GLSResult
report = assess(result)            # -> QCReport
print(result.metric_name, result.gls_percent)   # "GLS" -18.3
```

`topology="open"` yields **GLS** (longitudinal); `"closed"` yields **GCS**
(circumferential). `result.as_dict()` is JSON-serialisable for reporting. That
is the entire contract your program needs — the sections below are just adapters
that construct a `StrainSequence` for you.

## Why a separate global metric?

`getSTdataXY.m` computes a **pointwise** Lagrangian strain in a local
radial/longitudinal frame (`tmp_f ./ tmp_1`). Near the apex the reference
longitudinal segment component approaches zero, so that ratio can blow up for
individual points — a naive spatial mean of the pointwise strain is therefore
unstable (one sample loop peaks at ‑71% that way).

The headline metric here instead uses the **change in total endocardial wall
length**:

```
strain(f) = (L(f) − L(Q1)) / L(Q1) × 100
```

which is the definition clinical GLS/GCS is built on and is stable across every
sample loop. The pointwise MATLAB port is still available
(`compute_local_strain`) for per-point displacement/velocity/strain curves.

## Install

Only NumPy is required. Reading the legacy `.xls` subject registry additionally
needs `xlrd`.

```bash
pip install -r python/requirements.txt
```

## Adapter 1 — ECHOPAC CSV

An input adapter that parses ECHOPAC "store full trace" exports into a
`StrainSequence`:

```python
from gls_analysis import parse_csv, compute_gls, assess

seq = parse_csv(
    "../Data/VOL_0001/VOL_0001_OFF_4CH.CSV",
    is_4ch=True,
    ecg_events=[13, 14, 17, 28, 31, 44],  # Q1 MVC AVO AVC MVO Q2, 0-based
)
result = compute_gls(seq)                 # same core as the neutral example
print(result.metric_name, f"{result.gls_percent:.1f}%")   # GLS -14.8%
```

Command line (for this repo's bundled data):

```bash
cd python
python -m gls_analysis.cli --data ../Data                 # whole DEMO_DATA.xls registry
python -m gls_analysis.cli --csv ../Data/VOL_0001/VOL_0001_OFF_4CH.CSV \
    --4ch --events 13 14 17 28 31 44                       # one CSV
python -m gls_analysis.cli --data ../Data --json out.json  # machine-readable
```

## ECG events

Six frame indices per sequence, in the order used throughout the toolkit:

| # | Event | Meaning                                   |
|---|-------|-------------------------------------------|
| 1 | Q1    | onset of QRS / cycle start (reference/ED) |
| 2 | MVC   | mitral valve closure                      |
| 3 | AVO   | aortic valve opening                      |
| 4 | AVC   | aortic valve closure (end-systole)        |
| 5 | MVO   | mitral valve opening                      |
| 6 | Q2    | onset of next QRS / cycle end             |

These live in `DEMO_DATA.xls` and are read automatically by the registry path.
Unlike the MATLAB reader (which adds `+1` for 1-based indexing), the raw
spreadsheet values are used directly as 0-based Python frame indices.

Peak-systolic strain (the headline `GLS`/`GCS`) is the most negative value
between **Q1 and MVO**; the end-systolic value is taken at **AVC**.

## Adapter 2 — DICOM images

Pixels in, strain out, via a segmentation model. The strain core is validated,
but the model + mask→contour heuristic that feed it are **not** — treat any
number from this path as a research baseline, not a clinical measurement.

You bring the DICOM reader and the model; this package supplies the boundary
interfaces and the validated strain core in between:

```
your DICOM reader ──▶ CineLoop ──▶ Segmenter ──▶ mask_to_endocardial_contour
                                    (EchoNet)             │
                                                          ▼
                              contours ──▶ [validated strain core] ──▶ GLSResult
```

```python
from gls_analysis import CineLoop, EchoNetSegmenter, analyze_cine

# 1. Adapt YOUR DICOM reader's output into a CineLoop (see cine.py)
cine = CineLoop(frames=frames, frame_rate=fps, view="A4C",
                ed_frame=ed, es_frame=es)

# 2. Plug in a per-frame LV segmenter (EchoNet-Dynamic weights, MIT license)
seg = EchoNetSegmenter(weights_path="deeplabv3_resnet50.pt", device="cuda")

# 3. Segment every frame -> wall contour -> strain (reuses the validated core)
result = analyze_cine(cine, seg)
print(result.gls_percent)   # single-plane A4C longitudinal strain
```

Any model satisfying the `Segmenter` protocol (`frame -> binary LV mask`) plugs
in — EchoNet is just the reference adapter.

**Honest limitations of the image path:**

- **A4C-only ⇒ not true GLS.** EchoNet-Dynamic segments apical-4-chamber only, so
  you get the **4-chamber longitudinal component**, not the 3-view (A4C+A2C+A3C)
  clinical GLS. It is labelled `4CH` in the result for that reason.
- **`mask_to_endocardial_contour` is a heuristic** (PCA long axis → apex/annulus
  landmarks). Documented and testable, but needs tuning/validation on real masks.
- **Single-vendor model.** EchoNet is single-center; expect drift on other
  vendors. Cross-vendor robustness needs multi-vendor training data + validation.
- **No clinical validation.** There is no Bland-Altman vs EchoPAC here; the
  `EchoNetSegmenter` wrapper is written to EchoNet's API but not run in CI.

Optional deps for this path only: `opencv-python` (mask→contour) and
`torch`/`torchvision` (EchoNet). The CSV path needs neither.

## Layout

```
python/
  gls_analysis/
    core.py             # StrainSequence — the vendor-neutral core type (embed this)
    strain.py           # local-frame strain (ports getSTdataXY.m) + arc-length strain
    gls.py              # global + segmental GLS/GCS, phase handling
    quality.py          # evidence-based QC gates and reference ranges
    echopac_reader.py   # ADAPTER: ECHOPAC CSV + DEMO_DATA.xls (ports a1_ReadExportedData.m)
    cine.py             # ADAPTER: CineLoop, the DICOM-reader seam
    segmentation.py     # ADAPTER: Segmenter protocol, EchoNet, mask->contour
    image_gls.py        # ADAPTER: cine -> per-frame contours -> core
    cli.py              # command-line entry point (ECHOPAC data)
  tests/
    test_core.py        # neutral core, no adapters imported
    test_gls.py         # ECHOPAC path, runs against the real Data/ samples
    test_image_gls.py   # image glue, synthetic contours (numpy-only core)
  requirements.txt
```

Dependency layers: `core.py` + `strain.py` + `gls.py` + `quality.py` need only
**NumPy**. Each adapter adds its own optional deps (`xlrd` for the registry,
`opencv`/`torch` for the image path) and nothing in the core imports them.

## Results on the bundled sample data

| Sequence            | View | FR (Hz) | Metric | Peak systolic | End-systolic |
|---------------------|------|---------|--------|---------------|--------------|
| VOL_0001_OFF_4CH    | 4CH  | 52      | GLS    | −14.8%        | −14.0%       |
| VOL_0002_OFF_4CH    | 4CH  | 70      | GLS    | −11.0%        | −10.3%       |
| SUB_0014_POST_SAX   | SAX  | 79      | GCS    | −12.1%        | −11.9%       |
| SUB_0015_POST_SAX   | SAX  | 85      | GCS    | −9.8%         | −9.7%        |

(GCS from a single short-axis endocardial ring is of smaller magnitude than a
full-model GCS; the QC plausibility band accounts for this.)

## Tests

```bash
python python/tests/test_gls.py        # no pytest needed
# or
python -m pytest python/tests
```

The suite asserts physiological behaviour on the real samples: correct sign,
plausible magnitude, numerical stability, a zero-strain reference frame, and
agreement of the MATLAB port at the reference frame.

## References

- Duchateau N, De Craene M, Piella G, et al. *A spatiotemporal statistical atlas
  of motion for the quantification of abnormalities in myocardial tissue
  velocities.* Medical Image Analysis, 2011;15(3):316‑28.
- Voigt J-U, et al. *Definitions for a common standard for 2D speckle tracking
  echocardiography.* JASE 2015 (frame-rate and methodology thresholds).
