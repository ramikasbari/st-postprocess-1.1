"""
Segmentation seam: turn image frames into an ordered endocardial wall contour.

Two layers:

* :class:`Segmenter` — a minimal protocol (``frame -> binary LV mask``). Any
  model that produces a per-frame left-ventricle mask satisfies it.
* :class:`EchoNetSegmenter` — an adapter for the EchoNet-Dynamic DeepLabV3-R50
  segmentation model (https://github.com/echonet/dynamic, MIT). It is imported
  lazily so this package has no hard dependency on ``torch``; you only pay for
  it when you actually instantiate the adapter with weights.

* :func:`mask_to_endocardial_contour` — converts a blood-pool mask into the
  ordered wall contour (mitral annulus -> apex -> annulus, valve plane
  excluded) that the strain core consumes.

IMPORTANT — validation status:
    ``mask_to_endocardial_contour`` uses a geometric heuristic to locate the
    mitral annulus and apex. It is a reasonable, documented starting point but
    is NOT clinically validated; landmark detection is exactly the kind of step
    that needs tuning and checking against a reference on your data. The
    downstream strain math (which it feeds) *is* validated; the model and this
    adapter are the unproven links.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Segmenter(Protocol):
    """Anything that maps a single grayscale frame to a binary LV mask."""

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """Return a ``(H, W)`` binary (0/1) left-ventricle blood-pool mask."""
        ...


def mask_to_endocardial_contour(
    mask: np.ndarray,
    n_points: int = 100,
) -> Optional[np.ndarray]:
    """Extract the ordered endocardial wall contour from an LV blood-pool mask.

    The wall runs from one mitral-annulus hinge, along the free wall to the apex
    and back down the opposite wall to the other hinge. The straight mitral
    valve plane (hinge-to-hinge) is excluded, since it is not myocardium.

    Heuristic:
        1. Take the largest external contour of the mask.
        2. Estimate the LV long axis by PCA of the mask pixels; the two extremes
           along it are the apex and base ends.
        3. The apex end is the narrower one (smaller cross-section); the base end
           is wider (the valve plane). The two annulus hinges are the contour
           points nearest the base end that are extreme across the short axis.
        4. Walk the contour the "long way" (through the apex) between the hinges.
        5. Resample to ``n_points`` by arc length for cross-frame correspondence.

    Args:
        mask: ``(H, W)`` binary mask.
        n_points: Number of points to resample the wall to.

    Returns:
        ``(n_points, 2)`` array of ``(row, col)`` points ordered hinge -> apex ->
        hinge, or ``None`` if no usable contour is found.

    Requires ``opencv-python`` (imported lazily).
    """
    try:
        import cv2  # noqa: WPS433 - optional dependency
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "mask_to_endocardial_contour requires opencv-python (cv2). "
            "Install it, or supply contours directly to the strain pipeline."
        ) from exc

    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() < 10:
        return None

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).squeeze()
    if contour.ndim != 2 or len(contour) < 8:
        return None
    # cv2 returns (x, y); convert to (row, col) = (y, x).
    pts = contour[:, ::-1].astype(float)

    ys, xs = np.nonzero(m)
    coords = np.column_stack([ys, xs]).astype(float)
    centroid = coords.mean(axis=0)

    # Long axis via PCA of mask pixels.
    cov = np.cov((coords - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    long_axis = eigvecs[:, -1]
    long_axis /= np.linalg.norm(long_axis) + 1e-9

    proj = (coords - centroid) @ long_axis
    # Cross-sectional width near each long-axis extreme decides apex vs base.
    lo_band = coords[proj < np.percentile(proj, 15)]
    hi_band = coords[proj > np.percentile(proj, 85)]
    short_axis = eigvecs[:, 0]

    def width(band: np.ndarray) -> float:
        if len(band) == 0:
            return 0.0
        s = (band - centroid) @ short_axis
        return float(s.max() - s.min())

    if width(lo_band) < width(hi_band):
        apex_sign, base_sign = -1.0, +1.0  # apex at the low-projection end
    else:
        apex_sign, base_sign = +1.0, -1.0

    apex_dir = apex_sign * long_axis
    apex_idx = int(np.argmax((pts - centroid) @ apex_dir))

    # Annulus hinges: contour points toward the base, extreme across short axis.
    base_proj = (pts - centroid) @ (base_sign * long_axis)
    base_mask = base_proj > np.percentile(base_proj, 70)
    base_pts_idx = np.nonzero(base_mask)[0]
    if len(base_pts_idx) < 2:
        return None
    short_proj = (pts[base_pts_idx] - centroid) @ short_axis
    hinge1 = int(base_pts_idx[np.argmin(short_proj)])
    hinge2 = int(base_pts_idx[np.argmax(short_proj)])

    # Walk from hinge1 to hinge2 along the arc that passes through the apex.
    n = len(pts)

    def arc(a: int, b: int) -> List[int]:
        return [i % n for i in range(a, a + ((b - a) % n) + 1)]

    arc1 = arc(hinge1, hinge2)
    arc2 = arc(hinge2, hinge1)
    wall_idx = arc1 if (apex_idx in arc1) else arc2
    wall = pts[wall_idx]
    if len(wall) < 4:
        return None

    return _resample_open_contour(wall, n_points)


def _resample_open_contour(points: np.ndarray, n_points: int) -> np.ndarray:
    """Resample an open poly-line to ``n_points`` equal-arc-length samples."""
    pts = np.asarray(points, dtype=float)
    seg = np.sqrt((np.diff(pts, axis=0) ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total == 0:
        return np.repeat(pts[:1], n_points, axis=0)
    u = cum / total
    tu = np.linspace(0.0, 1.0, n_points)
    r = np.interp(tu, u, pts[:, 0])
    c = np.interp(tu, u, pts[:, 1])
    return np.column_stack([r, c])


class EchoNetSegmenter:
    """Adapter for the EchoNet-Dynamic DeepLabV3-ResNet50 LV segmenter.

    EchoNet-Dynamic (Ouyang et al., Nature 2020; https://github.com/echonet/dynamic,
    MIT license) segments the LV blood pool per frame in apical-4-chamber views.
    This adapter wraps its DeepLabV3 model behind the :class:`Segmenter` protocol.

    ``torch``/``torchvision`` are imported lazily; instantiate only when you have
    the pretrained weights available.

    Args:
        weights_path: Path to the EchoNet segmentation checkpoint
            (e.g. ``deeplabv3_resnet50_random.pt``). If ``None``, the model is
            built with random weights (useful only for smoke-testing the plumbing,
            NOT for real strain).
        device: Torch device string (``"cuda"`` or ``"cpu"``).
        input_size: Square size EchoNet expects (default 112).

    Note:
        This wrapper is written against EchoNet's documented API but is not
        exercised in this repository's tests (no torch/weights here). Treat it as
        an integration stub to validate on your machine.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cpu",
        input_size: int = 112,
    ) -> None:
        try:
            import torch  # noqa: WPS433
            import torchvision  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "EchoNetSegmenter requires torch and torchvision. "
                "See https://github.com/echonet/dynamic for setup and weights."
            ) from exc

        self._torch = torch
        self.device = device
        self.input_size = input_size

        model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=None, num_classes=1, aux_loss=False
        )
        if weights_path is not None:
            checkpoint = torch.load(weights_path, map_location=device)
            state = checkpoint.get("state_dict", checkpoint)
            # EchoNet checkpoints are often saved as DataParallel; strip prefix.
            state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
        model.to(device).eval()
        self.model = model

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """Segment one grayscale frame; returns a ``(H, W)`` binary mask."""
        torch = self._torch
        f = np.asarray(frame, dtype=np.float32)
        if f.ndim == 3:
            f = f[..., :3].mean(axis=-1)
        h, w = f.shape
        # EchoNet works on 3-channel square input; replicate grayscale.
        import cv2  # noqa: WPS433

        square = cv2.resize(f, (self.input_size, self.input_size))
        if square.max() > 1.0:
            square = square / 255.0
        tensor = torch.from_numpy(square)[None, None].repeat(1, 3, 1, 1).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor.float())["out"][0, 0]
        prob = torch.sigmoid(logits).cpu().numpy()
        small = (prob > 0.5).astype(np.uint8)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
