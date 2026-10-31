"""OOAL affordance heatmaps for grasp regions.

A user instruction is matched against the seen affordance names in
OOAL/data/agd20k_ego.py. The matched class, such as hold, supplies the
heatmap whose hot pixels seed CoTracker on the object SAM is grasping.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OOAL_ROOT = ROOT / "OOAL"
CROP_SIZE = 224
_CHECKPOINT_CANDIDATES = (
    ROOT / "checkpoints" / "ooal-models" / "seen_best",
    OOAL_ROOT / "models" / "checkpoints" / "seen_best",
)

_model = None
_seen_names: tuple[str, ...] | None = None
_status = "OOAL is not loaded"


def affordance_status() -> str:
    return _status


def seen_affordance_names() -> tuple[str, ...]:
    """Return the seen affordance keywords without loading the network."""
    global _seen_names
    if _seen_names is None:
        _import_ooal()
        from data.agd20k_ego import SEEN_AFF

        _seen_names = tuple(SEEN_AFF)
    return _seen_names


def match_seen_affordance(prompt: str) -> str | None:
    """Return the seen affordance named by the instruction.

    Longer names win when they start at the same word, so "cut with" matches
    cut_with rather than cut. The earliest match in the instruction wins.
    """
    text = re.sub(r"\s+", " ", prompt.lower().replace("-", " ").replace("_", " "))
    best_name = None
    best_at = None
    best_length = -1
    for name in seen_affordance_names():
        phrase = name.replace("_", " ")
        found = re.search(rf"\b{re.escape(phrase)}\b", text)
        if found is None:
            continue
        if best_at is None or found.start() < best_at or (
            found.start() == best_at and len(phrase) > best_length
        ):
            best_name = name
            best_at = found.start()
            best_length = len(phrase)
    return best_name


def peak_pixel(heatmap: np.ndarray, width: int, height: int) -> tuple[float, float]:
    """Map the hottest heatmap cell back onto the original image."""
    if heatmap.ndim != 2 or heatmap.size == 0 or width < 1 or height < 1:
        raise ValueError("A 2D heatmap and a positive image size are required")
    row, column = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    scale_x = width / heatmap.shape[1]
    scale_y = height / heatmap.shape[0]
    return (float(column) + 0.5) * scale_x, (float(row) + 0.5) * scale_y


def affordance_pixel(rgb: np.ndarray, affordance: str) -> tuple[float, float] | None:
    """Return one affordance peak in original image pixels."""
    heatmap = affordance_heatmap(rgb, affordance)
    if heatmap is None:
        return None
    height, width = heatmap.shape
    return peak_pixel(heatmap, width, height)


def affordance_heatmap(rgb: np.ndarray, affordance: str) -> np.ndarray | None:
    """Return the named seen-affordance heatmap at the RGB image size."""
    import cv2

    model = _load()
    names = _seen_names
    if model is None or names is None or affordance not in names or rgb.size == 0:
        return None
    import torch

    image = _as_uint8(rgb)
    height, width = image.shape[:2]
    batch = _imagenet_tensor(image).unsqueeze(0).cuda()
    index = torch.tensor([names.index(affordance)], device=batch.device)
    with torch.no_grad():
        heatmap = model(batch, gt_aff=index)
    peak = heatmap.squeeze().detach().float().cpu().numpy()
    return cv2.resize(peak, (width, height), interpolation=cv2.INTER_LINEAR)


def grasp_region_points(
    rgb: np.ndarray,
    mask: np.ndarray,
    affordance: str,
    count: int = 16,
) -> np.ndarray | None:
    """Sample CoTracker points where the affordance heatmap is hot inside a SAM mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    pad = 8
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(rgb.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(rgb.shape[1], int(xs.max()) + pad + 1)
    heatmap = affordance_heatmap(rgb[y0:y1, x0:x1], affordance)
    if heatmap is None:
        return None
    return region_points_from_heatmap(mask[y0:y1, x0:x1], heatmap, count=count, origin=(x0, y0))


def region_points_from_heatmap(
    mask: np.ndarray,
    heatmap: np.ndarray,
    count: int = 16,
    origin: tuple[int, int] = (0, 0),
) -> np.ndarray | None:
    """Sample points on the hot part of a heatmap restricted to a mask."""
    from pseudo_vla.geometry import sample_mask_points

    if mask.shape != heatmap.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match heatmap {heatmap.shape[:2]}")
    values = heatmap[mask.astype(bool)]
    if values.size == 0:
        return None
    peak = float(np.max(values))
    if not np.isfinite(peak) or peak <= 0:
        return None
    threshold = max(0.8 * peak, float(np.percentile(values, 90)))
    region = mask.astype(bool) & (heatmap >= threshold)
    if int(np.count_nonzero(region)) < 4:
        return None
    points = sample_mask_points(region, count=count)
    points[:, 0] += origin[0]
    points[:, 1] += origin[1]
    return points


def _load():
    global _model, _seen_names, _status
    if _model is not None:
        return _model
    if _status != "OOAL is not loaded":
        return None
    checkpoint = _checkpoint()
    if checkpoint is None:
        _status = "OOAL seen checkpoint was not found"
        return None
    try:
        import torch

        _import_ooal()
        from data.agd20k_ego import SEEN_AFF
        from models.ooal import Net as OoalNet

        _seen_names = tuple(SEEN_AFF)

        class _Args:
            class_names = SEEN_AFF

        with contextlib.redirect_stdout(io.StringIO()):
            network = OoalNet(_Args(), 768, 512, pretrained_backbones=False).cuda().eval()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = saved["model_state_dict"] if isinstance(saved, dict) and "model_state_dict" in saved else saved
        network.load_state_dict(state, strict=False)
        _model = network
        _status = f"loaded {checkpoint.name}"
        return _model
    except Exception as error:
        _status = str(error)
        return None


def _import_ooal() -> None:
    ooal_root = str(OOAL_ROOT)
    if ooal_root not in sys.path:
        sys.path.insert(0, ooal_root)


def _checkpoint() -> Path | None:
    for path in _CHECKPOINT_CANDIDATES:
        if path.is_file():
            return path
    return None


def _as_uint8(rgb: np.ndarray) -> np.ndarray:
    image = np.ascontiguousarray(rgb[..., :3])
    if image.dtype == np.uint8:
        return image
    values = image.astype(np.float32)
    if np.nanmax(values) <= 1.0:
        values = values * 255.0
    return np.clip(values, 0, 255).astype(np.uint8)


def _imagenet_tensor(image: np.ndarray):
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((CROP_SIZE, CROP_SIZE), antialias=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform(Image.fromarray(image))
