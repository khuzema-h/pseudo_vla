"""OWL-ViT text grounding for environment-camera images."""

from __future__ import annotations

import numpy as np
import torch

_MIN_SCORE = 0.008
_MAX_AREA_FRACTION = 0.2


def color_box(rgb: np.ndarray, text: str) -> tuple[float, float, float, float] | None:
    """Box the largest saturated region whose color is named in the text."""
    color = _named_color(text)
    if color is None:
        return None
    image = _as_uint8(rgb)
    mask = _color_mask(image, color)
    if int(mask.sum()) < 16:
        return None
    import cv2

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    height, width = mask.shape
    area_limit = _MAX_AREA_FRACTION * width * height
    best_area = 0
    best = None
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 16 or area > area_limit or area <= best_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        box_width = int(stats[label, cv2.CC_STAT_WIDTH])
        box_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        best_area = area
        best = (
            float(x),
            float(y),
            float(x + box_width - 1),
            float(y + box_height - 1),
        )
    return best


class ObjectGrounder:
    def __init__(self) -> None:
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self._processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self._model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        ).cuda().eval()

    def ground(
        self,
        rgb: np.ndarray,
        text: str,
    ) -> tuple[tuple[float, float, float, float], float] | None:
        """Return the highest-scoring plausible image box for a text query."""
        phrase = _normalize_phrase(text)
        return self.ground_all(rgb, [phrase]).get(phrase)

    def ground_all(
        self,
        rgb: np.ndarray,
        phrases: list[str],
    ) -> dict[str, tuple[tuple[float, float, float, float], float]]:
        """Box every phrase, using a named color region before OWL-ViT."""
        image = _as_uint8(rgb)
        located: dict[str, tuple[tuple[float, float, float, float], float]] = {}
        remaining: list[str] = []
        for phrase in dict.fromkeys(_normalize_phrase(text) for text in phrases):
            box = color_box(image, phrase)
            if box is not None:
                located[phrase] = (box, 1.0)
            else:
                remaining.append(phrase)
        if not remaining:
            return located
        located.update(_assign_distinct(self._owl_detections(image, remaining)))
        return located

    def _owl_detections(
        self,
        image: np.ndarray,
        phrases: list[str],
    ) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        self.load()
        from PIL import Image

        inputs = self._processor(
            text=[phrases],
            images=Image.fromarray(image),
            return_tensors="pt",
        )
        inputs = {
            key: value.to("cuda") if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.no_grad():
            outputs = self._model(**inputs)
        height, width = image.shape[:2]
        found = self._processor.post_process_grounded_object_detection(
            outputs,
            threshold=0.0,
            target_sizes=[(height, width)],
            text_labels=[phrases],
        )[0]
        if found["scores"].numel() == 0:
            return []
        labels = found.get("text_labels")
        order = torch.argsort(found["scores"], descending=True)
        detections = []
        for index in order.tolist():
            score = float(found["scores"][index])
            if score < _MIN_SCORE:
                break
            box = _accepted_box(found["boxes"][index].detach().cpu().tolist(), width, height)
            if box is None or not labels:
                continue
            detections.append((str(labels[index]), score, box))
        return detections


def _named_color(text: str) -> str | None:
    words = set(_normalize_phrase(text).split())
    for name in ("red", "green", "blue"):
        if name in words:
            return name
    return None


def _normalize_phrase(text: str) -> str:
    return " ".join(text.replace("_", " ").split())


def _as_uint8(rgb: np.ndarray) -> np.ndarray:
    image = np.ascontiguousarray(rgb[..., :3])
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _color_mask(image: np.ndarray, color: str) -> np.ndarray:
    red = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    blue = image[:, :, 2].astype(np.int16)
    if color == "red":
        return (red > 90) & (red > green + 35) & (red > blue + 35)
    if color == "green":
        return (green > 70) & (green > red + 25) & (green > blue + 25)
    return (blue > 70) & (blue > red + 25) & (blue > green + 10)


def _accepted_box(
    raw_box: list[float],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = (float(value) for value in raw_box)
    x1, x2 = sorted((min(max(x1, 0.0), width - 1.0), min(max(x2, 0.0), width - 1.0)))
    y1, y2 = sorted((min(max(y1, 0.0), height - 1.0), min(max(y2, 0.0), height - 1.0)))
    area = (x2 - x1) * (y2 - y1)
    if area < 16.0 or area > _MAX_AREA_FRACTION * width * height:
        return None
    return (x1, y1, x2, y2)


def _assign_distinct(
    detections: list[tuple[str, float, tuple[float, float, float, float]]],
) -> dict[str, tuple[tuple[float, float, float, float], float]]:
    chosen: dict[str, tuple[tuple[float, float, float, float], float]] = {}
    boxes: list[tuple[float, float, float, float]] = []
    for label, score, box in sorted(detections, key=lambda item: item[1], reverse=True):
        if label in chosen or any(_iou(box, other) > 0.5 for other in boxes):
            continue
        chosen[label] = (box, score)
        boxes.append(box)
    return chosen


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    area = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(area - intersection, 1e-6)
