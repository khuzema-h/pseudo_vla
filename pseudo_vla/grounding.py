"""OWL-ViT text grounding for environment-camera images."""

from __future__ import annotations

import numpy as np
import torch


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
        self.load()
        from PIL import Image

        image = np.ascontiguousarray(rgb[..., :3])
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        phrase = " ".join(text.replace("_", " ").split())
        inputs = self._processor(
            text=[[phrase]],
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
            text_labels=[[phrase]],
        )[0]
        if found["scores"].numel() == 0:
            return None
        order = torch.argsort(found["scores"], descending=True)
        for index in order.tolist():
            score = float(found["scores"][index])
            if score < 0.008:
                break
            x1, y1, x2, y2 = (
                float(value)
                for value in found["boxes"][index].detach().cpu().tolist()
            )
            x1, x2 = sorted(
                (min(max(x1, 0.0), width - 1.0), min(max(x2, 0.0), width - 1.0))
            )
            y1, y2 = sorted(
                (min(max(y1, 0.0), height - 1.0), min(max(y2, 0.0), height - 1.0))
            )
            area = (x2 - x1) * (y2 - y1)
            if area < 16.0 or area > 0.2 * width * height:
                continue
            return (x1, y1, x2, y2), score
        return None
