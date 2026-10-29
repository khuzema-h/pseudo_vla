"""Isaac Sim windows for the environment camera and the perception overlay."""

from __future__ import annotations

import numpy as np

from pseudo_vla.perception import compose_overlay


class PerceptionWindows:
    """A viewport locked to the environment camera, plus the SAM and CoTracker image."""

    def __init__(self, camera_path: str, resolution: tuple[int, int]) -> None:
        import omni.ui as ui
        from isaacsim.core.rendering_manager import ViewportManager

        width, height = resolution
        self.viewport = ViewportManager.create_viewport_window(
            camera=camera_path,
            title="Environment Camera",
            resolution=resolution,
        )
        self.viewport.position_x = 40
        self.viewport.position_y = 80
        self.viewport.width = width
        self.viewport.height = height

        self._provider = ui.ByteImageProvider()
        self.window = ui.Window("SAM + CoTracker", width=width + 16, height=height + 16)
        self.window.position_x = 64 + width
        self.window.position_y = 80
        with self.window.frame:
            ui.ImageWithProvider(self._provider, fill_policy=ui.IwpFillPolicy.IWP_STRETCH)

    def update(
        self,
        rgb: np.ndarray,
        mask: np.ndarray | None = None,
        points_xy: np.ndarray | None = None,
        visible: np.ndarray | None = None,
    ) -> None:
        if rgb is None or np.asarray(rgb).size == 0:
            return
        image = compose_overlay(rgb, mask, points_xy, visible)
        height, width = image.shape[:2]
        opaque = np.full((height, width, 1), 255, dtype=np.uint8)
        rgba = np.concatenate([image, opaque], axis=2)
        self._provider.set_data_array(np.ascontiguousarray(rgba), [width, height])
