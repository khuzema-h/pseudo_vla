"""Isaac Sim windows for the wrist camera and the perception overlay."""

from __future__ import annotations

import numpy as np

from pseudo_vla.perception import compose_overlay


class PerceptionWindows:
    """A viewport locked to the wrist camera, plus the SAM and CoTracker image."""

    def __init__(
        self,
        wrist_camera_path: str,
        wrist_resolution: tuple[int, int],
        overlay_resolution: tuple[int, int],
    ) -> None:
        import omni.ui as ui
        from isaacsim.core.rendering_manager import ViewportManager

        width, height = wrist_resolution
        overlay_width, overlay_height = overlay_resolution
        self.viewport = ViewportManager.create_viewport_window(
            camera=wrist_camera_path,
            title="Wrist Camera",
            resolution=wrist_resolution,
        )
        self.viewport.position_x = 40
        self.viewport.position_y = 80
        self.viewport.width = width
        self.viewport.height = height

        self._provider = ui.ByteImageProvider()
        self.window = ui.Window("SAM + CoTracker", width=overlay_width + 16, height=overlay_height + 16)
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
