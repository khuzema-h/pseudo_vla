"""Isaac Sim camera and perception windows."""

from __future__ import annotations

import numpy as np

from pseudo_vla.perception import compose_overlay


class PerceptionWindows:
    """Wrist viewport, environment perception overlay, and both camera depth feeds."""

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

        self._depth_providers = {
            "environment": ui.ByteImageProvider(),
            "wrist": ui.ByteImageProvider(),
        }
        self.depth_window = ui.Window("Environment + Wrist Depth", width=440, height=280)
        self.depth_window.position_x = 832
        self.depth_window.position_y = 80
        with self.depth_window.frame:
            with ui.VStack():
                with ui.HStack():
                    ui.Label("Environment")
                    ui.Label("Wrist")
                with ui.HStack():
                    ui.ImageWithProvider(
                        self._depth_providers["environment"],
                        fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                    )
                    ui.ImageWithProvider(
                        self._depth_providers["wrist"],
                        fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                    )

    def update(
        self,
        rgb: np.ndarray,
        mask: np.ndarray | None = None,
        points_xy: np.ndarray | None = None,
        visible: np.ndarray | None = None,
        environment_depth: np.ndarray | None = None,
        wrist_depth: np.ndarray | None = None,
    ) -> None:
        if rgb is not None and np.asarray(rgb).size > 0:
            image = compose_overlay(rgb, mask, points_xy, visible)
            height, width = image.shape[:2]
            opaque = np.full((height, width, 1), 255, dtype=np.uint8)
            rgba = np.concatenate([image, opaque], axis=2)
            self._provider.set_data_array(np.ascontiguousarray(rgba), [width, height])
        self._update_depth("environment", environment_depth)
        self._update_depth("wrist", wrist_depth)

    def _update_depth(self, name: str, depth: np.ndarray | None) -> None:
        if depth is None or np.asarray(depth).size == 0:
            return
        image = _depth_image(depth)
        height, width = image.shape[:2]
        opaque = np.full((height, width, 1), 255, dtype=np.uint8)
        rgba = np.concatenate([image, opaque], axis=2)
        self._depth_providers[name].set_data_array(np.ascontiguousarray(rgba), [width, height])


def _depth_image(depth: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Depth feed must be a 2D array, got shape {values.shape}")
    valid = np.isfinite(values) & (values > 0.0)
    image = np.zeros((*values.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return image
    near, far = np.percentile(values[valid], [5.0, 95.0])
    if far <= near:
        far = near + 1e-6
    normalized = np.clip((values[valid] - near) / (far - near), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    image[valid] = np.round(np.stack((red, green, blue), axis=1) * 255.0).astype(np.uint8)
    return image
