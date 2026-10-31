"""Isaac Sim camera windows docked on the right of the scene view."""

from __future__ import annotations

import numpy as np

from pseudo_vla.perception import compose_overlay

WRIST_WINDOW = "Wrist Camera"
ENV_WINDOW = "Environment Camera"


class PerceptionWindows:
    """Wrist viewport and environment-camera overlay, docked as right-side panels."""

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
            title=WRIST_WINDOW,
            resolution=wrist_resolution,
        )
        self.viewport.width = width
        self.viewport.height = height

        self._provider = ui.ByteImageProvider()
        self.window = ui.Window(ENV_WINDOW, width=overlay_width + 16, height=overlay_height + 16)
        with self.window.frame:
            ui.ImageWithProvider(self._provider, fill_policy=ui.IwpFillPolicy.IWP_STRETCH)
        _dock_camera_panels(self.viewport, self.window)

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


def _dock_camera_panels(wrist_window, environment_window) -> None:
    """Pin both camera windows into a column on the right of the main scene view."""
    import asyncio

    import omni.kit.app
    import omni.ui as ui

    async def pin() -> None:
        app = omni.kit.app.get_app()
        scene_view = None
        for _ in range(8):
            await app.next_update_async()
            scene_view = ui.Workspace.get_window("Viewport")
            if scene_view is not None:
                break
        if scene_view is None:
            return
        wrist_window.dock_in(scene_view, ui.DockPosition.RIGHT, 0.26)
        await app.next_update_async()
        environment_window.dock_in(wrist_window, ui.DockPosition.BOTTOM, 0.5)

    asyncio.ensure_future(pin())
