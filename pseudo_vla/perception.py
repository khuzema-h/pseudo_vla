"""SAM 2 mask, CoTracker3 tracks, and environment-camera depth unprojection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pseudo_vla.grounding import ObjectGrounder
from pseudo_vla.geometry import (
    ApproachGoal,
    DEPTH_RANGE,
    STANDOFF_Z,
    TOP_DOWN_WXYZ,
    base_to_world,
    mask_depth_points,
    prim_world_pose,
    project_world_point,
    quat_inverse,
    quat_multiply,
    sample_mask_points,
    stack_placement_world,
    tracked_surface_top,
    unproject_tracks,
    world_to_base,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SAM_CONFIG = "configs/sam2/sam2_hiera_b+.yaml"
SAM_CHECKPOINT = REPO_ROOT / "checkpoints" / "sam2_hiera_base_plus.pt"
COTRACKER_CHECKPOINT = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "scaled_online.pth"
TRACK_STRIDE = 8
MIN_TRACK_POINTS = 4


class Perception:
    def __init__(self) -> None:
        if not SAM_CHECKPOINT.is_file():
            raise FileNotFoundError(SAM_CHECKPOINT)
        if not COTRACKER_CHECKPOINT.is_file():
            raise FileNotFoundError(COTRACKER_CHECKPOINT)
        from cotracker.predictor import CoTrackerOnlinePredictor

        self.tracker = CoTrackerOnlinePredictor(
            checkpoint=str(COTRACKER_CHECKPOINT),
            offline=False,
        ).cuda()
        self.sam = None
        self._sam_model = None
        self.frames: list[np.ndarray] = []
        self.queries: torch.Tensor | None = None
        self.query_groups: dict[str, slice] = {}
        self._window_start = 0
        self.mask: np.ndarray | None = None
        self.query_xy: np.ndarray | None = None
        self.group_masks: dict[str, np.ndarray] = {}
        self.grounder = ObjectGrounder()
        self.grounder.load()
        self._ensure_sam()

    def _ensure_sam(self) -> None:
        if self.sam is not None:
            return
        import sam2  # noqa: F401  hydra config module
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._sam_model = build_sam2(SAM_CONFIG, str(SAM_CHECKPOINT), device="cuda")
        self.sam = SAM2ImagePredictor(self._sam_model)

    def ground(self, rgb: np.ndarray, text: str) -> tuple[tuple[float, float, float, float], float] | None:
        """Turn a free-text prompt into the best box on this image."""
        return self.grounder.ground(rgb, text)

    def segment(self, rgb: np.ndarray, point: tuple[float, float] | None, box: tuple[float, float, float, float] | None) -> np.ndarray:
        groups = self.segment_groups(rgb, {"target": (point, box)})
        return groups["target"]

    def segment_mask(self, rgb: np.ndarray, point: tuple[float, float]) -> np.ndarray:
        """Segment an object from a positive image point without changing tracker queries."""
        image = np.ascontiguousarray(rgb[..., :3])
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        self._ensure_sam()
        try:
            self.sam.set_image(image)
            masks, scores, _ = self.sam.predict(
                point_coords=np.array([[point[0], point[1]]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True,
            )
            mask = masks[int(np.argmax(scores))] > 0
            if not np.any(mask):
                raise RuntimeError(f"SAM 2 returned an empty mask for point {point}")
            return mask
        finally:
            self.sam.reset_predictor()

    def segment_groups(
        self,
        rgb: np.ndarray,
        prompts: dict[
            str,
            tuple[tuple[float, float] | None, tuple[float, float, float, float] | None],
        ],
        count_per_group: int = 16,
    ) -> dict[str, np.ndarray]:
        """Segment several objects in one frame and seed one tracker with named point groups."""
        if not prompts:
            raise ValueError("At least one named segmentation prompt is required")
        if len(set(prompts)) != len(prompts):
            raise ValueError("Segmentation group names must be unique")
        image = np.ascontiguousarray(rgb)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        self._ensure_sam()
        try:
            self.sam.set_image(image)
            group_points: dict[str, np.ndarray] = {}
            group_masks: dict[str, np.ndarray] = {}
            for name, (point, box) in prompts.items():
                if point is not None:
                    masks, scores, _ = self.sam.predict(
                        point_coords=np.array([[point[0], point[1]]], dtype=np.float32),
                        point_labels=np.array([1], dtype=np.int32),
                        multimask_output=True,
                    )
                elif box is not None:
                    masks, scores, _ = self.sam.predict(
                        box=np.array(box, dtype=np.float32),
                        multimask_output=False,
                    )
                else:
                    raise ValueError(f"Segmentation group {name!r} needs a point or box")
                mask = masks[int(np.argmax(scores))] > 0
                points_xy = sample_mask_points(mask, count=count_per_group)
                group_points[name] = points_xy
                group_masks[name] = mask

            self.group_masks = group_masks
            self._seed_tracker(image, group_points)
            return group_points
        finally:
            self.sam.reset_predictor()

    def retain_groups(self, rgb: np.ndarray, names: tuple[str, ...], count_per_group: int = 16) -> None:
        """Reseed CoTracker on selected cached masks, dropping other groups' query points."""
        if not names:
            raise ValueError("At least one CoTracker group must be retained")
        if len(set(names)) != len(names):
            raise ValueError("CoTracker group names must be unique")
        missing = [name for name in names if name not in self.group_masks]
        if missing:
            raise KeyError(f"No segmentation masks for CoTracker groups {missing}")
        image = np.ascontiguousarray(rgb)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        group_points = {
            name: sample_mask_points(self.group_masks[name], count=count_per_group)
            for name in names
        }
        self._seed_tracker(image, group_points)

    def _seed_tracker(self, image: np.ndarray, group_points: dict[str, np.ndarray]) -> None:
        query_groups: dict[str, slice] = {}
        query_chunks = []
        offset = 0
        for name, points_xy in group_points.items():
            query_groups[name] = slice(offset, offset + len(points_xy))
            offset += len(points_xy)
            query_chunks.append(
                np.concatenate(
                    [
                        np.zeros((len(points_xy), 1), dtype=np.float32),
                        points_xy.astype(np.float32),
                    ],
                    axis=1,
                )
            )
        self.query_groups = query_groups
        self.query_xy = np.concatenate(list(group_points.values()), axis=0)
        self.mask = np.logical_or.reduce([self.group_masks[name] for name in group_points])
        queries = np.concatenate(query_chunks, axis=0)
        self.queries = torch.from_numpy(queries).unsqueeze(0).cuda()
        self.tracker(
            video_chunk=_rgb_tensor(image),
            is_first_step=True,
            queries=self.queries,
            add_support_grid=True,
        )
        self.frames = [image]
        self._window_start = 0

    def track_group(self, tracks_xy: np.ndarray, visible: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Select the tracked coordinates and visibility for one seeded object group."""
        if name not in self.query_groups:
            raise KeyError(f"No CoTracker query group named {name!r}")
        group = self.query_groups[name]
        return tracks_xy[group], visible[group]

    def push_frame(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if self.queries is None:
            return None
        image = np.ascontiguousarray(rgb)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        self.frames.append(image)
        window = self.tracker.step * 2
        end = self._window_start + window
        if end > len(self.frames):
            return None
        chunk = np.stack(self.frames[self._window_start : end], axis=0)
        tracks, visible = self.tracker(video_chunk=_rgb_tensor(chunk), add_support_grid=True)
        self._window_start += self.tracker.step
        self.frames = self.frames[self._window_start :]
        self._window_start = 0
        tracks_np = tracks[0, -1].detach().cpu().numpy()
        visible_np = visible[0, -1].detach().cpu().numpy().astype(bool)
        return tracks_np, visible_np

    def approach_goal(
        self,
        tracks_xy: np.ndarray,
        visible: np.ndarray,
        depth: np.ndarray,
        camera_params: dict,
        camera_path: str,
        base_path: str,
        group: str | None = None,
    ) -> ApproachGoal | None:
        if group is not None:
            tracks_xy, visible = self.track_group(tracks_xy, visible, group)
        points = unproject_tracks(tracks_xy, visible, depth, camera_params, camera_path)
        if points.shape[0] < MIN_TRACK_POINTS:
            return None
        center_world = np.median(points, axis=0)
        top_z = tracked_surface_top(points)
        base_position, base_quat = prim_world_pose(base_path)
        goal_world = np.array([center_world[0], center_world[1], top_z + STANDOFF_Z], dtype=np.float64)
        goal_base = world_to_base(goal_world.reshape(1, 3), base_position, base_quat)[0]
        goal_quat = quat_multiply(quat_inverse(base_quat), TOP_DOWN_WXYZ)
        goal_quat = goal_quat / np.linalg.norm(goal_quat)
        return ApproachGoal(position=goal_base, quaternion=goal_quat, center_world=center_world, top_z=top_z)

    def tracked_group_points(
        self,
        name: str,
        tracks_xy: np.ndarray,
        visible: np.ndarray,
        depth: np.ndarray,
        camera_params: dict,
        camera_path: str,
    ) -> np.ndarray:
        group_xy, group_visible = self.track_group(tracks_xy, visible, name)
        return unproject_tracks(group_xy, group_visible, depth, camera_params, camera_path)


def compose_overlay(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
    points_xy: np.ndarray | None = None,
    visible: np.ndarray | None = None,
) -> np.ndarray:
    """RGB image with the SAM mask tinted green and CoTracker points drawn on top."""
    import cv2

    image = np.ascontiguousarray(rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = image.copy()
    if mask is not None and mask.shape[:2] == image.shape[:2]:
        painted = image.astype(np.float32)
        painted[mask] = 0.55 * painted[mask] + 0.45 * np.array([40, 220, 40], dtype=np.float32)
        image = painted.astype(np.uint8)
    if points_xy is not None:
        flags = np.ones(len(points_xy), dtype=bool) if visible is None else np.asarray(visible, dtype=bool)
        for (x, y), ok in zip(points_xy, flags):
            color = (220, 40, 40) if ok else (170, 170, 170)
            cv2.circle(image, (int(round(float(x))), int(round(float(y)))), 4, color, -1)
    return image


def save_overlay(path: Path, rgb: np.ndarray, mask: np.ndarray, points_xy: np.ndarray) -> None:
    import cv2

    tint = compose_overlay(rgb, mask, points_xy)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(tint, cv2.COLOR_RGB2BGR))


def prompt_on_frame(rgb: np.ndarray) -> tuple[tuple[float, float] | None, tuple[float, float, float, float] | None]:
    import cv2

    state: dict[str, tuple[int, int]] = {}
    view = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["down"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["up"] = (x, y)

    window = "environment camera"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while "up" not in state:
        cv2.imshow(window, view)
        if cv2.waitKey(20) == 27:
            cv2.destroyWindow(window)
            raise RuntimeError("Frame-0 prompt was cancelled")
    cv2.destroyWindow(window)
    x0, y0 = state["down"]
    x1, y1 = state["up"]
    if abs(x1 - x0) + abs(y1 - y0) < 6:
        return (float(x1), float(y1)), None
    return None, (float(min(x0, x1)), float(min(y0, y1)), float(max(x0, x1)), float(max(y0, y1)))


def _rgb_tensor(rgb: np.ndarray) -> torch.Tensor:
    array = np.asarray(rgb)
    if array.ndim == 3:
        array = array[None, ...]
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(0, 3, 1, 2).float().cuda()
    return tensor.unsqueeze(0)
