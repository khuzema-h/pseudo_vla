"""SAM 2 mask, CoTracker3 tracks, and environment-camera depth unprojection."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SAM_CONFIG = "configs/sam2/sam2_hiera_b+.yaml"
SAM_CHECKPOINT = REPO_ROOT / "checkpoints" / "sam2_hiera_base_plus.pt"
COTRACKER_CHECKPOINT = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "scaled_online.pth"
TRACK_STRIDE = 8
MIN_TRACK_POINTS = 4
DEPTH_RANGE = (0.05, 3.0)
STANDOFF_Z = 0.10
# panda_hand +Z points along the gripper. Rx(180) aims it at world -Z.
TOP_DOWN_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)


@dataclass
class ApproachGoal:
    position: np.ndarray
    quaternion: np.ndarray
    median_world: np.ndarray
    top_z: float


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_inverse(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64) / np.dot(q, q)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def world_to_base(points: np.ndarray, base_position: np.ndarray, base_quat_wxyz: np.ndarray) -> np.ndarray:
    rotation = quat_to_matrix(base_quat_wxyz)
    return (rotation.T @ (points - base_position).T).T


def base_to_world(points: np.ndarray, base_position: np.ndarray, base_quat_wxyz: np.ndarray) -> np.ndarray:
    rotation = quat_to_matrix(base_quat_wxyz)
    return (rotation @ points.T).T + base_position


def prim_world_pose(path: str) -> tuple[np.ndarray, np.ndarray]:
    import omni.timeline
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing prim {path}")
    timeline = omni.timeline.get_timeline_interface()
    time_code = Usd.TimeCode(timeline.get_current_time() * stage.GetTimeCodesPerSecond())
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time_code)
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    position = np.array([translation[0], translation[1], translation[2]], dtype=np.float64)
    quaternion = np.array(
        [rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]],
        dtype=np.float64,
    )
    return position, quaternion


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
        self._window_start = 0
        self.mask: np.ndarray | None = None
        self.query_xy: np.ndarray | None = None
        self._owl = None
        self._owl_processor = None

    def _ensure_grounder(self) -> None:
        if self._owl is not None:
            return
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        if self._owl_processor is None:
            self._owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self._owl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").cuda().eval()

    def unload_grounder(self) -> None:
        """Unload OWL-ViT model from GPU."""
        if self._owl is not None:
            self._owl.to("cpu")
            self._owl = None
        gc.collect()
        torch.cuda.empty_cache()

    def _ensure_sam(self) -> None:
        if self.sam is not None:
            return
        import sam2  # noqa: F401  hydra config module
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._sam_model = build_sam2(SAM_CONFIG, str(SAM_CHECKPOINT), device="cuda")
        self.sam = SAM2ImagePredictor(self._sam_model)

    def unload_sam(self) -> None:
        """Unload SAM 2 model and embeddings from GPU."""
        if self.sam is not None:
            self.sam.reset_predictor()
            if hasattr(self.sam.model, "to"):
                self.sam.model.to("cpu")
            self.sam = None
            self._sam_model = None
        gc.collect()
        torch.cuda.empty_cache()

    def unload_frame0_models(self) -> None:
        """Unload both OWL-ViT and SAM 2 from GPU to leave only CoTracker active."""
        self.unload_grounder()
        self.unload_sam()

    def ground(self, rgb: np.ndarray, text: str) -> tuple[tuple[float, float, float, float], float] | None:
        """Turn a free-text prompt into the best box on this image. Returns None when nothing matches."""
        self._ensure_grounder()
        try:
            from PIL import Image

            image = np.ascontiguousarray(rgb[..., :3])
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            phrase = " ".join(text.replace("_", " ").split())
            inputs = self._owl_processor(text=[[phrase]], images=Image.fromarray(image), return_tensors="pt")
            inputs = {key: value.to("cuda") if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self._owl(**inputs)
            height, width = image.shape[:2]
            found = self._owl_processor.post_process_grounded_object_detection(
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
                x1, y1, x2, y2 = (float(value) for value in found["boxes"][index].detach().cpu().tolist())
                x1, x2 = sorted((min(max(x1, 0.0), width - 1.0), min(max(x2, 0.0), width - 1.0)))
                y1, y2 = sorted((min(max(y1, 0.0), height - 1.0), min(max(y2, 0.0), height - 1.0)))
                area = (x2 - x1) * (y2 - y1)
                if area < 16.0 or area > 0.2 * width * height:
                    continue
                return (x1, y1, x2, y2), score
            return None
        finally:
            self.unload_grounder()

    def segment(self, rgb: np.ndarray, point: tuple[float, float] | None, box: tuple[float, float, float, float] | None) -> np.ndarray:
        image = np.ascontiguousarray(rgb)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        self._ensure_sam()
        try:
            self.sam.set_image(image)
            if point is not None:
                masks, scores, _ = self.sam.predict(
                    point_coords=np.array([[point[0], point[1]]], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32),
                    multimask_output=True,
                )
            else:
                masks, scores, _ = self.sam.predict(
                    box=np.array(box, dtype=np.float32),
                    multimask_output=False,
                )
            mask = masks[int(np.argmax(scores))] > 0
            self.mask = mask
            self.query_xy = sample_mask_points(mask, count=16)
            queries = np.concatenate(
                [np.zeros((self.query_xy.shape[0], 1), dtype=np.float32), self.query_xy.astype(np.float32)],
                axis=1,
            )
            self.queries = torch.from_numpy(queries).unsqueeze(0).cuda()
            frame = _rgb_tensor(image)
            self.tracker(
                video_chunk=frame,
                is_first_step=True,
                queries=self.queries,
                add_support_grid=True,
            )
            self.frames = [image]
            self._window_start = 0
            return self.query_xy
        finally:
            self.unload_frame0_models()

    def push_frame(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
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
    ) -> ApproachGoal | None:
        points = unproject_tracks(tracks_xy, visible, depth, camera_params, camera_path)
        if points.shape[0] < MIN_TRACK_POINTS:
            return None
        median_world = np.median(points, axis=0)
        top_z = float(np.max(points[:, 2]))
        base_position, base_quat = prim_world_pose(base_path)
        goal_world = median_world + np.array([0.0, 0.0, STANDOFF_Z], dtype=np.float64)
        goal_base = world_to_base(goal_world.reshape(1, 3), base_position, base_quat)[0]
        goal_quat = quat_multiply(quat_inverse(base_quat), TOP_DOWN_WXYZ)
        goal_quat = goal_quat / np.linalg.norm(goal_quat)
        return ApproachGoal(position=goal_base, quaternion=goal_quat, median_world=median_world, top_z=top_z)


def sample_mask_points(mask: np.ndarray, count: int = 16) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("SAM 2 returned an empty mask")
    side = int(np.ceil(np.sqrt(count)))
    y_samples = np.linspace(ys.min(), ys.max(), side)
    x_samples = np.linspace(xs.min(), xs.max(), side)
    chosen: list[tuple[float, float]] = []
    for y in y_samples:
        for x in x_samples:
            nearest = int(np.argmin((ys - y) ** 2 + (xs - x) ** 2))
            chosen.append((float(xs[nearest]), float(ys[nearest])))
            if len(chosen) >= count:
                break
        if len(chosen) >= count:
            break
    unique = list(dict.fromkeys(chosen))
    if len(unique) < count:
        order = np.linspace(0, len(xs) - 1, count, dtype=int)
        for index in order:
            unique.append((float(xs[index]), float(ys[index])))
            if len(unique) >= count:
                break
    return np.asarray(unique[:count], dtype=np.float32)


def unproject_tracks(
    tracks_xy: np.ndarray,
    visible: np.ndarray,
    depth: np.ndarray,
    camera_params: dict,
    camera_path: str,
) -> np.ndarray:
    height, width = depth.shape[:2]
    fx, fy, cx, cy = _intrinsics(camera_params, camera_path, width, height)
    camera_to_world = _camera_to_world(camera_params, camera_path)
    points = []
    for (u, v), ok in zip(tracks_xy, visible):
        if not ok:
            continue
        ui = int(np.clip(round(float(u)), 0, width - 1))
        vi = int(np.clip(round(float(v)), 0, height - 1))
        z = float(depth[vi, ui])
        if not np.isfinite(z) or z < DEPTH_RANGE[0] or z > DEPTH_RANGE[1]:
            continue
        # USD cameras look down -Z with +Y up.
        local = np.array([(u - cx) * z / fx, -((v - cy) * z / fy), -z, 1.0], dtype=np.float64)
        world = camera_to_world @ local
        points.append(world[:3])
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(points, axis=0)


def project_world_point(
    point: np.ndarray,
    camera_params: dict,
    camera_path: str,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Project a world point into environment-camera pixels. Returns None if it is behind the camera."""
    fx, fy, cx, cy = _intrinsics(camera_params, camera_path, width, height)
    camera_to_world = _camera_to_world(camera_params, camera_path)
    world = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    local = np.linalg.inv(camera_to_world) @ world
    depth = -float(local[2])
    if depth < DEPTH_RANGE[0]:
        return None
    return np.array(
        [fx * local[0] / depth + cx, cy - fy * local[1] / depth],
        dtype=np.float64,
    )


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


def _intrinsics(camera_params: dict, camera_path: str, width: int, height: int) -> tuple[float, float, float, float]:
    focal, horizontal, vertical = _usd_camera_optics(camera_path)
    if "cameraFocalLength" in camera_params and "cameraAperture" in camera_params:
        focal = float(np.array(camera_params["cameraFocalLength"]).reshape(-1)[0])
        aperture = np.array(camera_params["cameraAperture"], dtype=np.float64).reshape(-1)
        horizontal = float(aperture[0])
        vertical = float(aperture[1]) if aperture.size > 1 else horizontal * height / width
    fx = width * focal / horizontal
    fy = height * focal / vertical
    return fx, fy, (width - 1) / 2.0, (height - 1) / 2.0


def _usd_camera_optics(camera_path: str) -> tuple[float, float, float]:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
    return (
        float(camera.GetFocalLengthAttr().Get()),
        float(camera.GetHorizontalApertureAttr().Get()),
        float(camera.GetVerticalApertureAttr().Get()),
    )


def _camera_to_world(camera_params: dict, camera_path: str) -> np.ndarray:
    usd = _usd_camera_matrix(camera_path)
    view = camera_params.get("cameraViewTransform")
    if view is None:
        return usd
    matrix = np.array(view, dtype=np.float64).reshape(4, 4)
    usd_translation = usd[:3, 3]
    candidates = []
    for candidate in (matrix, matrix.T):
        try:
            inverse = np.linalg.inv(candidate)
        except np.linalg.LinAlgError:
            continue
        candidates.append(inverse)
    if not candidates:
        return usd
    best = min(candidates, key=lambda item: np.linalg.norm(item[:3, 3] - usd_translation))
    if np.linalg.norm(best[:3, 3] - usd_translation) < 0.05:
        return best
    return usd


def _usd_camera_matrix(camera_path: str) -> np.ndarray:
    import omni.timeline
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(camera_path)
    timeline = omni.timeline.get_timeline_interface()
    time_code = Usd.TimeCode(timeline.get_current_time() * stage.GetTimeCodesPerSecond())
    # USD transforms row vectors. Transpose so the matrix multiplies column vectors.
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time_code)
    return np.array(matrix, dtype=np.float64).T
