"""Robot-frame, camera-frame, and tracked-depth geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEPTH_RANGE = (0.05, 3.0)
STANDOFF_Z = 0.10
# panda_hand +Z points along the gripper. Rx(180) aims it at world -Z.
TOP_DOWN_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)


@dataclass
class ApproachGoal:
    position: np.ndarray
    quaternion: np.ndarray
    center_world: np.ndarray
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


def world_to_base(
    points: np.ndarray,
    base_position: np.ndarray,
    base_quat_wxyz: np.ndarray,
) -> np.ndarray:
    rotation = quat_to_matrix(base_quat_wxyz)
    return (rotation.T @ (points - base_position).T).T


def base_to_world(
    points: np.ndarray,
    base_position: np.ndarray,
    base_quat_wxyz: np.ndarray,
) -> np.ndarray:
    rotation = quat_to_matrix(base_quat_wxyz)
    return (rotation @ points.T).T + base_position


def pose_from_depth_points(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Return center, top height, and extent from environment-camera depth points."""
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
        raise ValueError("Depth points must be an Nx3 array with at least 4 points")
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.shape[0] < 4:
        raise ValueError("Depth points contain fewer than 4 finite samples")
    top_z = tracked_surface_top(finite)
    low = np.percentile(finite, 5, axis=0)
    high = np.percentile(finite, 95, axis=0)
    dims = np.maximum(high - low, 0.02).astype(np.float64)
    xy_size = float(np.median(dims[:2]))
    if dims[2] < 0.5 * xy_size:
        dims[2] = xy_size
    center = np.median(finite, axis=0).astype(np.float64)
    center[2] = top_z - float(dims[2]) / 2.0
    return center, top_z, dims


def tracked_surface_top(points: np.ndarray) -> float:
    """Estimate the top of a tracked surface while rejecting isolated depth outliers."""
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("Tracked points must be a non-empty Nx3 array")
    z_values = points[:, 2]
    z_values = z_values[np.isfinite(z_values)]
    if z_values.size == 0:
        raise ValueError("Tracked points contain no finite depths")
    median = float(np.median(z_values))
    mad = float(np.median(np.abs(z_values - median)))
    upper_bound = median + max(3.0 * 1.4826 * mad, 0.015)
    inliers = z_values[z_values <= upper_bound]
    return float(np.max(inliers))


def stack_placement_world(
    support_points: np.ndarray,
    moving_dims: np.ndarray,
    finger_reach: float,
    clearance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return pre-place and release tool points above the tracked support surface."""
    if support_points.ndim != 2 or support_points.shape[1] != 3 or support_points.shape[0] == 0:
        raise ValueError("Support points must be a non-empty Nx3 array")
    if moving_dims.shape != (3,) or np.any(moving_dims <= 0):
        raise ValueError("Moving cube dimensions must be a positive 3-vector")
    center_xy = np.median(support_points[:, :2], axis=0)
    support_top = tracked_surface_top(support_points)
    moving_center = np.array(
        [center_xy[0], center_xy[1], support_top + moving_dims[2] / 2.0],
        dtype=np.float64,
    )
    place_world = moving_center.copy()
    place_world[2] += finger_reach
    preplace_world = place_world.copy()
    preplace_world[2] += clearance
    return preplace_world, place_world, moving_center, support_top


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


def mask_depth_points(
    mask: np.ndarray,
    depth: np.ndarray,
    camera_params: dict,
    camera_path: str,
    count: int = 64,
) -> np.ndarray:
    """Unproject sampled mask pixels with valid depth into world coordinates."""
    if mask.shape != depth.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match depth image {depth.shape[:2]}")
    points_xy = sample_mask_points(mask, count=count)
    visible = np.ones(len(points_xy), dtype=bool)
    return unproject_tracks(points_xy, visible, depth, camera_params, camera_path)


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


def project_tracks_into_view(points_world: np.ndarray, project, width: int, height: int) -> np.ndarray:
    """Project world points into a camera and keep the ones inside the image."""
    pixels = []
    for point in np.asarray(points_world, dtype=np.float64).reshape(-1, 3):
        pixel = project(point)
        if pixel is None:
            continue
        u, v = float(pixel[0]), float(pixel[1])
        if 0.0 <= u < width and 0.0 <= v < height:
            pixels.append((u, v))
    if not pixels:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pixels, dtype=np.float32)


def project_world_point(
    point: np.ndarray,
    camera_params: dict,
    camera_path: str,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Project a world point into camera pixels. Returns None if behind the camera."""
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


def _intrinsics(
    camera_params: dict,
    camera_path: str,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
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
