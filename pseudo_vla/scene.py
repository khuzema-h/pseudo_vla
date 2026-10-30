"""Isaac Sim Franka scene in the simple room, with a wrist camera and an environment camera."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Bounds of table_low in Isaac's Simple Room. The cuRobo cuboid uses the same box.
TABLE_MIN = np.array([-1.592, -0.813, -0.77], dtype=np.float64)
TABLE_MAX = np.array([1.593, 0.813, 0.01], dtype=np.float64)
TABLE_DIMS = TABLE_MAX - TABLE_MIN
TABLE_CENTER = (TABLE_MIN + TABLE_MAX) / 2.0
TABLE_TOP_Z = float(TABLE_MAX[2])
# The arm stands on the near edge of the table and faces +Y, toward the props.
ARM_POSITION = (0.0, -0.60, TABLE_TOP_Z + 0.002)
# Three grasp-sized blocks in front of the arm.
BLOCK_SIZE = 0.05
SCENE_OBJECTS = (
    {"name": "red_block", "color": (0.9, 0.05, 0.05), "xy": (-0.22, -0.28)},
    {"name": "green_block", "color": (0.05, 0.8, 0.05), "xy": (0.0, -0.28)},
    {"name": "blue_block", "color": (0.05, 0.15, 0.95), "xy": (0.22, -0.28)},
)
ENV_CAMERA_PATH = "/World/EnvCamera"
# In front of the arm (it faces +Y) and a little lower, nearer the tabletop.
ENV_CAMERA_EYE = np.array([0.0, 1.15, 0.90], dtype=np.float64)
ENV_CAMERA_TARGET = np.array([0.0, -0.35, 0.30], dtype=np.float64)
ENV_RESOLUTION = (480, 480)
WRIST_RESOLUTION = (256, 256)
# Pose on panda_hand. Offset to the side of the palm and pitched down the fingers
# so the open gap, not the back of the hand, is in frame. Fingertips are near z=0.11.
WRIST_EYE = np.array([0.12, 0.0, 0.03], dtype=np.float64)
WRIST_LOOK_TARGET = np.array([0.0, 0.0, 0.13], dtype=np.float64)
HOME_ARM = {
    "panda_joint1": 0.0,
    "panda_joint2": -1.3,
    "panda_joint3": 0.0,
    "panda_joint4": -2.5,
    "panda_joint5": 0.0,
    "panda_joint6": 1.5,
    "panda_joint7": 0.8,
}
HOME_FINGERS = {
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}
ARM_JOINT_NAMES = list(HOME_ARM)


def launch(headless: bool):
    """Start Kit. Call this before importing omni or isaacsim.core."""
    import sys

    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    # SimulationApp forwards sys.argv into Kit. Keep the script flags out of that list.
    argv = sys.argv[:]
    sys.argv = [argv[0]]
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": headless, "width": 1280, "height": 720})
    sys.argv = argv
    return app


def _ros_optical_to_usd(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Map a ROS optical orientation (x right, y down, z forward) onto a USD camera."""
    # Rx(180) takes ROS +Z onto USD -Z and ROS +Y onto USD -Y.
    aw, ax, ay, az = quat_wxyz
    bw, bx, by, bz = 0.0, 1.0, 0.0, 0.0
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _log(message: str) -> None:
    print(message, flush=True)
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "run.log")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _find_prim_path(stage, name: str) -> str:
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return str(prim.GetPath())
    raise RuntimeError(f"USD prim named {name} was not found")


@dataclass
class FrankaScene:
    app: object
    arm: object
    camera_path: str
    wrist_camera_path: str
    hand_path: str
    base_path: str
    object_paths: dict[str, str]
    rgb_annot: object
    depth_annot: object
    camera_annot: object

    def step(self) -> None:
        self.app.update()

    def read_sensors(self) -> tuple[np.ndarray, np.ndarray, dict]:
        import omni.replicator.core as rep

        rgb = self.rgb_annot.get_data()
        depth = self.depth_annot.get_data()
        params = self.camera_annot.get_data()
        if _empty(rgb) or _empty(depth):
            rep.orchestrator.step()
            rgb = self.rgb_annot.get_data()
            depth = self.depth_annot.get_data()
            params = self.camera_annot.get_data()
        rgb = np.asarray(rgb)
        depth = np.asarray(depth)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        depth = np.squeeze(depth).astype(np.float64)
        if not isinstance(params, dict):
            params = {}
        return np.ascontiguousarray(rgb), depth, params

    def dof_index(self, name: str) -> int | None:
        names = list(self.arm.dof_names)
        if name not in names:
            return None
        return names.index(name)

    def joint_positions(self) -> dict[str, float]:
        values = np.asarray(self.arm.get_dof_positions().numpy(), dtype=np.float64).reshape(-1)
        names = list(self.arm.dof_names)
        return {name: float(values[i]) for i, name in enumerate(names)}

    def set_joint_targets(self, targets: dict[str, float]) -> None:
        names = []
        values = []
        for name, value in targets.items():
            index = self.dof_index(name)
            if index is None:
                continue
            names.append(index)
            values.append(value)
        if not names:
            return
        self.arm.set_dof_position_targets(np.asarray(values, dtype=np.float32).reshape(1, -1), dof_indices=names)

    def set_home(self) -> None:
        targets = {**HOME_ARM, **HOME_FINGERS}
        indices = []
        values = []
        for name, value in targets.items():
            index = self.dof_index(name)
            if index is None:
                continue
            indices.append(index)
            values.append(value)
        positions = np.asarray(values, dtype=np.float32).reshape(1, -1)
        self.arm.set_dof_positions(positions, dof_indices=indices)
        self.arm.set_dof_position_targets(positions, dof_indices=indices)


def _empty(data) -> bool:
    if data is None:
        return True
    array = np.asarray(data)
    return array.size == 0


def _ensure_physics_scene() -> None:
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    path = "/physicsScene"
    if not stage.GetPrimAtPath(path).IsValid():
        scene = UsdPhysics.Scene.Define(stage, path)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(path))


def _add_camera_marker(camera_path: str, color: tuple[float, float, float], size: float) -> None:
    """A body and lens sitting behind the camera, so the camera itself is visible in the stage.

    USD cameras look down their local -Z, so +Z is behind the lens and stays out of that camera's image.
    """
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    body = UsdGeom.Cube.Define(stage, camera_path + "/body")
    body.CreateSizeAttr(1.0)
    body.CreateDisplayColorAttr([color])
    body.CreateDoubleSidedAttr(True)
    body_xform = UsdGeom.Xformable(body.GetPrim())
    body_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, size))
    body_xform.AddScaleOp().Set(Gf.Vec3d(size, size * 0.72, size * 0.55))

    lens = UsdGeom.Cone.Define(stage, camera_path + "/lens")
    lens.CreateRadiusAttr(size * 0.38)
    lens.CreateHeightAttr(size * 0.9)
    lens.CreateAxisAttr("Z")
    lens.CreateDisplayColorAttr([color])
    lens.CreateDoubleSidedAttr(True)
    lens_xform = UsdGeom.Xformable(lens.GetPrim())
    lens_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, size * 0.45))
    lens_xform.AddRotateXOp().Set(180.0)


def _show_camera_gizmos() -> None:
    import carb

    carb.settings.get_settings().set("/app/viewport/show/camera", True)


def _author_camera_pose(camera_path: str, translation: tuple[float, float, float], quat_wxyz: tuple[float, float, float, float]) -> None:
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(camera_path)
    for name in ("xformOp:translate", "xformOp:orient", "xformOp:scale", "xformOpOrder"):
        if prim.GetAttribute(name):
            prim.RemoveProperty(name)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    precision = UsdGeom.XformOp.PrecisionDouble
    translate = xform.AddTranslateOp(precision)
    translate.Set(Gf.Vec3d(*translation))
    orient = xform.AddOrientOp(precision)
    orient.Set(Gf.Quatd(quat_wxyz[0], Gf.Vec3d(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3])))


def _ros_look_at_wxyz(eye: np.ndarray, target: np.ndarray, right_hint: np.ndarray) -> np.ndarray:
    """ROS optical orientation (wxyz). +Z looks at the target and +X follows the right hint."""
    forward = target - eye
    z_axis = forward / np.linalg.norm(forward)
    x_axis = right_hint - z_axis * float(np.dot(right_hint, z_axis))
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return _matrix_to_quat_wxyz(np.stack([x_axis, y_axis, z_axis], axis=1))


def _author_wrist_camera(camera_path: str) -> None:
    # Hand +Y is the finger opening, so the two fingers fall on the left and right of the image.
    quat = _ros_look_at_wxyz(WRIST_EYE, WRIST_LOOK_TARGET, np.array([0.0, 1.0, 0.0]))
    usd_quat = _ros_optical_to_usd(tuple(float(value) for value in quat))
    _author_camera_pose(camera_path, tuple(float(value) for value in WRIST_EYE), usd_quat)


def _matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _look_at_wxyz(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """USD camera orientation (wxyz) that looks from eye toward target, with world +Z up."""
    forward = target - eye
    z_axis = -forward / np.linalg.norm(forward)
    x_axis = np.cross(np.array([0.0, 0.0, 1.0]), z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return _matrix_to_quat_wxyz(np.stack([x_axis, y_axis, z_axis], axis=1))


def object_center(spec: dict) -> np.ndarray:
    return np.asarray(spec["center"], dtype=np.float64)


def object_dims(spec: dict) -> np.ndarray:
    return np.asarray(spec["dims"], dtype=np.float64)


def _record_prop_bounds(spec: dict) -> None:
    """Store the prop's world-aligned center and size for the cuRobo cuboid."""
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    box = cache.ComputeWorldBound(stage.GetPrimAtPath(f"/World/{spec['name']}")).ComputeAlignedBox()
    minimum, maximum = box.GetMin(), box.GetMax()
    spec["center"] = np.array(
        [float(minimum[i] + maximum[i]) / 2.0 for i in range(3)],
        dtype=np.float64,
    )
    spec["dims"] = np.array([float(maximum[i] - minimum[i]) for i in range(3)], dtype=np.float64)
    _log(f"prop {spec['name']} center {spec['center'].round(3).tolist()} dims {spec['dims'].round(3).tolist()}")


def _place_arm(arm_path: str) -> None:
    """Stand the Franka on the table, facing the props."""
    yaw = np.pi / 2.0
    quat = (float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0)))
    _author_camera_pose(arm_path, ARM_POSITION, quat)


def _collide_room_table() -> None:
    """Give the room table a static box collider so props rest on it."""
    import omni.usd
    from isaacsim.core.experimental.prims import GeomPrim
    from pxr import UsdPhysics

    mesh_path = "/World/SimpleRoom/table_low_327/table_low"
    GeomPrim(mesh_path).apply_collision_apis()
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(mesh_path)
    api = UsdPhysics.MeshCollisionAPI(prim)
    api.CreateApproximationAttr("boundingCube")


def _place_block(spec: dict) -> str:
    """Create a colored, dynamic cube on the tabletop."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    path = f"/World/{spec['name']}"
    stage = omni.usd.get_context().get_stage()
    xy = spec["xy"]
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(BLOCK_SIZE)
    cube.CreateDisplayColorAttr([spec["color"]])
    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(
        Gf.Vec3d(float(xy[0]), float(xy[1]), TABLE_TOP_Z + BLOCK_SIZE / 2.0 + 0.002)
    )
    _record_prop_bounds(spec)
    return path


def build_scene(app) -> FrankaScene:
    """Load the simple room, Franka, three colored blocks, and both cameras."""
    import omni.replicator.core as rep
    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.objects import Camera
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.storage.native import get_assets_root_path

    _log("new stage")
    app_utils.enable_extension("omni.replicator.core")
    stage_utils.create_new_stage(template="empty")
    _ensure_physics_scene()
    _log("physics scene ready")

    assets = get_assets_root_path()
    if not assets:
        raise RuntimeError("Isaac Sim assets root was not found")
    room = assets + "/Isaac/Environments/Simple_Room/simple_room.usd"
    stage_utils.add_reference_to_stage(room, "/World/SimpleRoom")
    _log(f"referenced {room}")
    _collide_room_table()

    usd = assets + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    stage_utils.add_reference_to_stage(usd, "/World/Arm")
    _place_arm("/World/Arm")
    _log(f"referenced {usd}")

    object_paths = {spec["name"]: _place_block(spec) for spec in SCENE_OBJECTS}

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    hand_path = _find_prim_path(stage, "panda_hand")
    base_path = _find_prim_path(stage, "panda_link0")
    wrist_camera_path = hand_path + "/wrist_cam"
    wrist = Camera(wrist_camera_path)
    wrist.set_focal_lengths([16.0])
    wrist.set_apertures(horizontal_apertures=[20.955], vertical_apertures=[15.2908])
    wrist.set_clipping_ranges(near_distances=[0.02], far_distances=[2.0])
    _author_wrist_camera(wrist_camera_path)
    _add_camera_marker(wrist_camera_path, (0.95, 0.45, 0.08), 0.028)
    _log(f"wrist camera eye {WRIST_EYE.tolist()} looking at {WRIST_LOOK_TARGET.tolist()}")

    env_camera = Camera(ENV_CAMERA_PATH)
    env_camera.set_focal_lengths([18.0])
    env_camera.set_apertures(horizontal_apertures=[20.955], vertical_apertures=[20.955])
    env_camera.set_clipping_ranges(near_distances=[0.05], far_distances=[15.0])
    env_quat = _look_at_wxyz(ENV_CAMERA_EYE, ENV_CAMERA_TARGET)
    _author_camera_pose(ENV_CAMERA_PATH, tuple(float(value) for value in ENV_CAMERA_EYE), tuple(float(value) for value in env_quat))
    _add_camera_marker(ENV_CAMERA_PATH, (0.15, 0.75, 0.95), 0.06)

    scene_arm = Articulation("/World/Arm")
    app_utils.play()
    for _ in range(15):
        app.update()
    render_product = rep.create.render_product(ENV_CAMERA_PATH, ENV_RESOLUTION)
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    camera_annot = rep.AnnotatorRegistry.get_annotator("camera_params")
    rgb_annot.attach(render_product)
    depth_annot.attach(render_product)
    camera_annot.attach(render_product)

    scene = FrankaScene(
        app=app,
        arm=scene_arm,
        camera_path=ENV_CAMERA_PATH,
        wrist_camera_path=wrist_camera_path,
        hand_path=hand_path,
        base_path=base_path,
        object_paths=object_paths,
        rgb_annot=rgb_annot,
        depth_annot=depth_annot,
        camera_annot=camera_annot,
    )
    scene.set_home()
    _log("home pose set")
    for _ in range(20):
        scene.step()
    for spec in SCENE_OBJECTS:
        _record_prop_bounds(spec)
    _show_camera_gizmos()
    _log("scene ready")
    return scene
