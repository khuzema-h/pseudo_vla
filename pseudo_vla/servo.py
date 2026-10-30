"""Collision-free Franka trajectories from a moving tool pose."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import GoalToolPose, JointState, Pose

from pseudo_vla.perception import ApproachGoal, quat_inverse, world_to_base
from pseudo_vla.scene import ARM_JOINT_NAMES, SCENE_OBJECTS, TABLE_CENTER, TABLE_DIMS, object_center, object_dims

REPLAN_DISTANCE = 0.02
GOAL_TOLERANCE = 0.005
KINEMATICS_TOLERANCE = 0.02
# Match block collision boxes to their actual dimensions so the 10 cm hover pose clears them.
OBSTACLE_PADDING = 0.0


@dataclass
class ActiveTrajectory:
    positions: np.ndarray
    joint_names: list[str]
    index: int = 0

    @property
    def finished(self) -> bool:
        return self.index >= len(self.positions)


class ArmServo:
    def __init__(self, base_position: np.ndarray, base_quat: np.ndarray) -> None:
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            self_collision_check=True,
            collision_cache={"cuboid": 8},
            optimizer_collision_activation_distance=0.03,
            use_cuda_graph=True,
        )
        self.planner = MotionPlanner(config)
        self._base_position = np.asarray(base_position, dtype=np.float64)
        self._base_quat = np.asarray(base_quat, dtype=np.float64)
        self._boxes = _obstacle_boxes(self._base_position, self._base_quat)
        self.planner.update_world(Scene(cuboid=[box.cuboid for box in self._boxes]))
        self._enabled_obstacles = {box.name for box in self._boxes}
        for box in self._boxes:
            if box.name != "table":
                self.planner.scene_collision_checker.enable_obstacle(box.name, True)
        self.planner.warmup()
        self.trajectory: ActiveTrajectory | None = None
        self.goal: ApproachGoal | None = None
        self.home_goal: ApproachGoal | None = None

    def check_kinematics(self, joint_map: dict[str, float], usd_hand_base: np.ndarray) -> None:
        state = self._joint_state(joint_map)
        pose = self.planner.compute_kinematics(state).tool_poses.get_link_pose("panda_hand")
        fk = pose.position[0].detach().cpu().numpy()
        error = float(np.linalg.norm(fk - usd_hand_base))
        print(f"panda_hand FK {fk.tolist()} USD {usd_hand_base.tolist()} error {error:.4f} m")
        if error > KINEMATICS_TOLERANCE:
            raise RuntimeError(
                f"panda_hand frames differ by {error:.3f} m. The robot base transform does not match cuRobo."
            )

    def remember_home(self, joint_map: dict[str, float]) -> None:
        state = self._joint_state(joint_map)
        pose = self.planner.compute_kinematics(state).tool_poses.get_link_pose("panda_hand")
        self.home_goal = ApproachGoal(
            position=pose.position[0].detach().cpu().numpy().copy(),
            quaternion=pose.quaternion[0].detach().cpu().numpy().copy(),
            center_world=np.zeros(3, dtype=np.float64),
            top_z=0.0,
        )

    def clear(self) -> None:
        self.trajectory = None
        self.goal = None

    def plan_home(self, joint_map: dict[str, float]) -> None:
        if self.home_goal is None:
            raise RuntimeError("Home pose was not recorded")
        self.clear()
        self.update_goal(self.home_goal, joint_map)

    def hand_position(self, joint_map: dict[str, float]) -> np.ndarray:
        state = self._joint_state(joint_map)
        pose = self.planner.compute_kinematics(state).tool_poses.get_link_pose("panda_hand")
        return pose.position[0].detach().cpu().numpy()

    def set_obstacle_enabled(self, name: str, enabled: bool) -> None:
        if name not in {box.name for box in self._boxes}:
            raise ValueError(f"Unknown collision obstacle {name!r}")
        if (name in self._enabled_obstacles) == enabled:
            return
        self.planner.scene_collision_checker.enable_obstacle(name, enabled)
        if enabled:
            self._enabled_obstacles.add(name)
        else:
            self._enabled_obstacles.remove(name)

    def update_block_positions(self, centers: dict[str, np.ndarray]) -> None:
        enabled = self._enabled_obstacles.copy()
        self._boxes = _obstacle_boxes(self._base_position, self._base_quat, centers)
        self.planner.update_world(Scene(cuboid=[box.cuboid for box in self._boxes]))
        self._enabled_obstacles = set()
        for box in self._boxes:
            if box.name != "table":
                is_enabled = box.name in enabled
                self.planner.scene_collision_checker.enable_obstacle(box.name, is_enabled)
                if is_enabled:
                    self._enabled_obstacles.add(box.name)

    def update_goal(self, goal: ApproachGoal, joint_map: dict[str, float]) -> None:
        moved = self.goal is None or float(np.linalg.norm(goal.position - self.goal.position)) > REPLAN_DISTANCE
        inactive = self.trajectory is None or self.trajectory.finished
        off_goal = False
        if inactive and self.goal is not None:
            off_goal = float(np.linalg.norm(self.hand_position(joint_map) - self.goal.position)) > GOAL_TOLERANCE
        if not moved and (not inactive or not off_goal):
            return
        self.goal = goal
        self._plan(joint_map)

    def command(self) -> dict[str, float] | None:
        if self.trajectory is None or self.trajectory.finished:
            return None
        waypoint = self.trajectory.positions[self.trajectory.index]
        self.trajectory.index += 1
        return {
            name: float(value)
            for name, value in zip(self.trajectory.joint_names, waypoint)
            if name in ARM_JOINT_NAMES
        }

    def _plan(self, joint_map: dict[str, float]) -> None:
        assert self.goal is not None
        state = self._joint_state(joint_map)
        position = torch.tensor(self.goal.position, device="cuda", dtype=torch.float32).view(1, 3)
        quaternion = torch.tensor(self.goal.quaternion, device="cuda", dtype=torch.float32).view(1, 4)
        tool_pose = Pose(position=position, quaternion=quaternion)
        goal = GoalToolPose.from_poses(
            {"panda_hand": tool_pose},
            ordered_tool_frames=list(self.planner.tool_frames),
        )
        result = self.planner.plan_pose(goal, state, max_attempts=3)
        trajectory = _trajectory_positions(result)
        if trajectory is None:
            start = self.planner.compute_kinematics(state).tool_poses.get_link_pose("panda_hand")
            start_position = start.position[0].detach().cpu().numpy()
            print(
                "cuRobo plan failed; holding the current joints "
                f"(start position {start_position.round(3).tolist()}, "
                f"goal position {self.goal.position.round(3).tolist()}, "
                f"quaternion {self.goal.quaternion.round(3).tolist()})"
            )
            self.trajectory = None
            return
        names = list(result.js_solution.joint_names or self.planner.joint_names)
        self.trajectory = ActiveTrajectory(positions=trajectory, joint_names=names, index=0)
        print(f"planned {len(trajectory)} waypoints")

    def _joint_state(self, joint_map: dict[str, float]) -> JointState:
        names = list(self.planner.joint_names)
        missing = [name for name in names if name not in joint_map]
        if missing:
            raise RuntimeError(f"Articulation is missing joints {missing}")
        values = torch.tensor([[joint_map[name] for name in names]], device="cuda", dtype=torch.float32)
        return JointState.from_position(values, joint_names=names)


@dataclass
class _ObstacleBox:
    name: str
    cuboid: Cuboid


def _obstacle_boxes(
    base_position: np.ndarray,
    base_quat: np.ndarray,
    centers: dict[str, np.ndarray] | None = None,
) -> list[_ObstacleBox]:
    """World-aligned boxes expressed in the robot base frame."""
    specs = [("table", TABLE_CENTER, TABLE_DIMS, 0.0)]
    for spec in SCENE_OBJECTS:
        center = (centers or {}).get(spec["name"], object_center(spec))
        specs.append((spec["name"], center, object_dims(spec), OBSTACLE_PADDING))
    boxes = []
    for name, center_world, dims_world, padding in specs:
        center = world_to_base(np.asarray(center_world, dtype=np.float64).reshape(1, 3), base_position, base_quat)[0]
        dims = np.asarray(dims_world, dtype=np.float64) + 2.0 * padding
        quat = quat_inverse(base_quat)
        pose = [float(center[i]) for i in range(3)] + [float(quat[i]) for i in range(4)]
        boxes.append(
            _ObstacleBox(
                name=name,
                cuboid=Cuboid(name=name, dims=[float(value) for value in dims], pose=pose),
            )
        )
        print(f"obstacle {name} center {center.round(3).tolist()} dims {dims.round(3).tolist()}")
    return boxes


def _trajectory_positions(result) -> np.ndarray | None:
    if result is None or result.js_solution is None or result.success is None:
        return None
    success = result.success.detach().reshape(-1)
    hits = torch.nonzero(success, as_tuple=False)
    if hits.numel() == 0:
        return None
    slot = int(hits[0])
    position = result.js_solution.position.detach()
    if position.ndim == 4:
        position = position.reshape(-1, position.shape[-2], position.shape[-1])[slot]
    elif position.ndim == 3:
        position = position[slot]
    elif position.ndim != 2:
        raise RuntimeError(f"Unexpected trajectory shape {tuple(position.shape)}")
    return position.cpu().numpy()
