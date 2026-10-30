"""Collision-free Franka trajectories from a moving tool pose."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import GoalToolPose, JointState, Pose

from pseudo_vla.perception import ApproachGoal, quat_inverse, quat_to_matrix, world_to_base
from pseudo_vla.scene import ARM_JOINT_NAMES, SCENE_OBJECTS, TABLE_CENTER, TABLE_DIMS, object_center, object_dims

REPLAN_DISTANCE = 0.02
GOAL_TOLERANCE = 0.005
KINEMATICS_TOLERANCE = 0.02
# Extra box around each prop so the arm's collision spheres clear the mesh.
OBSTACLE_PADDING = 0.02


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
        self.planner.warmup()
        self._ignored: set[str] = set()
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
            median_world=np.zeros(3, dtype=np.float64),
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

    def update_goal(self, goal: ApproachGoal, joint_map: dict[str, float], ignore_name: str | None = None) -> None:
        obstacles_changed = self._ignore(ignore_name, joint_map)
        moved = self.goal is None or float(np.linalg.norm(goal.position - self.goal.position)) > REPLAN_DISTANCE
        inactive = self.trajectory is None or self.trajectory.finished
        off_goal = False
        if inactive and self.goal is not None:
            off_goal = float(np.linalg.norm(self.hand_position(joint_map) - self.goal.position)) > GOAL_TOLERANCE
        if not obstacles_changed and not moved and not inactive and not off_goal:
            return
        if not obstacles_changed and not moved and inactive and not off_goal:
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
            print("cuRobo plan failed; holding the current joints")
            self.trajectory = None
            return
        names = list(result.js_solution.joint_names or self.planner.joint_names)
        self.trajectory = ActiveTrajectory(positions=trajectory, joint_names=names, index=0)
        print(f"planned {len(trajectory)} waypoints")

    def _ignore(self, destination: str | None, joint_map: dict[str, float]) -> bool:
        """Keep the destination and any prop the arm is already inside out of the collision world."""
        ignored = self._penetrating(joint_map)
        if destination is not None:
            ignored.add(destination)
        if ignored == self._ignored:
            return False
        checker = self.planner.scene_collision_checker
        for name in self._ignored - ignored:
            checker.enable_obstacle(name, True)
        for name in ignored - self._ignored:
            checker.enable_obstacle(name, False)
        self._ignored = ignored
        print(f"collision ignores {sorted(ignored) if ignored else 'nothing'}")
        return True

    def _penetrating(self, joint_map: dict[str, float]) -> set[str]:
        state = self._joint_state(joint_map)
        spheres = self.planner.compute_kinematics(state).robot_spheres
        centers = spheres.detach().reshape(-1, spheres.shape[-1])[:, :3].cpu().numpy()
        radii = spheres.detach().reshape(-1, spheres.shape[-1])[:, 3].cpu().numpy()
        hit: set[str] = set()
        for box in self._boxes:
            if box.name == "table":
                continue
            local = (box.rotation.T @ (centers - box.center).T).T
            delta = np.abs(local) - box.half
            outside = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
            inside = np.max(delta, axis=1)
            distance = np.where(np.all(delta < 0.0, axis=1), inside, outside)
            if np.any(distance < radii - 0.001):
                hit.add(box.name)
        return hit

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
    center: np.ndarray
    half: np.ndarray
    rotation: np.ndarray
    cuboid: Cuboid


def _obstacle_boxes(base_position: np.ndarray, base_quat: np.ndarray) -> list[_ObstacleBox]:
    """World-aligned boxes expressed in the robot base frame."""
    specs = [("table", TABLE_CENTER, TABLE_DIMS, 0.0)]
    specs.extend(
        (spec["name"], object_center(spec), object_dims(spec), OBSTACLE_PADDING) for spec in SCENE_OBJECTS
    )
    rotation = quat_to_matrix(quat_inverse(base_quat))
    boxes = []
    for name, center_world, dims_world, padding in specs:
        center = world_to_base(np.asarray(center_world, dtype=np.float64).reshape(1, 3), base_position, base_quat)[0]
        dims = np.asarray(dims_world, dtype=np.float64) + 2.0 * padding
        quat = quat_inverse(base_quat)
        pose = [float(center[i]) for i in range(3)] + [float(quat[i]) for i in range(4)]
        boxes.append(
            _ObstacleBox(
                name=name,
                center=center,
                half=dims / 2.0,
                rotation=rotation,
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
