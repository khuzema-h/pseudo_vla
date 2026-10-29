"""Collision-free Franka trajectories from a moving tool pose."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import GoalToolPose, JointState, Pose

from pseudo_vla.perception import ApproachGoal
from pseudo_vla.scene import ARM_JOINT_NAMES, TABLE_CENTER, TABLE_DIMS

REPLAN_DISTANCE = 0.02
GOAL_TOLERANCE = 0.005
KINEMATICS_TOLERANCE = 0.02


@dataclass
class ActiveTrajectory:
    positions: np.ndarray
    joint_names: list[str]
    index: int = 0

    @property
    def finished(self) -> bool:
        return self.index >= len(self.positions)


class ArmServo:
    def __init__(self) -> None:
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            self_collision_check=True,
            collision_cache={"cuboid": 8},
            use_cuda_graph=True,
        )
        self.planner = MotionPlanner(config)
        pose = [
            float(TABLE_CENTER[0]),
            float(TABLE_CENTER[1]),
            float(TABLE_CENTER[2]),
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        self.planner.update_world(
            Scene(
                cuboid=[
                    Cuboid(
                        name="table",
                        dims=[float(value) for value in TABLE_DIMS],
                        pose=pose,
                    )
                ]
            )
        )
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
            median_world=np.zeros(3, dtype=np.float64),
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

    def update_goal(self, goal: ApproachGoal, joint_map: dict[str, float]) -> None:
        moved = self.goal is None or float(np.linalg.norm(goal.position - self.goal.position)) > REPLAN_DISTANCE
        inactive = self.trajectory is None or self.trajectory.finished
        off_goal = False
        if inactive and self.goal is not None:
            off_goal = float(np.linalg.norm(self.hand_position(joint_map) - self.goal.position)) > GOAL_TOLERANCE
        if not moved and not inactive and not off_goal:
            return
        if not moved and inactive and not off_goal:
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

    def _joint_state(self, joint_map: dict[str, float]) -> JointState:
        names = list(self.planner.joint_names)
        missing = [name for name in names if name not in joint_map]
        if missing:
            raise RuntimeError(f"Articulation is missing joints {missing}")
        values = torch.tensor([[joint_map[name] for name in names]], device="cuda", dtype=torch.float32)
        return JointState.from_position(values, joint_names=names)


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
