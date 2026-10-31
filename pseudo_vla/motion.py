"""Reusable robot-motion loops and sensor/tracker updates."""

from __future__ import annotations

import queue

import numpy as np

APPROACH_TOLERANCE = 0.02
HOME_DISTANCE = 0.02
GRIPPER_SETTLE_STEPS = 45


def poll_command(commands: queue.Queue[str]) -> str | None:
    try:
        return commands.get_nowait()
    except queue.Empty:
        return None


def execute_pose(
    scene,
    servo,
    goal,
    perception,
    commands,
    max_steps: int,
    log,
    view,
) -> tuple[bool, str | None]:
    servo.clear()
    servo.update_goal(goal, scene.joint_positions())
    if servo.trajectory is None:
        return False, None

    for step in range(max_steps):
        incoming = poll_command(commands)
        if incoming:
            return False, incoming
        scene.step()
        servo.update_goal(goal, scene.joint_positions())
        command = servo.command()
        if command:
            scene.set_joint_targets(command)
        rgb, depth, _ = scene.read_sensors()
        update_tracker(scene, perception, rgb, depth, view)
        if float(np.linalg.norm(servo.hand_position(scene.joint_positions()) - goal.position)) <= APPROACH_TOLERANCE:
            return True, None
        if servo.trajectory is None:
            log(f"pose motion lost its trajectory at step {step}")
            return False, None
        if step % 60 == 0:
            distance = float(np.linalg.norm(servo.hand_position(scene.joint_positions()) - goal.position))
            log(f"pose step {step}: hand is {distance:.3f} m from target")
    return False, None


def close_gripper_in_place(scene, perception, commands, view, max_steps: int = 240) -> str | None:
    """Close the fingers while the arm holds its current joints.

    Returns a queued command if one arrives. The arm stays put until the
    fingers stop moving, including when an object blocks them short of fully shut.
    """
    from pseudo_vla.scene import ARM_JOINT_NAMES, CLOSED_FINGERS

    positions = scene.joint_positions()
    hold = {name: positions[name] for name in ARM_JOINT_NAMES if name in positions}
    scene.set_joint_targets({**hold, **CLOSED_FINGERS})
    scene.gripper_closed = True
    previous = None
    still = 0
    for _ in range(max_steps):
        incoming = poll_command(commands)
        if incoming:
            return incoming
        scene.step()
        rgb, depth, _ = scene.read_sensors()
        update_tracker(scene, perception, rgb, depth, view)
        current = scene.joint_positions()
        scene.set_joint_targets({**hold, **CLOSED_FINGERS})
        if previous is not None:
            moved = max(abs(current.get(name, 0.0) - previous[name]) for name in CLOSED_FINGERS)
            if moved < 0.001:
                still += 1
                if still >= 12:
                    return None
            else:
                still = 0
        previous = {name: current[name] for name in CLOSED_FINGERS}
    return None


def settle_tracker(
    scene,
    perception,
    steps: int,
    view,
    retain_groups: tuple[str, ...] | None = None,
) -> None:
    if retain_groups:
        scene.step()
        rgb, depth, _ = scene.read_sensors()
        perception.retain_groups(rgb, retain_groups)
        update_camera_views(scene, view, rgb, depth, perception.mask, perception.query_xy)
        steps -= 1
    for _ in range(max(steps, 0)):
        scene.step()
        rgb, depth, _ = scene.read_sensors()
        update_tracker(scene, perception, rgb, depth, view)


def home(scene, perception, servo, commands, max_steps: int, say, log, view) -> str | None:
    from pseudo_vla.scene import HOME_ARM

    say("homing")
    servo.plan_home(scene.joint_positions())
    goal = servo.home_goal
    for step in range(max_steps):
        incoming = poll_command(commands)
        if incoming:
            return incoming
        scene.step()
        rgb, depth, _ = scene.read_sensors()
        update_tracker(scene, perception, rgb, depth, view)
        command = servo.command()
        if command:
            scene.set_joint_targets(command)
        elif servo.trajectory is None and step > 5:
            scene.set_joint_targets(HOME_ARM)
        hand = servo.hand_position(scene.joint_positions())
        distance = float(np.linalg.norm(hand - goal.position))
        if distance <= HOME_DISTANCE:
            say("at home")
            servo.clear()
            return None
        if step % 60 == 0:
            log(f"home step {step}: hand is {distance:.3f} m from the home pose")
    scene.set_joint_targets(HOME_ARM)
    say("home motion finished; holding the home joint targets")
    servo.clear()
    return None


def update_tracker(scene, perception, rgb: np.ndarray, depth: np.ndarray, view) -> None:
    tracked = perception.push_frame(rgb)
    if view is None:
        return
    if tracked is None:
        update_camera_views(scene, view, rgb, depth, perception.mask, perception.query_xy)
    else:
        points, visible = tracked
        update_camera_views(scene, view, rgb, depth, None, points, visible)


def update_camera_views(
    scene,
    view,
    environment_rgb: np.ndarray,
    environment_depth: np.ndarray,
    mask: np.ndarray | None = None,
    points_xy: np.ndarray | None = None,
    visible: np.ndarray | None = None,
) -> None:
    if view is None:
        return
    _, wrist_depth, _ = scene.read_wrist_sensors()
    view.update(
        environment_rgb,
        mask,
        points_xy,
        visible,
        environment_depth=environment_depth,
        wrist_depth=wrist_depth,
    )


def open_gripper_for_next_command(scene, perception, say, view) -> None:
    say("opening gripper for the next command")
    scene.set_gripper_closed(False)
    for _ in range(GRIPPER_SETTLE_STEPS):
        scene.step()
        rgb, depth, _ = scene.read_sensors()
        update_tracker(scene, perception, rgb, depth, view)


def gripper_place(scene, servo, goal) -> tuple[float, float]:
    """Return gripper height above the object and horizontal error from its center."""
    from pseudo_vla.geometry import base_to_world, prim_world_pose

    hand = servo.hand_position(scene.joint_positions())
    base_world, base_quat = prim_world_pose(scene.base_path)
    hand_world = base_to_world(hand.reshape(1, 3), base_world, base_quat)[0]
    height = float(hand_world[2] - goal.top_z)
    horizontal = float(np.linalg.norm(hand_world[:2] - goal.center_world[:2]))
    return height, horizontal
