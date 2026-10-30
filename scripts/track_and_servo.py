"""Find an object with the environment camera, then servo the Franka to it.

The simulation stays open. Type any description and SAM segments it, or type
`home` to return the hand to the pose it started in. Type `quit` to close Kit.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pseudo_vla.scene import launch

# Vertical gap from the gripper down to the top of the tracked object.
ARRIVED_DISTANCE = 0.30
# The gripper must also be over that object. Height alone matches the next target.
OVER_OBJECT_DISTANCE = 0.08
HOME_DISTANCE = 0.02
QUIT_COMMANDS = {"quit", "exit", "q"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=str, help="First text prompt for SAM, before the prompt")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=600, help="Steps to spend on one move before returning to the prompt")
    return parser.parse_args()


def run(app, args: argparse.Namespace) -> None:
    from pseudo_vla.perception import Perception, prim_world_pose, world_to_base
    from pseudo_vla.scene import SCENE_OBJECTS, build_scene, object_center
    from pseudo_vla.servo import ArmServo

    log_path = ROOT / "outputs" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")

    def log(message: str) -> None:
        with log_path.open("a") as handle:
            handle.write(message + "\n")

    def say(message: str) -> None:
        print(message, flush=True)
        log(message)

    commands: queue.Queue[str] = queue.Queue()

    def read_commands() -> None:
        while True:
            line = sys.stdin.readline()
            if line == "":
                commands.put("quit")
                return
            commands.put(line.strip())

    try:
        say("building scene")
        scene = build_scene(app)
        say(f"environment camera {scene.camera_path}")
        say(f"wrist camera {scene.wrist_camera_path}")
        for spec in SCENE_OBJECTS:
            say(f"object {spec['name']} {object_center(spec).round(3).tolist()}")
        say("loading perception")
        perception = Perception()
        say("loading cuRobo")
        base_world, base_quat = prim_world_pose(scene.base_path)
        servo = ArmServo(base_world, base_quat)

        joints = scene.joint_positions()
        hand_world, _ = prim_world_pose(scene.hand_path)
        hand_base = world_to_base(hand_world.reshape(1, 3), base_world, base_quat)[0]
        servo.check_kinematics(joints, hand_base)
        servo.remember_home(joints)

        view = None
        if not args.headless:
            from pseudo_vla.scene import ENV_RESOLUTION, WRIST_RESOLUTION
            from pseudo_vla.view import PerceptionWindows

            view = PerceptionWindows(scene.wrist_camera_path, WRIST_RESOLUTION, ENV_RESOLUTION)
            say("opened Wrist Camera and SAM + CoTracker windows")

        say("Type anything for SAM, such as \"mustard bottle\" or \"tomato soup can\".")
        say("Two numbers are a pixel. Four numbers are a box.")
        say("Commands: home, quit")
        threading.Thread(target=read_commands, daemon=True).start()

        pending = _initial_command(args)
        prompt_shown = False
        while True:
            if pending is None:
                if not prompt_shown:
                    print("> ", end="", flush=True)
                    prompt_shown = True
                scene.step()
                if view is not None:
                    rgb, _, _ = scene.read_sensors()
                    view.update(rgb)
                pending = _poll(commands)
                continue

            command = pending.strip().lower()
            pending = None
            prompt_shown = False
            if not command:
                continue
            if command in QUIT_COMMANDS:
                say("closing")
                return
            if command == "home":
                pending = _home(scene, servo, commands, args.max_steps, say, log, view)
                continue
            pending = _move_to_prompt(scene, perception, servo, command, commands, args.max_steps, say, log, view)
    except Exception:
        import traceback

        say(traceback.format_exc())
        raise


def _initial_command(args: argparse.Namespace) -> str | None:
    return args.target


def _poll(commands: queue.Queue[str]) -> str | None:
    try:
        return commands.get_nowait()
    except queue.Empty:
        return None


def _move_to_prompt(scene, perception, servo, prompt, commands, max_steps: int, say, log, view) -> str | None:
    from pseudo_vla.perception import save_overlay

    say(f"moving to {prompt}")
    servo.clear()
    started = False
    goal = None
    last_points = None
    last_visible = None
    for step in range(max_steps):
        incoming = _poll(commands)
        if incoming:
            return incoming
        scene.step()
        rgb, depth, camera_params = scene.read_sensors()
        if rgb.size == 0 or depth.size == 0:
            log(f"step {step}: waiting for environment RGB-D")
            continue
        if not started:
            point, box = _prompt_to_sam(perception, rgb, prompt, say)
            if point is None and box is None:
                return None
            queries = perception.segment(rgb, point, box)
            save_overlay(ROOT / "outputs" / "frame0_overlay.png", rgb, perception.mask, queries)
            if view is not None:
                view.update(rgb, perception.mask, queries)
            say(f"SAM 2 mask pixels {int(perception.mask.sum())}")
            say("unloaded SAM 2 and OWL-ViT; only CoTracker active on GPU")
            started = True
            continue
        tracked = perception.push_frame(rgb)
        if tracked is not None:
            last_points, last_visible = tracked
            updated = perception.approach_goal(
                last_points,
                last_visible,
                depth,
                camera_params,
                scene.camera_path,
                scene.base_path,
            )
            if updated is None:
                log(f"step {step}: fewer than 4 valid depth points")
            else:
                goal = updated
                nearest_name, nearest_world, error = _nearest_object(goal.median_world)
                log(
                    f"step {step}: median {goal.median_world.round(3).tolist()} "
                    f"nearest {nearest_name} {nearest_world.round(3).tolist()} error {error:.3f} m "
                    f"goal {goal.position.round(3).tolist()}"
                )
                servo.update_goal(goal, scene.joint_positions(), ignore_name=nearest_name)
        if view is not None:
            if last_points is None:
                view.update(rgb, perception.mask, perception.query_xy)
            else:
                view.update(rgb, None, last_points, last_visible)
        command = servo.command()
        if command:
            scene.set_joint_targets(command)
        if goal is None:
            continue
        height, horizontal = _gripper_place(scene, servo, goal)
        if horizontal <= OVER_OBJECT_DISTANCE and height <= ARRIVED_DISTANCE:
            say(f"reached {prompt}, gripper {height:.3f} m above the top")
            servo.clear()
            return None
    if goal is None:
        say(f"no 3D goal for {prompt}. Check outputs/frame0_overlay.png")
    else:
        height, horizontal = _gripper_place(scene, servo, goal)
        say(f"stopped {horizontal:.3f} m to the side and {height:.3f} m above {prompt}")
    return None


def _prompt_to_sam(perception, rgb: np.ndarray, prompt: str, say) -> tuple[tuple[float, float] | None, tuple[float, float, float, float] | None]:
    numbers = _numbers(prompt)
    if numbers is not None and len(numbers) == 2:
        say(f"SAM point {numbers[0]:.1f}, {numbers[1]:.1f}")
        return (numbers[0], numbers[1]), None
    if numbers is not None and len(numbers) == 4:
        say(f"SAM box {numbers}")
        return None, (numbers[0], numbers[1], numbers[2], numbers[3])
    say(f"grounding \"{prompt}\"")
    found = perception.ground(rgb, prompt)
    if found is None:
        say(f"nothing in the environment camera matched \"{prompt}\"")
        return None, None
    box, score = found
    say(f"SAM box {[round(value, 1) for value in box]} score {score:.2f}")
    return None, box


def _numbers(prompt: str) -> list[float] | None:
    parts = prompt.split()
    if not parts:
        return None
    try:
        return [float(part) for part in parts]
    except ValueError:
        return None


def _home(scene, servo, commands, max_steps: int, say, log, view) -> str | None:
    from pseudo_vla.scene import HOME_ARM

    say("homing")
    servo.plan_home(scene.joint_positions())
    goal = servo.home_goal
    for step in range(max_steps):
        incoming = _poll(commands)
        if incoming:
            return incoming
        scene.step()
        if view is not None:
            rgb, _, _ = scene.read_sensors()
            view.update(rgb)
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


def _gripper_place(scene, servo, goal) -> tuple[float, float]:
    """Gripper height above the object top, and horizontal distance to its tracked center."""
    from pseudo_vla.perception import base_to_world, prim_world_pose

    hand = servo.hand_position(scene.joint_positions())
    base_world, base_quat = prim_world_pose(scene.base_path)
    hand_world = base_to_world(hand.reshape(1, 3), base_world, base_quat)[0]
    height = float(hand_world[2] - goal.top_z)
    horizontal = float(np.linalg.norm(hand_world[:2] - goal.median_world[:2]))
    return height, horizontal


def _nearest_object(median_world: np.ndarray) -> tuple[str, np.ndarray, float]:
    from pseudo_vla.scene import SCENE_OBJECTS, object_center

    nearest_name = ""
    nearest_world = object_center(SCENE_OBJECTS[0])
    nearest_error = float("inf")
    for spec in SCENE_OBJECTS:
        center = object_center(spec)
        error = float(np.linalg.norm(median_world - center))
        if error < nearest_error:
            nearest_name = spec["name"]
            nearest_world = center
            nearest_error = error
    return nearest_name, nearest_world, nearest_error


def main() -> None:
    args = parse_args()
    app = launch(args.headless)
    try:
        run(app, args)
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
