"""Find, stack, or servo to colored blocks with the environment camera.

The simulation stays open. Type any description and SAM segments it, or type
`stack` to stack red, blue, and green in order. Type `home` to return the hand
to the pose it started in. Type `quit` to close Kit.
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

APPROACH_TOLERANCE = 0.02
HOME_DISTANCE = 0.02
GRIPPER_FINGER_REACH = 0.11
GRIPPER_SETTLE_STEPS = 45
LIFT_HEIGHT = 0.20
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

        say("Type a block color, such as \"red block\", \"green block\", or \"blue block\".")
        say("Two numbers are a pixel. Four numbers are a box.")
        say("Commands: stack, home, quit")
        threading.Thread(target=read_commands, daemon=True).start()

        pending = _initial_command(args)
        prompt_shown = False
        stack_stage = "idle"
        while True:
            if pending is None:
                if not prompt_shown:
                    print("> ", end="", flush=True)
                    prompt_shown = True
                scene.step()
                rgb, depth, _ = scene.read_sensors()
                _update_tracker(scene, perception, rgb, depth, view)
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
            if scene.gripper_closed:
                _open_gripper_for_next_command(scene, perception, say, view)
            if stack_stage == "await_green":
                if command not in {"continue", "green", "green cube"}:
                    say("blue is stacked on red; type continue to stack green on blue, or quit")
                    continue
                complete = _run_stack_phase(
                    scene, perception, servo, "blue", "green", commands, args.max_steps, say, log, view
                )
                if complete:
                    stack_stage = "idle"
                    say("stack complete; object prompts are available")
                continue
            if command == "stack":
                complete = _run_stack_phase(
                    scene, perception, servo, "red", "blue", commands, args.max_steps, say, log, view
                )
                if complete:
                    stack_stage = "await_green"
                    say("blue is stacked on red; type continue when ready to stack green on blue")
                continue
            if command == "home":
                pending = _home(scene, perception, servo, commands, args.max_steps, say, log, view)
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
    from pseudo_vla.perception import ApproachGoal, STANDOFF_Z, prim_world_pose, save_overlay, world_to_base
    from pseudo_vla.scene import object_center, object_dims

    say(f"moving to {prompt}")
    servo.clear()
    started = False
    goal = None
    target_spec = None
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
            _update_camera_views(scene, view, rgb, depth, perception.mask, queries)
            say(f"SAM 2 mask pixels {int(perception.mask.sum())}")
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
                nearest_spec, error = _nearest_object(goal.center_world)
                target_spec = nearest_spec
                nearest_world = object_center(nearest_spec)
                dims = object_dims(nearest_spec)
                top_z = float(nearest_world[2] + dims[2] / 2.0)
                waypoint_world = np.array(
                    [nearest_world[0], nearest_world[1], top_z + STANDOFF_Z],
                    dtype=np.float64,
                )
                base_world, base_quat = prim_world_pose(scene.base_path)
                goal.position = world_to_base(
                    waypoint_world.reshape(1, 3), base_world, base_quat
                )[0]
                goal.center_world = nearest_world.copy()
                goal.top_z = top_z
                log(
                    f"step {step}: centered over {nearest_spec['name']} at "
                    f"{goal.center_world.round(3).tolist()} top z {goal.top_z:.3f} "
                    f"tracked-center error {error:.3f} m "
                    f"goal {goal.position.round(3).tolist()}"
                )
                servo.update_goal(goal, scene.joint_positions())
        if last_points is None:
            _update_camera_views(scene, view, rgb, depth, perception.mask, perception.query_xy)
        else:
            _update_camera_views(scene, view, rgb, depth, None, last_points, last_visible)
        command = servo.command()
        if command:
            scene.set_joint_targets(command)
        if goal is None:
            continue
        height, horizontal = _gripper_place(scene, servo, goal)
        if horizontal <= APPROACH_TOLERANCE and abs(height - STANDOFF_Z) <= APPROACH_TOLERANCE:
            say(f"reached {prompt}, gripper {height:.3f} m above the top")
            if target_spec is None:
                raise RuntimeError("Reached approach waypoint without a selected block")
            hover_goal = ApproachGoal(
                position=goal.position.copy(),
                quaternion=goal.quaternion.copy(),
                center_world=goal.center_world.copy(),
                top_z=goal.top_z,
            )
            return _grasp_and_lift(
                scene,
                servo,
                target_spec,
                hover_goal,
                perception,
                commands,
                max_steps,
                say,
                log,
                view,
            )
    if goal is None:
        say(f"no 3D goal for {prompt}. Check outputs/frame0_overlay.png")
    else:
        height, horizontal = _gripper_place(scene, servo, goal)
        say(f"stopped {horizontal:.3f} m to the side and {height:.3f} m above {prompt}")
    return None


def _grasp_and_lift(scene, servo, spec, hover_goal, perception, commands, max_steps: int, say, log, view) -> str | None:
    from pseudo_vla.perception import ApproachGoal, prim_world_pose, world_to_base
    from pseudo_vla.scene import object_center, object_dims

    center_world = object_center(spec)
    grasp_world = center_world.copy()
    grasp_world[2] += GRIPPER_FINGER_REACH
    base_world, base_quat = prim_world_pose(scene.base_path)
    grasp_goal = ApproachGoal(
        position=world_to_base(grasp_world.reshape(1, 3), base_world, base_quat)[0],
        quaternion=hover_goal.quaternion.copy(),
        center_world=center_world.copy(),
        top_z=hover_goal.top_z,
    )

    say(f"lowering fingers around {spec['name']}")
    servo.set_obstacle_enabled(spec["name"], False)
    try:
        reached, pending = _execute_pose(scene, servo, grasp_goal, perception, commands, max_steps, log, view)
        if pending is not None:
            return pending
        if not reached:
            say(f"could not plan the grasp approach for {spec['name']}; leaving the gripper open")
            return None

        say(f"closing gripper on {spec['name']}")
        scene.set_gripper_closed(True)
        for _ in range(GRIPPER_SETTLE_STEPS):
            incoming = _poll(commands)
            if incoming:
                return incoming
            scene.step()
            rgb, depth, _ = scene.read_sensors()
            _update_tracker(scene, perception, rgb, depth, view)

        top_z = float(center_world[2] + object_dims(spec)[2] / 2.0)
        lift_world = center_world.copy()
        lift_world[2] = top_z + LIFT_HEIGHT
        base_world, base_quat = prim_world_pose(scene.base_path)
        lift_goal = ApproachGoal(
            position=world_to_base(lift_world.reshape(1, 3), base_world, base_quat)[0],
            quaternion=hover_goal.quaternion.copy(),
            center_world=center_world.copy(),
            top_z=top_z,
        )
        say(f"lifting {spec['name']} to {LIFT_HEIGHT:.2f} m above the block")
        reached, pending = _execute_pose(scene, servo, lift_goal, perception, commands, max_steps, log, view)
        if pending is not None:
            return pending
        if not reached:
            say(f"could not plan the lift for {spec['name']}")
            return None

        cube_world, _ = prim_world_pose(scene.object_paths[spec["name"]])
        lift_height = float(cube_world[2] - center_world[2])
        if lift_height < 0.015:
            say(f"gripper lifted, but {spec['name']} did not follow ({lift_height:.3f} m)")
        else:
            say(f"grasped and lifted {spec['name']} by {lift_height:.3f} m")
        return None
    finally:
        servo.set_obstacle_enabled(spec["name"], True)


def _execute_pose(scene, servo, goal, perception, commands, max_steps: int, log, view) -> tuple[bool, str | None]:
    servo.clear()
    servo.update_goal(goal, scene.joint_positions())
    if servo.trajectory is None:
        return False, None

    for step in range(max_steps):
        incoming = _poll(commands)
        if incoming:
            return False, incoming
        scene.step()
        servo.update_goal(goal, scene.joint_positions())
        command = servo.command()
        if command:
            scene.set_joint_targets(command)
        rgb, depth, _ = scene.read_sensors()
        _update_tracker(scene, perception, rgb, depth, view)
        if float(np.linalg.norm(servo.hand_position(scene.joint_positions()) - goal.position)) <= APPROACH_TOLERANCE:
            return True, None
        if servo.trajectory is None:
            log(f"pose motion lost its trajectory at step {step}")
            return False, None
        if step % 60 == 0:
            distance = float(np.linalg.norm(servo.hand_position(scene.joint_positions()) - goal.position))
            log(f"pose step {step}: hand is {distance:.3f} m from target")
    return False, None


def _run_stack_phase(scene, perception, servo, support_name: str, moving_name: str, commands, max_steps: int, say, log, view) -> bool:
    scene.step()
    rgb, depth, camera_params = scene.read_sensors()
    prompts = {}
    for group_name, object_name in (("support", support_name), ("moving", moving_name)):
        point, box = _projected_cube_prompt(scene, rgb, camera_params, object_name, say)
        if point is None and box is None:
            return False
        prompts[group_name] = (point, box)

    perception.segment_groups(rgb, prompts)
    if moving_name == "green":
        perception.retain_groups(rgb, ("moving",))
    _update_camera_views(scene, view, rgb, depth, perception.mask, perception.query_xy)
    if moving_name == "green":
        say(f"tracking {moving_name} only until pickup with {len(perception.query_xy)} CoTracker points")
    else:
        say(
            f"tracking {support_name} support and {moving_name} cube with "
            f"{len(perception.query_xy)} CoTracker points"
        )
    return _stack_tracked_cube(
        scene,
        perception,
        servo,
        support_name,
        moving_name,
        commands,
        max_steps,
        say,
        log,
        view,
    )


def _stack_tracked_cube(scene, perception, servo, support_name: str, moving_name: str, commands, max_steps: int, say, log, view) -> bool:
    from pseudo_vla.perception import (
        ApproachGoal,
        STANDOFF_Z,
        prim_world_pose,
        world_to_base,
    )
    from pseudo_vla.scene import SCENE_OBJECTS, object_dims

    specs = {spec["name"].removesuffix("_block"): spec for spec in SCENE_OBJECTS}
    support_spec = specs[support_name]
    moving_spec = specs[moving_name]
    moving_dims = object_dims(moving_spec)
    hover_goal = None

    say(f"approaching {moving_name} while tracking {support_name} as the placement target")
    servo.clear()
    for step in range(max_steps):
        incoming = _poll(commands)
        if incoming:
            commands.put(incoming)
            return False
        scene.step()
        rgb, depth, camera_params = scene.read_sensors()
        tracked = perception.push_frame(rgb)
        if tracked is not None:
            tracks_xy, visible = tracked
            goal = perception.approach_goal(
                tracks_xy,
                visible,
                depth,
                camera_params,
                scene.camera_path,
                scene.base_path,
                group="moving",
            )
            if goal is None:
                log(f"stack approach {moving_name}: fewer than 4 valid CoTracker depth points")
            else:
                goal.center_world[2] = goal.top_z - moving_dims[2] / 2.0
                hover_goal = goal
                servo.update_goal(goal, scene.joint_positions())
                log(
                    f"stack approach {moving_name} from CoTracker depth: "
                    f"center {goal.center_world.round(3).tolist()} top z {goal.top_z:.3f} "
                    f"base goal {goal.position.round(3).tolist()}"
                )
        _update_camera_views(
            scene,
            view,
            rgb,
            depth,
            None if tracked is not None else perception.mask,
            tracks_xy if tracked is not None else perception.query_xy,
            visible if tracked is not None else None,
        )
        command = servo.command()
        if command:
            scene.set_joint_targets(command)
        if hover_goal is None:
            continue
        height, horizontal = _gripper_place(scene, servo, hover_goal)
        if step % 16 == 0:
            log(
                f"stack hover {moving_name}: horizontal error {horizontal:.3f} m, "
                f"height {height:.3f} m, target {STANDOFF_Z:.3f} m, "
                f"height error {abs(height - STANDOFF_Z):.3f} m"
            )
        if horizontal <= APPROACH_TOLERANCE and abs(height - STANDOFF_Z) <= APPROACH_TOLERANCE:
            break
    else:
        say(f"could not reach the hover waypoint for {moving_name}")
        return False

    say(f"refining {moving_name} alignment with wrist RGB-D")
    wrist_hover_goal = _wrist_rgbd_goal(
        scene, perception, moving_name, hover_goal.center_world, hover_goal.quaternion, moving_dims, say
    )
    if wrist_hover_goal is None:
        say(f"wrist RGB-D could not localize {moving_name}; not attempting the grasp")
        return False
    reached, pending = _execute_pose(
        scene, servo, wrist_hover_goal, perception, commands, max_steps, log, view
    )
    if pending is not None:
        say(f"stack phase interrupted by {pending}")
        commands.put(pending)
        return False
    if not reached:
        say(f"could not reach the wrist-refined hover pose for {moving_name}")
        return False

    grasp_alignment = _wrist_rgbd_goal(
        scene, perception, moving_name, wrist_hover_goal.center_world, hover_goal.quaternion, moving_dims, say
    )
    if grasp_alignment is None:
        say(f"wrist RGB-D could not confirm {moving_name} at grasp height; not closing the gripper")
        return False
    moving_center = grasp_alignment.center_world.copy()
    grasp_world = moving_center.copy()
    grasp_world[2] += GRIPPER_FINGER_REACH
    base_world, base_quat = prim_world_pose(scene.base_path)
    grasp_goal = ApproachGoal(
        position=world_to_base(grasp_world.reshape(1, 3), base_world, base_quat)[0],
        quaternion=grasp_alignment.quaternion.copy(),
        center_world=moving_center.copy(),
        top_z=grasp_alignment.top_z,
    )

    servo.set_obstacle_enabled(moving_spec["name"], False)
    try:
        say(f"grasping {moving_name}")
        reached, pending = _execute_pose(scene, servo, grasp_goal, perception, commands, max_steps, log, view)
        if pending is not None or not reached:
            if pending is not None:
                say(f"stack phase interrupted by {pending}")
                commands.put(pending)
            else:
                say(f"could not lower the gripper to grasp {moving_name}")
            return False
        scene.set_gripper_closed(True)
        _settle_tracker(
            scene,
            perception,
            GRIPPER_SETTLE_STEPS,
            view,
            retain_groups=("support",),
        )

        moving_top = float(moving_center[2] + moving_dims[2] / 2.0)
        lift_world = moving_center.copy()
        lift_world[2] = moving_top + LIFT_HEIGHT
        base_world, base_quat = prim_world_pose(scene.base_path)
        lift_goal = ApproachGoal(
            position=world_to_base(lift_world.reshape(1, 3), base_world, base_quat)[0],
            quaternion=hover_goal.quaternion.copy(),
            center_world=moving_center.copy(),
            top_z=moving_top,
        )
        say(f"lifting {moving_name}")
        reached, pending = _execute_pose(scene, servo, lift_goal, perception, commands, max_steps, log, view)
        if pending is not None or not reached:
            if pending is not None:
                say(f"stack phase interrupted by {pending}")
                commands.put(pending)
            else:
                say(f"could not lift {moving_name}")
            return False

        preplace_goal = None
        place_goal = None
        say(f"tracking {support_name} to plan the {moving_name} placement")
        for step in range(max_steps):
            incoming = _poll(commands)
            if incoming:
                commands.put(incoming)
                return False
            scene.step()
            rgb, depth, camera_params = scene.read_sensors()
            tracked = perception.push_frame(rgb)
            if tracked is not None:
                tracks_xy, visible = tracked
                support_points = perception.tracked_group_points(
                    "support", tracks_xy, visible, depth, camera_params, scene.camera_path
                )
                if support_points.shape[0] >= 4:
                    preplace_goal, place_goal = _stack_placement_goals(
                        support_points,
                        moving_dims,
                        scene.base_path,
                    )
                    servo.update_goal(preplace_goal, scene.joint_positions())
            command = servo.command()
            if command:
                scene.set_joint_targets(command)
            _update_camera_views(
                scene,
                view,
                rgb,
                depth,
                None if tracked is not None else perception.mask,
                tracks_xy if tracked is not None else perception.query_xy,
                visible if tracked is not None else None,
            )
            if preplace_goal is not None and float(
                np.linalg.norm(servo.hand_position(scene.joint_positions()) - preplace_goal.position)
            ) <= APPROACH_TOLERANCE:
                break
        else:
            say(f"could not reach a tracked placement waypoint above {support_name}")
            return False

        assert place_goal is not None and preplace_goal is not None
        servo.set_obstacle_enabled(support_spec["name"], False)
        try:
            say(f"lowering {moving_name} onto tracked {support_name} target")
            reached, pending = _execute_pose(
                scene, servo, place_goal, perception, commands, max_steps, log, view
            )
            if pending is not None or not reached:
                if pending is not None:
                    say(f"stack phase interrupted by {pending}")
                    commands.put(pending)
                else:
                    say(f"could not lower {moving_name} onto {support_name}")
                return False
            scene.set_gripper_closed(False)
            _settle_tracker(scene, perception, GRIPPER_SETTLE_STEPS, view)
        finally:
            servo.set_obstacle_enabled(support_spec["name"], True)

        reached, pending = _execute_pose(
            scene, servo, preplace_goal, perception, commands, max_steps, log, view
        )
        if pending is not None:
            commands.put(pending)
            return False
        if not reached:
            say(f"{moving_name} was released, but the gripper could not retreat")
            return False
        if moving_name in {"blue", "green"}:
            _seed_placed_cube_tracking(scene, perception, moving_name, say, view)
        say(f"placed {moving_name} on {support_name}")
        return True
    finally:
        servo.set_obstacle_enabled(moving_spec["name"], True)
        centers = {
            spec["name"]: prim_world_pose(scene.object_paths[spec["name"]])[0]
            for spec in SCENE_OBJECTS
        }
        servo.update_block_positions(centers)


def _stack_placement_goals(support_points: np.ndarray, moving_dims: np.ndarray, base_path: str):
    from pseudo_vla.perception import (
        ApproachGoal,
        STANDOFF_Z,
        TOP_DOWN_WXYZ,
        prim_world_pose,
        quat_inverse,
        quat_multiply,
        stack_placement_world,
        world_to_base,
    )

    preplace_world, place_world, moving_center, support_top = stack_placement_world(
        support_points,
        moving_dims,
        GRIPPER_FINGER_REACH,
        STANDOFF_Z,
    )
    base_world, base_quat = prim_world_pose(base_path)
    quaternion = quat_multiply(quat_inverse(base_quat), TOP_DOWN_WXYZ)
    quaternion /= np.linalg.norm(quaternion)
    preplace = ApproachGoal(
        position=world_to_base(preplace_world.reshape(1, 3), base_world, base_quat)[0],
        quaternion=quaternion,
        center_world=moving_center.copy(),
        top_z=support_top + moving_dims[2],
    )
    place = ApproachGoal(
        position=world_to_base(place_world.reshape(1, 3), base_world, base_quat)[0],
        quaternion=quaternion.copy(),
        center_world=moving_center.copy(),
        top_z=support_top + moving_dims[2],
    )
    return preplace, place


def _settle_tracker(scene, perception, steps: int, view, retain_groups: tuple[str, ...] | None = None) -> None:
    if retain_groups:
        scene.step()
        rgb, depth, _ = scene.read_sensors()
        perception.retain_groups(rgb, retain_groups)
        _update_camera_views(scene, view, rgb, depth, perception.mask, perception.query_xy)
        steps -= 1
    for _ in range(max(steps, 0)):
        scene.step()
        rgb, depth, _ = scene.read_sensors()
        _update_tracker(scene, perception, rgb, depth, view)


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


def _projected_cube_prompt(scene, rgb: np.ndarray, camera_params: dict, object_name: str, say):
    from pseudo_vla.perception import prim_world_pose, project_world_point

    center, _ = prim_world_pose(scene.object_paths[f"{object_name}_block"])
    height, width = rgb.shape[:2]
    point = project_world_point(
        center,
        camera_params,
        scene.camera_path,
        width,
        height,
    )
    if point is None:
        raise RuntimeError(f"The {object_name} cube is behind the environment camera")
    x, y = (float(value) for value in point)
    if not (0 <= x < width and 0 <= y < height):
        raise RuntimeError(f"The projected {object_name} cube point is outside the camera image: {point.tolist()}")
    say(f"seeding {object_name} SAM mask at projected cube center ({x:.1f}, {y:.1f})")
    return (x, y), None


def _seed_placed_cube_tracking(scene, perception, object_name: str, say, view) -> None:
    scene.step()
    rgb, depth, camera_params = scene.read_sensors()
    point, _ = _projected_cube_prompt(scene, rgb, camera_params, object_name, say)
    perception.segment_groups(rgb, {"placed": (point, None)})
    say(f"dropped stale CoTracker points; tracking the placed {object_name} cube")
    _update_camera_views(scene, view, rgb, depth, perception.mask, perception.query_xy)


def _wrist_rgbd_goal(scene, perception, object_name, center_hint, quaternion, dims, say):
    from pseudo_vla.perception import (
        ApproachGoal,
        STANDOFF_Z,
        mask_depth_points,
        prim_world_pose,
        project_world_point,
        tracked_surface_top,
        world_to_base,
    )

    rgb, depth, camera_params = scene.read_wrist_sensors()
    height, width = depth.shape[:2]
    point = project_world_point(
        center_hint,
        camera_params,
        scene.wrist_camera_path,
        width,
        height,
    )
    if point is None or not (0 <= point[0] < width and 0 <= point[1] < height):
        say(f"{object_name} is outside the wrist camera view")
        return None

    mask = perception.segment_mask(rgb, (float(point[0]), float(point[1])))
    points = mask_depth_points(mask, depth, camera_params, scene.wrist_camera_path)
    if points.shape[0] < 4:
        say(f"wrist RGB-D found only {points.shape[0]} valid {object_name} points")
        return None

    center_world = np.median(points, axis=0)
    top_z = tracked_surface_top(points)
    center_world[2] = top_z - dims[2] / 2.0
    goal_world = np.array(
        [center_world[0], center_world[1], top_z + STANDOFF_Z],
        dtype=np.float64,
    )
    base_world, base_quat = prim_world_pose(scene.base_path)
    goal_base = world_to_base(goal_world.reshape(1, 3), base_world, base_quat)[0]
    say(
        f"wrist RGB-D aligned {object_name}: center {center_world.round(3).tolist()}, "
        f"top z {top_z:.3f} from {points.shape[0]} depth points"
    )
    return ApproachGoal(
        position=goal_base,
        quaternion=quaternion.copy(),
        center_world=center_world,
        top_z=top_z,
    )


def _numbers(prompt: str) -> list[float] | None:
    parts = prompt.split()
    if not parts:
        return None
    try:
        return [float(part) for part in parts]
    except ValueError:
        return None


def _home(scene, perception, servo, commands, max_steps: int, say, log, view) -> str | None:
    from pseudo_vla.scene import HOME_ARM

    say("homing")
    servo.plan_home(scene.joint_positions())
    goal = servo.home_goal
    for step in range(max_steps):
        incoming = _poll(commands)
        if incoming:
            return incoming
        scene.step()
        rgb, depth, _ = scene.read_sensors()
        _update_tracker(scene, perception, rgb, depth, view)
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


def _update_tracker(scene, perception, rgb: np.ndarray, depth: np.ndarray, view) -> None:
    tracked = perception.push_frame(rgb)
    if view is None:
        return
    if tracked is None:
        _update_camera_views(scene, view, rgb, depth, perception.mask, perception.query_xy)
    else:
        points, visible = tracked
        _update_camera_views(scene, view, rgb, depth, None, points, visible)


def _update_camera_views(
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


def _open_gripper_for_next_command(scene, perception, say, view) -> None:
    say("opening gripper for the next command")
    scene.set_gripper_closed(False)
    for _ in range(GRIPPER_SETTLE_STEPS):
        scene.step()
        rgb, depth, _ = scene.read_sensors()
        _update_tracker(scene, perception, rgb, depth, view)


def _gripper_place(scene, servo, goal) -> tuple[float, float]:
    """Gripper height above the object top, and horizontal distance to its tracked center."""
    from pseudo_vla.perception import base_to_world, prim_world_pose

    hand = servo.hand_position(scene.joint_positions())
    base_world, base_quat = prim_world_pose(scene.base_path)
    hand_world = base_to_world(hand.reshape(1, 3), base_world, base_quat)[0]
    height = float(hand_world[2] - goal.top_z)
    horizontal = float(np.linalg.norm(hand_world[:2] - goal.center_world[:2]))
    return height, horizontal


def _nearest_object(point_world: np.ndarray) -> tuple[dict, float]:
    from pseudo_vla.scene import SCENE_OBJECTS, object_center

    nearest_spec = SCENE_OBJECTS[0]
    nearest_error = float("inf")
    for spec in SCENE_OBJECTS:
        center = object_center(spec)
        error = float(np.linalg.norm(point_world - center))
        if error < nearest_error:
            nearest_spec = spec
            nearest_error = error
    return nearest_spec, nearest_error


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
