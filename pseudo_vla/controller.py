"""Robot controller for natural-language pick-and-place tasks in Isaac Sim.

The simulation stays open for natural-language instructions. Type `home` to
return the hand to its starting pose, or `quit` to close Kit.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pseudo_vla.motion import (
    APPROACH_TOLERANCE,
    GRIPPER_SETTLE_STEPS,
    execute_pose as _execute_pose,
    gripper_place as _gripper_place,
    home as _home,
    open_gripper_for_next_command as _open_gripper_for_next_command,
    poll_command as _poll,
    close_gripper_in_place as _close_gripper_in_place,
    settle_tracker as _settle_tracker,
    update_camera_views as _update_camera_views,
    update_tracker as _update_tracker,
)
from pseudo_vla.stacking_workflow import (
    run_prompted_stack_sequence,
    run_requested_stack_relations,
)
from pseudo_vla.task_workflow import run_task_plan

GRIPPER_FINGER_REACH = 0.07
LIFT_HEIGHT = 0.20
STACK_TRANSIT_CLEARANCE = 0.15
APPROACH_TRANSIT_RADIUS = 0.35
APPROACH_TRANSIT_HEIGHT = 0.02
QUIT_COMMANDS = {"quit", "exit", "q"}


@dataclass(frozen=True)
class EnvObject:
    """One object's pose measured from the environment camera's depth image."""

    object_id: str
    spec_name: str
    center_world: np.ndarray
    top_z: float
    dims: np.ndarray
    box: tuple[float, float, float, float]


def run(app, args: argparse.Namespace) -> None:
    from pseudo_vla.perception import Perception, prim_world_pose, world_to_base
    from pseudo_vla.scene import build_scene
    from pseudo_vla.servo import ArmServo

    from pseudo_vla.runlog import append_run_log

    def log(message: str) -> None:
        append_run_log(message)

    def say(message: str) -> None:
        print(message, flush=True)

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
        scene = build_scene(app, getattr(args, "scene", "blocks"))
        say(f"environment camera {scene.camera_path}")
        say(f"wrist camera {scene.wrist_camera_path}")
        say("loading perception")
        perception = Perception()
        say("loading cuRobo")
        base_world, base_quat = prim_world_pose(scene.base_path)
        servo = ArmServo(base_world, base_quat)
        say("measuring objects from the environment camera")
        _refresh_object_measurements(scene, perception, servo, say)

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

        from pseudo_vla.scene import SCENE_NAME, SCENE_OBJECTS, object_id, surface_location

        if SCENE_NAME == "kitchen":
            say(
                'Enter a pick-and-place instruction, such as "hold mug and place it on the plate" '
                'or "move the fork to the left of the spoon".'
            )
        else:
            say(
                'Enter a pick-and-place instruction, such as "stack blue on red", '
                '"unstack the tower", or "put blue to the left of red".'
            )
        say(
            f"This {SCENE_NAME} scene has {', '.join(object_id(spec) for spec in SCENE_OBJECTS)}. "
            f"Place them at {surface_location()}, on, or beside one another."
        )
        say("Commands: home, quit")
        threading.Thread(target=read_commands, daemon=True).start()

        stack_order = getattr(args, "stack_order", None)
        if stack_order is not None:
            complete = _run_prompted_stack_sequence(
                scene, perception, servo, stack_order, commands, args.max_steps, say, log, view
            )
            if complete:
                say("prompted stack complete; object prompts are available")
            else:
                say("prompted stack stopped; returning to the interactive prompt")

        pending = _initial_command(args)
        prompt_shown = False
        while True:
            if pending is None:
                if not prompt_shown:
                    print("Instruction> ", end="", flush=True)
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
            if command == "home":
                pending = _home(scene, perception, servo, commands, args.max_steps, say, log, view)
                continue
            pending = _dispatch_user_prompt(
                scene,
                perception,
                servo,
                command,
                commands,
                args.max_steps,
                say,
                log,
                view,
            )
    except Exception:
        import traceback

        say(traceback.format_exc())
        raise


def _initial_command(args: argparse.Namespace) -> str | None:
    return getattr(args, "instruction", None) or getattr(args, "target", None)


def _dispatch_user_prompt(
    scene,
    perception,
    servo,
    prompt: str,
    commands,
    max_steps: int,
    say,
    log,
    view,
) -> str | None:
    numbers = _numbers(prompt)
    if numbers is not None and len(numbers) in {2, 4}:
        return _move_to_prompt(
            scene, perception, servo, prompt, commands, max_steps, say, log, view
        )

    from pseudo_vla.stacking_llm import StackingModelError
    from pseudo_vla.task_llm import interpret_task_prompt

    try:
        parsed = interpret_task_prompt(prompt, _task_scene_state(scene, perception))
    except (StackingModelError, ValueError) as error:
        say(f"could not interpret prompt with Llama: {error}")
        return None

    if parsed.intent == "find":
        say(f'LLM planned to find "{parsed.query}"')

        def find_object(query):
            return _find_object(scene, perception, prompt, query, say, view)

        complete = run_task_plan(
            parsed,
            say,
            find_object,
            lambda _action, _authorize: False,
        )
        say("find task complete" if complete else "find task stopped")
        return None

    descriptions = []
    for action in parsed.actions:
        destination = action.destination.kind
        if action.destination.target:
            destination += f" {action.destination.target}"
        descriptions.append(f"move {action.object_id} {destination}")
        if _is_table_destination(action.destination) and not _table_slot_is_clear(
            scene, action.object_id
        ):
            say(
                f"could not execute plan: {action.destination.target} destination for "
                f"{action.object_id} is occupied"
            )
            return None
        if action.destination.kind in {"left_of", "right_of"} and not _relative_slot_is_clear(
            scene, action.object_id, action.destination
        ):
            say(
                f"could not execute plan: destination {action.destination.kind} "
                f"{action.destination.target} is occupied"
            )
            return None
    say("validated plan: " + " then ".join(descriptions))

    def execute_action(action, authorize_event):
        destination = action.destination
        if destination.kind == "inside":
            say(f"destination {destination.target!r} is not available in this scene")
            return False
        from pseudo_vla.scene import surface_location

        if destination.kind == "at" and destination.target != surface_location():
            say(f"location {destination.target!r} has no placement behavior in this scene")
            return False
        if destination.kind in {"left_of", "right_of"} and not _relative_slot_is_clear(
            scene, action.object_id, destination
        ):
            say(
                f"destination {destination.kind} {destination.target} is occupied; "
                "not moving the block"
            )
            return False
        if destination.kind == "on":
            support_name = destination.target
        else:
            support_name = destination.target

        def authorize_stack_stage(stage, completed=False):
            if stage in {"lift", "verify_placement"}:
                return True
            return authorize_event(stage, completed)

        return _run_stack_phase(
            scene,
            perception,
            servo,
            support_name,
            action.object_id,
            commands,
            max_steps,
            say,
            log,
            view,
            authorize_stack_stage,
            destination,
            prompt,
        )

    complete = run_task_plan(
        parsed,
        say,
        lambda _query: False,
        execute_action,
    )
    say("task plan complete" if complete else "task plan stopped safely")
    return None


def _task_scene_state(scene, perception) -> dict[str, object]:
    from pseudo_vla.scene import SCENE_OBJECTS, object_id, surface_location

    observed = _observe_objects(scene, perception)
    _remember_home(scene, observed)
    objects = []
    centers = {}
    for spec in SCENE_OBJECTS:
        object_id_name = object_id(spec)
        item = observed.get(object_id_name)
        entry: dict[str, object] = {"id": object_id_name}
        if item is not None:
            center = item.center_world
            centers[object_id_name] = center
            entry["position_world_m"] = [round(float(value), 4) for value in center]
        objects.append(entry)
    supported_by = {}
    for object_id, center in centers.items():
        supports = [
            (float(candidate[2]), candidate_id)
            for candidate_id, candidate in centers.items()
            if candidate_id != object_id
            and 0.03 <= float(center[2] - candidate[2]) <= 0.08
            and float(np.linalg.norm(center[:2] - candidate[:2])) <= 0.035
        ]
        if supports:
            supported_by[object_id] = max(supports)[1]
    return {
        "objects": objects,
        "locations": [surface_location()],
        "containers": [],
        "supported_by": supported_by,
    }


def _find_object(scene, perception, prompt: str, query: str, say, view) -> bool:
    from pseudo_vla.perception import save_overlay

    scene.step()
    rgb, depth, _ = scene.read_sensors()
    if rgb.size == 0 or depth.size == 0:
        say("could not find object: environment camera has no RGB-D frame")
        return False
    point, box = _prompt_to_sam(
        perception,
        rgb,
        prompt,
        say,
        grounding_prompt=query,
    )
    if point is None and box is None:
        return False
    queries = perception.segment(rgb, point, box)
    save_overlay(ROOT / "outputs" / "frame0_overlay.png", rgb, perception.mask, queries)
    _update_camera_views(scene, view, rgb, depth, perception.mask, queries)
    say(f'found "{query}" in the environment camera')
    return True


def _run_requested_stack_relations(
    scene,
    perception,
    servo,
    relations,
    commands,
    max_steps: int,
    say,
    log,
    view,
) -> bool:
    return run_requested_stack_relations(
        scene,
        perception,
        servo,
        relations,
        commands,
        max_steps,
        say,
        log,
        view,
        _run_stack_phase,
    )


def _move_to_prompt(
    scene,
    perception,
    servo,
    prompt,
    commands,
    max_steps: int,
    say,
    log,
    view,
    *,
    grounding_prompt: str | None = None,
) -> str | None:
    from pseudo_vla.perception import ApproachGoal, STANDOFF_Z, save_overlay

    target_prompt = grounding_prompt or prompt
    say(f"moving to {prompt}")
    _observe_objects(scene, perception, say)
    _remember_home(scene, _observed(scene))
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
            point, box = _prompt_to_sam(
                perception,
                rgb,
                target_prompt,
                say,
                grounding_prompt=grounding_prompt,
            )
            if point is None and box is None:
                return None
            tracked_group = _tracked_prompt_group(perception, target_prompt)
            if tracked_group is None:
                queries = perception.segment(rgb, point, box)
            else:
                group_name, _ = tracked_group
                queries = perception.segment_groups(
                    rgb, {group_name: (point, box)}
                )[group_name]
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
                nearest, error = _nearest_observed(goal.center_world, _observed(scene))
                if nearest is not None:
                    target_spec = {"name": nearest.spec_name}
                log(
                    f"step {step}: environment depth center {goal.center_world.round(3).tolist()} "
                    f"top z {goal.top_z:.3f} "
                    f"nearest measured object error {error:.3f} m "
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
        hand_dist = float(np.linalg.norm(servo.hand_position(scene.joint_positions()) - goal.position))
        axial, lateral = _gripper_place(scene, servo, goal)
        if hand_dist <= APPROACH_TOLERANCE:
            say(f"reached {prompt}, gripper at hover waypoint")
            if target_spec is None:
                say("tracked depth did not match an object measured by the environment camera")
                return None
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
        hand_dist = float(np.linalg.norm(servo.hand_position(scene.joint_positions()) - goal.position))
        say(f"stopped {hand_dist:.3f} m from {prompt} hover waypoint")
    return None


def _grasp_and_lift(scene, servo, spec, hover_goal, perception, commands, max_steps: int, say, log, view) -> str | None:
    from pseudo_vla.geometry import adaptive_approach_pose, base_to_world_quat
    from pseudo_vla.perception import ApproachGoal, prim_world_pose, world_to_base

    center_world = hover_goal.center_world.copy()
    base_world, base_quat = prim_world_pose(scene.base_path)
    quat_world = base_to_world_quat(hover_goal.quaternion, base_quat)
    grasp_world = adaptive_approach_pose(
        center_world,
        quat_world,
        standoff=0.0,
        finger_reach=GRIPPER_FINGER_REACH,
    )
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
        incoming = _close_gripper_in_place(scene, perception, commands, view)
        if incoming:
            return incoming

        lift_world = grasp_world.copy()
        lift_world[2] += LIFT_HEIGHT
        base_world, base_quat = prim_world_pose(scene.base_path)
        lift_goal = ApproachGoal(
            position=world_to_base(lift_world.reshape(1, 3), base_world, base_quat)[0],
            quaternion=grasp_goal.quaternion.copy(),
            center_world=center_world.copy(),
            top_z=float(hover_goal.top_z),
        )
        say(f"lifting {spec['name']} by {LIFT_HEIGHT:.2f} m")
        reached, pending = _execute_pose(scene, servo, lift_goal, perception, commands, max_steps, log, view)
        if pending is not None:
            return pending
        if not reached:
            say(f"could not plan the lift for {spec['name']}")
            return None

        from pseudo_vla.scene import object_id

        measured = _observe_objects(scene, perception).get(object_id(spec))
        if measured is None:
            say(f"environment camera could not remeasure {spec['name']} after the lift")
        else:
            lift_height = float(measured.center_world[2] - center_world[2])
            if lift_height < 0.015:
                say(f"gripper lifted, but {spec['name']} did not follow ({lift_height:.3f} m)")
            else:
                say(f"grasped and lifted {spec['name']} by {lift_height:.3f} m")
        return None
    finally:
        servo.set_obstacle_enabled(spec["name"], True)


def _run_stack_phase(
    scene,
    perception,
    servo,
    support_name: str | None,
    moving_name: str,
    commands,
    max_steps: int,
    say,
    log,
    view,
    authorize_event=None,
    destination=None,
    instruction: str = "",
) -> bool:
    from pseudo_vla.affordance import grasp_region_points, match_seen_affordance
    from pseudo_vla.task_plan import Destination

    if destination is None:
        destination = Destination(kind="on", target=support_name)
    scene.step()
    rgb, depth, camera_params = scene.read_sensors()
    measured = _measure_frame(perception, rgb, depth, camera_params, scene.camera_path, say)
    _remember_home(scene, measured)
    _store_observed(scene, measured)
    prompts = {}
    target_name = None if destination.kind == "at" else destination.target
    tracked_objects = []
    if target_name is not None:
        if target_name not in measured:
            say(f"environment camera could not localize {target_name}")
            return False
        tracked_objects.append(("support", target_name))
    tracked_objects.append(("moving", moving_name))
    for group_name, object_name in tracked_objects:
        item = measured.get(object_name)
        if item is None:
            say(f"environment camera could not localize {object_name}")
            return False
        prompts[group_name] = (None, item.box)

    perception.segment_groups(rgb, prompts)
    affordance = match_seen_affordance(instruction)
    if affordance is None:
        say(f"no seen affordance in the instruction; tracking the full SAM mask of {moving_name}")
    else:
        region_points = grasp_region_points(rgb, perception.group_masks["moving"], affordance)
        if region_points is None:
            say(
                f"seen affordance {affordance} had no grasp region inside {moving_name}; "
                "tracking the full SAM mask"
            )
            affordance = None
        else:
            perception.replace_group_points(rgb, "moving", region_points)
            say(
                f"seen affordance {affordance} seeded {len(region_points)} CoTracker points "
                f"on the grasp region of {moving_name}"
            )
    _update_camera_views(scene, view, rgb, depth, perception.mask, perception.query_xy)
    if target_name is not None:
        say(
            f"tracking {target_name} and {moving_name} with "
            f"{len(perception.query_xy)} CoTracker points"
        )
    else:
        say(f"tracking {moving_name} with {len(perception.query_xy)} CoTracker points")
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
        authorize_event,
        destination,
        affordance,
    )


def _run_prompted_stack_sequence(
    scene,
    perception,
    servo,
    goal_order: list[str],
    commands,
    max_steps: int,
    say,
    log,
    view,
) -> bool:
    return run_prompted_stack_sequence(
        scene,
        perception,
        servo,
        goal_order,
        commands,
        max_steps,
        say,
        log,
        view,
        _run_stack_phase,
    )


def _transfer_env_tracks_to_wrist(scene, points_world: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Project environment-camera tracks into the wrist image and read wrist depth there."""
    from pseudo_vla.geometry import (
        project_tracks_into_view,
        project_world_point,
        tracked_surface_top,
        unproject_tracks,
    )

    _rgb, depth, camera_params = scene.read_wrist_sensors()
    if depth.size == 0 or np.asarray(points_world).size == 0:
        return None
    height, width = depth.shape[:2]

    def project(point: np.ndarray):
        return project_world_point(point, camera_params, scene.wrist_camera_path, width, height)

    pixels = project_tracks_into_view(points_world, project, width, height)
    if len(pixels) < 4:
        return None
    cloud = unproject_tracks(
        pixels,
        np.ones(len(pixels), dtype=bool),
        depth,
        camera_params,
        scene.wrist_camera_path,
    )
    if cloud.shape[0] < 4:
        return None
    return np.median(cloud, axis=0), tracked_surface_top(cloud)


def _set_obstacle_enabled(servo, name: str, enabled: bool) -> bool:
    try:
        servo.set_obstacle_enabled(name, enabled)
    except ValueError:
        return False
    return True


def _stack_tracked_cube(
    scene,
    perception,
    servo,
    support_name: str | None,
    moving_name: str,
    commands,
    max_steps: int,
    say,
    log,
    view,
    authorize_event=None,
    destination=None,
    affordance: str | None = None,
) -> bool:
    from pseudo_vla.geometry import (
        adaptive_approach_pose,
        adaptive_grasp_orientation,
        base_to_world_quat,
        world_to_base_quat,
    )
    from pseudo_vla.perception import (
        ApproachGoal,
        STANDOFF_Z,
        prim_world_pose,
        world_to_base,
    )
    from pseudo_vla.scene import SCENE_OBJECTS, object_id

    specs = {object_id(spec): spec for spec in SCENE_OBJECTS}
    if destination is None:
        from pseudo_vla.task_plan import Destination

        destination = Destination(kind="on", target=support_name)
    target_name = None if destination.kind == "at" else destination.target
    support_spec = specs.get(target_name) if target_name is not None else None
    moving_spec = specs[moving_name]
    moving_item = _observed(scene).get(moving_name)
    if moving_item is None:
        say(f"environment camera has no depth pose for {moving_name}")
        return False
    moving_dims = moving_item.dims
    hover_goal = None
    hover_surface = None
    hover_points = None
    transit = _needs_approach_transit(scene, moving_name)

    if authorize_event is not None and not authorize_event("approach"):
        return False
    destination_label = destination.kind
    if destination.target:
        destination_label = f"{destination.kind} {destination.target}"
    say(f"approaching {moving_name} for destination {destination_label}")
    servo.clear()
    grasp_box_ignored = _set_obstacle_enabled(servo, moving_spec["name"], False)
    if grasp_box_ignored:
        say(f"ignoring the {moving_name} collision box so the gripper can reach the grasp")

    def stop_approach(pending=None) -> bool:
        if grasp_box_ignored:
            _set_obstacle_enabled(servo, moving_spec["name"], True)
        if pending is not None:
            commands.put(pending)
        return False

    for step in range(max_steps):
        incoming = _poll(commands)
        if incoming:
            commands.put(incoming)
            return stop_approach()
        scene.step()
        rgb, depth, camera_params = scene.read_sensors()
        tracked = perception.push_frame(rgb)
        if tracked is not None:
            tracks_xy, visible = tracked
            obj_center = moving_item.center_world if moving_item is not None else None
            is_tall_or_mug = (
                moving_spec.get("kind") in {"asset", "mug", "cup"}
                or affordance == "hold"
            )
            grasp_style = "tilted" if is_tall_or_mug else "auto"
            goal = perception.approach_goal(
                tracks_xy,
                visible,
                depth,
                camera_params,
                scene.camera_path,
                scene.base_path,
                group="moving",
                object_center=obj_center,
                grasp_style=grasp_style,
            )
            if goal is None:
                log(f"stack approach {moving_name}: fewer than 4 valid CoTracker depth points")
            else:
                surface = goal.center_world.copy()
                if hover_goal is not None and abs(float(goal.top_z) - float(hover_goal.top_z)) > 0.04:
                    log(
                        f"stack approach {moving_name}: ignoring depth spike "
                        f"top z {goal.top_z:.3f}"
                    )
                else:
                    goal.center_world[2] = goal.top_z - moving_dims[2] / 2.0
                    hover_goal = goal
                    hover_surface = surface
                    hover_points = perception.tracked_group_points(
                        "moving",
                        tracks_xy,
                        visible,
                        depth,
                        camera_params,
                        scene.camera_path,
                    )
                    plan_goal = hover_goal
                    if transit:
                        base_world, base_quat = prim_world_pose(scene.base_path)
                        plan_goal = _stack_transit_goal(hover_goal, base_world, base_quat)
                    servo.update_goal(plan_goal, scene.joint_positions())
                    log(
                        f"stack approach {moving_name} from CoTracker depth: "
                        f"center {hover_surface.round(3).tolist()} top z {hover_goal.top_z:.3f} "
                        f"base goal {plan_goal.position.round(3).tolist()}"
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
        current_hand = servo.hand_position(scene.joint_positions())
        hand_error = float(np.linalg.norm(current_hand - plan_goal.position))
        axial, lateral = _gripper_place(scene, servo, plan_goal)
        if step % 16 == 0:
            log(
                f"stack hover {moving_name}: goal error {hand_error:.3f} m, "
                f"axial {axial:.3f} m, lateral {lateral:.3f} m"
            )
        if hand_error <= APPROACH_TOLERANCE:
            if transit:
                transit = False
                servo.update_goal(hover_goal, scene.joint_positions())
                say(f"at safe approach height above {moving_name}; descending to the hover waypoint")
                continue
            break
    else:
        say(f"could not reach the hover waypoint for {moving_name}")
        return stop_approach()

    if affordance:
        source_points = hover_points if hover_points is not None and len(hover_points) >= 4 else hover_surface.reshape(1, 3)
        transferred = _transfer_env_tracks_to_wrist(scene, source_points)
        if transferred is None:
            say(
                "could not transfer the CoTracker points into the wrist camera; "
                "aligning over the environment-camera points"
            )
            moving_center = hover_surface.copy()
            grasp_top = float(hover_goal.top_z)
        else:
            moving_center, grasp_top = transferred
            say(
                f"transferred CoTracker points into the wrist camera at "
                f"{moving_center.round(3).tolist()}"
            )
        base_world, base_quat = prim_world_pose(scene.base_path)
        quat_world = adaptive_grasp_orientation(
            source_points,
            object_center=obj_center,
            grasp_style=grasp_style,
        )
        grasp_quaternion = world_to_base_quat(quat_world, base_quat)
        align_world = adaptive_approach_pose(
            moving_center,
            quat_world,
            standoff=STANDOFF_Z,
            finger_reach=GRIPPER_FINGER_REACH,
        )
        align_goal = ApproachGoal(
            position=world_to_base(align_world.reshape(1, 3), base_world, base_quat)[0],
            quaternion=grasp_quaternion,
            center_world=moving_center.copy(),
            top_z=grasp_top,
        )
        say(f"aligning over the {affordance} points before the grasp")
        reached, pending = _execute_pose(
            scene, servo, align_goal, perception, commands, max_steps, log, view
        )
        if pending is not None:
            say(f"pick-and-place interrupted by {pending}")
            return stop_approach(pending)
        if not reached:
            say(f"could not align over the {affordance} points of {moving_name}")
            return stop_approach()
        transferred = _transfer_env_tracks_to_wrist(scene, source_points)
        if transferred is not None:
            moving_center, grasp_top = transferred
            say(
                f"wrist view of the transferred points is {moving_center.round(3).tolist()}, "
                f"top z {grasp_top:.3f}"
            )
        if authorize_event is not None:
            authorize_event("approach", completed=True)
    else:
        say(f"refining {moving_name} alignment with wrist RGB-D")
        wrist_hover_goal = _wrist_rgbd_goal(
            scene, perception, moving_name, hover_goal.center_world, hover_goal.quaternion, moving_dims, say
        )
        if wrist_hover_goal is None:
            say(f"wrist RGB-D could not localize {moving_name}; not attempting the grasp")
            return stop_approach()
        reached, pending = _execute_pose(
            scene, servo, wrist_hover_goal, perception, commands, max_steps, log, view
        )
        if pending is not None:
            say(f"pick-and-place interrupted by {pending}")
            return stop_approach(pending)
        if not reached:
            say(f"could not reach the wrist-refined hover pose for {moving_name}")
            return stop_approach()
        if authorize_event is not None:
            authorize_event("approach", completed=True)

        grasp_alignment = _wrist_rgbd_goal(
            scene, perception, moving_name, wrist_hover_goal.center_world, hover_goal.quaternion, moving_dims, say
        )
        if grasp_alignment is None:
            say(f"wrist RGB-D could not confirm {moving_name} at grasp height; not closing the gripper")
            return stop_approach()
        moving_center = grasp_alignment.center_world.copy()
        grasp_top = float(grasp_alignment.top_z)
        grasp_quaternion = grasp_alignment.quaternion.copy()
    base_world, base_quat = prim_world_pose(scene.base_path)
    quat_world = base_to_world_quat(grasp_quaternion, base_quat)
    grasp_world = adaptive_approach_pose(
        moving_center,
        quat_world,
        standoff=0.0,
        finger_reach=GRIPPER_FINGER_REACH,
    )
    grasp_goal = ApproachGoal(
        position=world_to_base(grasp_world.reshape(1, 3), base_world, base_quat)[0],
        quaternion=grasp_quaternion,
        center_world=moving_center.copy(),
        top_z=grasp_top,
    )

    if authorize_event is not None and not authorize_event("grasp"):
        return stop_approach()
    _set_obstacle_enabled(servo, moving_spec["name"], False)
    try:
        say(f"grasping {moving_name}")
        reached, pending = _execute_pose(scene, servo, grasp_goal, perception, commands, max_steps, log, view)
        if pending is not None or not reached:
            if pending is not None:
                say(f"pick-and-place interrupted by {pending}")
                commands.put(pending)
            else:
                say(f"could not lower the gripper to grasp {moving_name}")
            return False
        retained_groups = ("support",) if destination.kind == "on" else None
        if retained_groups:
            scene.step()
            rgb, depth, _ = scene.read_sensors()
            perception.retain_groups(rgb, retained_groups)
            _update_camera_views(scene, view, rgb, depth, perception.mask, perception.query_xy)
        say(f"closing the gripper on {moving_name}")
        pending = _close_gripper_in_place(scene, perception, commands, view)
        if pending is not None:
            say(f"pick-and-place interrupted by {pending}")
            commands.put(pending)
            return False
        if authorize_event is not None:
            authorize_event("grasp", completed=True)

        moving_top = grasp_top
        lift_world = grasp_world.copy()
        lift_world[2] += LIFT_HEIGHT
        base_world, base_quat = prim_world_pose(scene.base_path)
        lift_goal = ApproachGoal(
            position=world_to_base(lift_world.reshape(1, 3), base_world, base_quat)[0],
            quaternion=grasp_goal.quaternion.copy(),
            center_world=moving_center.copy(),
            top_z=moving_top,
        )
        if authorize_event is not None and not authorize_event("lift"):
            return False
        say(f"lifting {moving_name}")
        reached, pending = _execute_pose(scene, servo, lift_goal, perception, commands, max_steps, log, view)
        if pending is not None or not reached:
            if pending is not None:
                say(f"pick-and-place interrupted by {pending}")
                commands.put(pending)
            else:
                say(f"could not lift {moving_name}")
            return False
        if authorize_event is not None:
            authorize_event("lift", completed=True)

        preplace_goal = None
        place_goal = None
        say(f"planning {moving_name} placement at {destination_label}")
        if _is_table_destination(destination) and not _table_slot_is_clear(scene, moving_name):
            say(f"{destination.target} destination for {moving_name} is occupied; not moving the block")
            return False
        for step in range(max_steps):
            incoming = _poll(commands)
            if incoming:
                commands.put(incoming)
                return False
            scene.step()
            rgb, depth, camera_params = scene.read_sensors()
            tracked = perception.push_frame(rgb)
            tracks_xy = None
            visible = None
            if tracked is not None:
                tracks_xy, visible = tracked
            if destination.kind == "on" and tracks_xy is not None:
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
            elif destination.kind != "on":
                if (
                    destination.kind in {"left_of", "right_of"}
                    and tracked is not None
                    and "support" in getattr(perception, "query_groups", {})
                ):
                    support_points = perception.tracked_group_points(
                        "support", tracks_xy, visible, depth, camera_params, scene.camera_path
                    )
                    _update_observed_from_points(scene, destination.target, support_points)
                preplace_goal, place_goal = _relative_placement_goals(
                    scene, moving_name, destination, scene.base_path
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
                None if tracks_xy is not None else perception.mask,
                tracks_xy if tracks_xy is not None else perception.query_xy,
                visible,
            )
            if preplace_goal is not None and float(
                np.linalg.norm(servo.hand_position(scene.joint_positions()) - preplace_goal.position)
            ) <= APPROACH_TOLERANCE:
                break
        else:
            say(f"could not reach the placement waypoint for {destination_label}")
            return False

        assert place_goal is not None and preplace_goal is not None
        if destination.kind in {"left_of", "right_of"} and not _relative_slot_is_clear(
            scene, moving_name, destination
        ):
            say(
                f"destination {destination.kind} {destination.target} is occupied; "
                "not releasing the block"
            )
            return False
        if authorize_event is not None and not authorize_event("place"):
            return False
        if destination.kind == "on" and support_spec is not None:
            servo.set_obstacle_enabled(support_spec["name"], False)
        try:
            say(f"placing {moving_name} at {destination_label}")
            reached, pending = _execute_pose(
                scene, servo, place_goal, perception, commands, max_steps, log, view
            )
            if pending is not None or not reached:
                if pending is not None:
                    say(f"pick-and-place interrupted by {pending}")
                    commands.put(pending)
                else:
                    say(f"could not place {moving_name} at {destination_label}")
                return False
            scene.set_gripper_closed(False)
            _settle_tracker(scene, perception, GRIPPER_SETTLE_STEPS, view)
            if authorize_event is not None:
                authorize_event("place", completed=True)
        finally:
            if destination.kind == "on" and support_spec is not None:
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
        if not _confirm_placement(scene, perception, moving_name, destination, say):
            return False
        _seed_placed_cube_tracking(scene, perception, moving_name, say, view)
        if authorize_event is not None and not authorize_event("verify_placement"):
            return False
        if authorize_event is not None:
            authorize_event("verify_placement", completed=True)
        say(f"placed {moving_name} at {destination_label}")
        return True
    finally:
        servo.set_obstacle_enabled(moving_spec["name"], True)
        measured = _observe_objects(scene, perception)
        if measured:
            servo.update_block_positions(
                {item.spec_name: item.center_world for item in measured.values()},
                {item.spec_name: item.dims for item in measured.values()},
            )


def _stack_transit_goal(hover_goal, base_world: np.ndarray, base_quat: np.ndarray):
    from pseudo_vla.perception import ApproachGoal, STANDOFF_Z, world_to_base

    transit_world = hover_goal.center_world.copy()
    transit_world[2] = hover_goal.top_z + STANDOFF_Z + STACK_TRANSIT_CLEARANCE
    return ApproachGoal(
        position=world_to_base(transit_world.reshape(1, 3), base_world, base_quat)[0],
        quaternion=hover_goal.quaternion.copy(),
        center_world=hover_goal.center_world.copy(),
        top_z=hover_goal.top_z,
    )


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


def _relative_placement_goals(scene, moving_name: str, destination, base_path: str):
    from pseudo_vla.perception import (
        ApproachGoal,
        STANDOFF_Z,
        TOP_DOWN_WXYZ,
        prim_world_pose,
        quat_inverse,
        quat_multiply,
        world_to_base,
    )
    from pseudo_vla.scene import TABLE_TOP_Z, surface_location

    observed = _observed(scene)
    moving = observed.get(moving_name)
    if moving is None:
        raise ValueError(f"No environment-camera pose for {moving_name}")
    moving_dims = moving.dims
    if destination.kind == "at" and destination.target == surface_location():
        home = getattr(scene, "home_xy", {}).get(moving_name)
        if home is None:
            raise ValueError(f"No environment-camera home position for {moving_name}")
        center_world = np.array(
            [home[0], home[1], TABLE_TOP_Z + moving_dims[2] / 2.0 + 0.002],
            dtype=np.float64,
        )
    else:
        target_name = destination.target
        target = observed.get(target_name) if target_name is not None else None
        if target is None:
            raise ValueError(f"No environment-camera pose for {target_name}")
        target_dims = target.dims
        center_world = target.center_world.copy()
        if destination.kind in {"left_of", "right_of"}:
            # Robot faces +Y, so its left is world -X and its right is world +X.
            direction = -1.0 if destination.kind == "left_of" else 1.0
            center_world[0] += direction * (
                target_dims[0] / 2.0 + moving_dims[0] / 2.0 + 0.02
            )
        elif destination.kind == "on":
            center_world[2] += target_dims[2] / 2.0 + moving_dims[2] / 2.0
        else:
            raise ValueError(f"Unsupported destination kind {destination.kind!r}")

    top_z = float(center_world[2] + moving_dims[2] / 2.0)
    place_world = center_world.copy()
    place_world[2] = top_z - moving_dims[2] / 2.0 + GRIPPER_FINGER_REACH
    preplace_world = place_world.copy()
    preplace_world[2] += STANDOFF_Z
    base_world, base_quat = prim_world_pose(base_path)
    quaternion = quat_multiply(quat_inverse(base_quat), TOP_DOWN_WXYZ)
    quaternion /= np.linalg.norm(quaternion)
    goals = []
    for point in (preplace_world, place_world):
        goals.append(
            ApproachGoal(
                position=world_to_base(point.reshape(1, 3), base_world, base_quat)[0],
                quaternion=quaternion.copy(),
                center_world=center_world.copy(),
                top_z=top_z,
            )
        )
    return goals[0], goals[1]


def _is_table_destination(destination) -> bool:
    from pseudo_vla.scene import surface_location

    return destination.kind == "at" and destination.target == surface_location()


def _needs_approach_transit(scene, moving_name: str) -> bool:
    """Route the approach above the block when a nearby measured block is taller."""
    observed = _observed(scene)
    moving = observed.get(moving_name)
    if moving is None:
        return False
    for name, other in observed.items():
        if name == moving_name:
            continue
        if float(np.linalg.norm(other.center_world[:2] - moving.center_world[:2])) > APPROACH_TRANSIT_RADIUS:
            continue
        if other.top_z > moving.top_z + APPROACH_TRANSIT_HEIGHT:
            return True
    return False


def _table_slot_is_clear(scene, moving_name: str) -> bool:
    homes = getattr(scene, "home_xy", {})
    target_xy = homes.get(moving_name)
    moving = _observed(scene).get(moving_name)
    if target_xy is None or moving is None:
        return False
    target_xy = np.asarray(target_xy, dtype=np.float64)
    for other_name, other in _observed(scene).items():
        if other_name == moving_name:
            continue
        minimum_clearance = (moving.dims[0] + other.dims[0]) / 2.0 + 0.01
        if np.linalg.norm(other.center_world[:2] - target_xy) < minimum_clearance:
            return False
    return True


def _relative_slot_is_clear(scene, moving_name: str, destination) -> bool:
    _, place_goal = _relative_placement_goals(
        scene, moving_name, destination, scene.base_path
    )
    target_center = place_goal.center_world
    moving = _observed(scene).get(moving_name)
    if moving is None:
        return False
    target_name = destination.target
    for other_name, other in _observed(scene).items():
        if other_name in {moving_name, target_name}:
            continue
        overlaps = (
            abs(float(target_center[0] - other.center_world[0]))
            < (moving.dims[0] + other.dims[0]) / 2.0 + 0.005
            and abs(float(target_center[1] - other.center_world[1]))
            < (moving.dims[1] + other.dims[1]) / 2.0 + 0.005
            and abs(float(target_center[2] - other.center_world[2]))
            < (moving.dims[2] + other.dims[2]) / 2.0
        )
        if overlaps:
            return False
    return True


def _prompt_to_sam(
    perception,
    rgb: np.ndarray,
    prompt: str,
    say,
    *,
    grounding_prompt: str | None = None,
) -> tuple[tuple[float, float] | None, tuple[float, float, float, float] | None]:
    numbers = _numbers(prompt)
    if numbers is not None and len(numbers) == 2:
        say(f"SAM point {numbers[0]:.1f}, {numbers[1]:.1f}")
        return (numbers[0], numbers[1]), None
    if numbers is not None and len(numbers) == 4:
        say(f"SAM box {numbers}")
        return None, (numbers[0], numbers[1], numbers[2], numbers[3])
    if grounding_prompt is None:
        from pseudo_vla.stacking_llm import StackingModelError, normalize_object_prompt

        try:
            grounding_prompt = normalize_object_prompt(prompt)
        except (StackingModelError, ValueError) as error:
            say(f"could not interpret object prompt with Llama: {error}")
            return None, None
    say(f'grounding "{prompt}" as "{grounding_prompt}"')
    found = perception.ground(rgb, grounding_prompt)
    if found is None:
        tracked_group = _tracked_prompt_group(perception, grounding_prompt)
        if tracked_group is not None:
            group_name, point = tracked_group
            say(
                f"grounding found no match; reusing active CoTracker points "
                f"for {group_name}"
            )
            return point, None
        say(f"nothing in the environment camera matched \"{prompt}\"")
        return None, None
    box, score = found
    say(f"SAM box {[round(value, 1) for value in box]} score {score:.2f}")
    return None, box


def _tracked_prompt_group(perception, prompt: str):
    prompt_words = prompt.lower().replace("_", " ").replace("-", " ").split()
    query_groups = getattr(perception, "query_groups", {})
    query_xy = getattr(perception, "query_xy", None)
    if query_xy is None:
        return None
    for group_name, point_slice in query_groups.items():
        group_words = group_name.lower().replace("_", " ").replace("-", " ").split()
        if not group_words or not _contains_word_sequence(prompt_words, group_words):
            continue
        points = query_xy[point_slice]
        if len(points):
            center = np.median(points, axis=0)
            return group_name, (float(center[0]), float(center[1]))
    return None


def _contains_word_sequence(words: list[str], sequence: list[str]) -> bool:
    return any(words[index : index + len(sequence)] == sequence for index in range(len(words) - len(sequence) + 1))


def _confirm_placement(scene, perception, moving_name: str, destination, say) -> bool:
    """Require a fresh environment-camera depth pose at the requested destination."""
    scene.step()
    rgb, depth, camera_params = scene.read_sensors()
    measured = _measure_frame(perception, rgb, depth, camera_params, scene.camera_path, say)
    _store_observed(scene, measured)
    reason = _placement_failure(
        measured,
        moving_name,
        destination,
        getattr(scene, "home_xy", {}) or {},
    )
    if reason is not None:
        say(reason)
        return False
    return True


def _placement_failure(measured, moving_name: str, destination, home_xy) -> str | None:
    moving = measured.get(moving_name)
    if moving is None:
        return f"environment camera could not find {moving_name} after placement"
    from pseudo_vla.scene import surface_location

    if destination.kind == "at" and destination.target == surface_location():
        home = home_xy.get(moving_name)
        if home is None:
            return f"environment camera has no home position for {moving_name}"
        if float(np.linalg.norm(moving.center_world[:2] - np.asarray(home, dtype=np.float64))) > 0.06:
            return f"{moving_name} is not at {destination.target}"
        return None
    target = measured.get(destination.target)
    if target is None:
        return f"environment camera could not find {destination.target} after placement"
    if destination.kind == "on":
        xy = float(np.linalg.norm(moving.center_world[:2] - target.center_world[:2]))
        dz = float(moving.center_world[2] - target.center_world[2])
        if xy <= 0.04 and 0.02 <= dz <= 0.10:
            return None
        return f"{moving_name} is not on {destination.target}"
    gap = (float(moving.dims[0]) + float(target.dims[0])) / 2.0
    if destination.kind == "left_of" and float(moving.center_world[0]) < float(target.center_world[0]) - gap * 0.5:
        return None
    if destination.kind == "right_of" and float(moving.center_world[0]) > float(target.center_world[0]) + gap * 0.5:
        return None
    if destination.kind in {"left_of", "right_of"}:
        return f"{moving_name} is not {destination.kind} {destination.target}"
    return f"could not confirm {moving_name} at {destination.kind} {destination.target}"


def _seed_placed_cube_tracking(scene, perception, object_name: str, say, view) -> None:
    scene.step()
    rgb, depth, _camera_params = scene.read_sensors()
    _point, box = _visual_object_prompt(perception, rgb, object_name, say)
    if box is None:
        say(f"environment camera could not reseed tracking for {object_name}")
        return
    perception.segment_groups(rgb, {object_name: (None, box)})
    say(f"dropped stale CoTracker points; tracking the placed {object_name} cube")
    _update_camera_views(scene, view, rgb, depth, perception.mask, perception.query_xy)


def _ooal_grasp_point(scene, say, affordance: str | None) -> np.ndarray | None:
    """Unproject the matched affordance peak from the current wrist frame."""
    from pseudo_vla.affordance import affordance_pixel, affordance_status

    if not affordance:
        return None
    rgb, depth, camera_params = scene.read_wrist_sensors()
    if rgb.size == 0 or depth.size == 0:
        say("OOAL skipped because the wrist camera has no RGB-D frame")
        return None
    pixel = affordance_pixel(rgb, affordance)
    if pixel is None:
        detail = affordance_status()
        say(f"OOAL {affordance} map unavailable: {detail}")
        return None
    point = _unproject_grasp_pixel(pixel, depth, camera_params, scene.wrist_camera_path)
    if point is None:
        say(f"OOAL {affordance} pixel {[round(value, 1) for value in pixel]} has no wrist depth")
        return None
    say(
        f"OOAL {affordance} point {point.round(3).tolist()} "
        f"at wrist pixel {[round(value, 1) for value in pixel]}"
    )
    return point


def _unproject_grasp_pixel(pixel, depth, camera_params, camera_path) -> np.ndarray | None:
    from pseudo_vla.geometry import unproject_tracks

    u, v = float(pixel[0]), float(pixel[1])
    peak = unproject_tracks(
        np.array([[u, v]], dtype=np.float32),
        np.array([True]),
        depth,
        camera_params,
        camera_path,
    )
    samples = [(u + dx, v + dy) for dy in range(-3, 4) for dx in range(-3, 4)]
    cloud = unproject_tracks(
        np.asarray(samples, dtype=np.float32),
        np.ones(len(samples), dtype=bool),
        depth,
        camera_params,
        camera_path,
    )
    if peak.shape[0] == 0:
        if cloud.shape[0] == 0:
            return None
        return np.median(cloud, axis=0)
    if cloud.shape[0] == 0:
        return peak[0]
    near = cloud[np.linalg.norm(cloud - peak[0], axis=1) <= 0.02]
    if near.shape[0] == 0:
        return peak[0]
    return np.median(near, axis=0)


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


def _refresh_object_measurements(scene, perception, servo, say) -> None:
    measured = _observe_objects(scene, perception, say)
    _remember_home(scene, measured)
    if not measured:
        say("environment camera did not measure any objects")
        return
    servo.update_block_positions(
        {item.spec_name: item.center_world for item in measured.values()},
        {item.spec_name: item.dims for item in measured.values()},
    )


def _observe_objects(scene, perception, say=None) -> dict[str, EnvObject]:
    scene.step()
    rgb, depth, camera_params = scene.read_sensors()
    measured = _measure_frame(perception, rgb, depth, camera_params, scene.camera_path, say)
    _store_observed(scene, measured)
    return _observed(scene)


def _measure_frame(perception, rgb, depth, camera_params, camera_path: str, say=None) -> dict[str, EnvObject]:
    from pseudo_vla.perception import mask_depth_points, pose_from_depth_points
    from pseudo_vla.scene import SCENE_OBJECTS, object_id, object_phrase

    if rgb.size == 0 or depth.size == 0:
        if say is not None:
            say("environment camera has no RGB-D frame")
        return {}
    measured: dict[str, EnvObject] = {}
    requests = []
    for spec in SCENE_OBJECTS:
        requests.append((object_id(spec), spec, object_phrase(spec)))
    located = perception.ground_all(rgb, [phrase for _object_id, _spec, phrase in requests])
    for object_id, spec, phrase in requests:
        found = located.get(phrase)
        if found is None:
            if say is not None:
                say(f"environment camera could not find {object_id}")
            continue
        box, _score = found
        try:
            mask = perception.segment_region(rgb, box=box)
        except RuntimeError as error:
            if say is not None:
                say(f"environment camera could not segment {object_id}: {error}")
            continue
        points = mask_depth_points(mask, depth, camera_params, camera_path)
        if points.shape[0] < 4:
            if say is not None:
                say(f"environment depth has only {points.shape[0]} points on {object_id}")
            continue
        center, top_z, dims = pose_from_depth_points(points)
        measured[object_id] = EnvObject(
            object_id=object_id,
            spec_name=spec["name"],
            center_world=center,
            top_z=top_z,
            dims=dims,
            box=tuple(float(value) for value in box),
        )
        if say is not None:
            say(
                f"env depth {object_id} center {center.round(3).tolist()} "
                f"top z {top_z:.3f}"
            )
    return measured


def _store_observed(scene, measured: dict[str, EnvObject]) -> None:
    current = dict(getattr(scene, "last_observed", {}) or {})
    current.update(measured)
    scene.last_observed = current


def _observed(scene) -> dict[str, EnvObject]:
    return getattr(scene, "last_observed", {}) or {}


def _remember_home(scene, measured: dict[str, EnvObject]) -> None:
    homes = getattr(scene, "home_xy", None)
    if homes is None:
        homes = {}
        scene.home_xy = homes
    for object_id, item in measured.items():
        homes.setdefault(object_id, item.center_world[:2].copy())


def _update_observed_from_points(scene, object_id: str | None, points: np.ndarray) -> None:
    from pseudo_vla.perception import pose_from_depth_points

    if object_id is None or points.shape[0] < 4:
        return
    current = _observed(scene).get(object_id)
    if current is None:
        return
    center, top_z, dims = pose_from_depth_points(points)
    scene.last_observed[object_id] = EnvObject(
        object_id=current.object_id,
        spec_name=current.spec_name,
        center_world=center,
        top_z=top_z,
        dims=dims,
        box=current.box,
    )


def _visual_object_prompt(perception, rgb: np.ndarray, object_name: str, say):
    from pseudo_vla.scene import object_phrase, spec_by_id

    spec = spec_by_id(object_name)
    phrase = object_phrase(spec) if spec is not None else f"{object_name.replace('_', ' ')} block"
    found = perception.ground(rgb, phrase)
    if found is None:
        say(f"environment camera could not find {object_name}")
        return None, None
    box, score = found
    say(
        f"seeding {object_name} SAM mask from environment camera box "
        f"{[round(value, 1) for value in box]} score {score:.2f}"
    )
    return None, box


def _nearest_observed(point_world: np.ndarray, observed: dict[str, EnvObject]):
    nearest = None
    nearest_error = float("inf")
    for item in observed.values():
        error = float(np.linalg.norm(point_world - item.center_world))
        if error < nearest_error:
            nearest = item
            nearest_error = error
    if nearest is None or nearest_error > 0.08:
        return None, nearest_error
    return nearest, nearest_error
