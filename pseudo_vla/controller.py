"""Robot controller for natural-language pick-and-place tasks in Isaac Sim.

The simulation stays open for natural-language instructions. Type `home` to
return the hand to its starting pose, or `quit` to close Kit.
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

from pseudo_vla.motion import (
    APPROACH_TOLERANCE,
    GRIPPER_SETTLE_STEPS,
    execute_pose as _execute_pose,
    gripper_place as _gripper_place,
    home as _home,
    open_gripper_for_next_command as _open_gripper_for_next_command,
    poll_command as _poll,
    settle_tracker as _settle_tracker,
    update_camera_views as _update_camera_views,
    update_tracker as _update_tracker,
)
from pseudo_vla.stacking_workflow import (
    run_prompted_stack_sequence,
    run_requested_stack_relations,
)
from pseudo_vla.task_workflow import run_task_plan

GRIPPER_FINGER_REACH = 0.11
LIFT_HEIGHT = 0.20
STACK_TRANSIT_CLEARANCE = 0.15
APPROACH_TRANSIT_RADIUS = 0.35
APPROACH_TRANSIT_HEIGHT = 0.02
QUIT_COMMANDS = {"quit", "exit", "q"}


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

        say(
            'Enter a pick-and-place instruction, such as "stack blue on red", '
            '"unstack the tower", or "put blue to the left of red".'
        )
        say("The current scene supports red, green, and blue blocks. Box and shelf destinations are not in this scene.")
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
        parsed = interpret_task_prompt(prompt, _task_scene_state(scene))
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
        if action.destination.kind == "table" and not _table_slot_is_clear(scene, action.object_id):
            say(
                f"could not execute plan: table destination for {action.object_id} "
                "is occupied"
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
        )

    complete = run_task_plan(
        parsed,
        say,
        lambda _query: False,
        execute_action,
    )
    say("task plan complete" if complete else "task plan stopped safely")
    return None


def _task_scene_state(scene) -> dict[str, object]:
    from pseudo_vla.perception import prim_world_pose
    from pseudo_vla.scene import SCENE_OBJECTS

    objects = []
    centers = {}
    for spec in SCENE_OBJECTS:
        center, _ = prim_world_pose(scene.object_paths[spec["name"]])
        object_id = spec["name"].removesuffix("_block")
        centers[object_id] = center
        objects.append(
            {
                "id": object_id,
                "position_world_m": [round(float(value), 4) for value in center],
            }
        )
    stacked_on = {}
    for object_id, center in centers.items():
        supports = [
            (float(candidate[2]), candidate_id)
            for candidate_id, candidate in centers.items()
            if candidate_id != object_id
            and 0.03 <= float(center[2] - candidate[2]) <= 0.08
            and float(np.linalg.norm(center[:2] - candidate[:2])) <= 0.035
        ]
        if supports:
            stacked_on[object_id] = max(supports)[1]
    return {
        "movable_objects": objects,
        "stacked_on": stacked_on,
        "destinations": ["table", "on", "left_of", "right_of"],
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
    from pseudo_vla.perception import ApproachGoal, STANDOFF_Z, prim_world_pose, save_overlay, world_to_base
    from pseudo_vla.scene import object_center, object_dims

    target_prompt = grounding_prompt or prompt
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
) -> bool:
    from pseudo_vla.task_plan import Destination
    from pseudo_vla.scene import SCENE_OBJECTS

    if destination is None:
        destination = Destination(kind="on", target=support_name)
    scene.step()
    rgb, depth, camera_params = scene.read_sensors()
    prompts = {}
    target_name = destination.target if destination.kind != "table" else None
    tracked_objects = []
    if target_name in {spec["name"].removesuffix("_block") for spec in SCENE_OBJECTS}:
        tracked_objects.append(("support", target_name))
    tracked_objects.append(("moving", moving_name))
    for group_name, object_name in tracked_objects:
        point, box = _projected_cube_prompt(scene, rgb, camera_params, object_name, say)
        if point is None and box is None:
            return False
        prompts[group_name] = (point, box)

    perception.segment_groups(rgb, prompts)
    _update_camera_views(scene, view, rgb, depth, perception.mask, perception.query_xy)
    if target_name is not None:
        say(
            f"tracking {target_name} target and {moving_name} cube with "
            f"{len(perception.query_xy)} CoTracker points"
        )
    else:
        say(f"tracking {moving_name} cube with {len(perception.query_xy)} CoTracker points")
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
) -> bool:
    from pseudo_vla.perception import (
        ApproachGoal,
        STANDOFF_Z,
        prim_world_pose,
        world_to_base,
    )
    from pseudo_vla.scene import SCENE_OBJECTS, object_dims

    specs = {spec["name"].removesuffix("_block"): spec for spec in SCENE_OBJECTS}
    if destination is None:
        from pseudo_vla.task_plan import Destination

        destination = Destination(kind="on", target=support_name)
    target_name = destination.target if destination.kind != "table" else None
    support_spec = specs.get(target_name) if target_name is not None else None
    moving_spec = specs[moving_name]
    moving_dims = object_dims(moving_spec)
    hover_goal = None
    transit = _needs_approach_transit(scene, moving_name)

    if authorize_event is not None and not authorize_event("approach"):
        return False
    destination_label = destination.kind
    if destination.target:
        destination_label = f"{destination.kind} {destination.target}"
    say(f"approaching {moving_name} for destination {destination_label}")
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
                plan_goal = goal
                if transit:
                    base_world, base_quat = prim_world_pose(scene.base_path)
                    plan_goal = _stack_transit_goal(goal, base_world, base_quat)
                servo.update_goal(plan_goal, scene.joint_positions())
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
        target_height = STANDOFF_Z + (STACK_TRANSIT_CLEARANCE if transit else 0.0)
        if step % 16 == 0:
            log(
                f"stack hover {moving_name}: horizontal error {horizontal:.3f} m, "
                f"height {height:.3f} m, target {target_height:.3f} m, "
                f"height error {abs(height - target_height):.3f} m"
            )
        if horizontal <= APPROACH_TOLERANCE and abs(height - target_height) <= APPROACH_TOLERANCE:
            if transit:
                transit = False
                servo.update_goal(hover_goal, scene.joint_positions())
                say(f"at safe approach height above {moving_name}; descending to the hover waypoint")
                continue
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
        say(f"pick-and-place interrupted by {pending}")
        commands.put(pending)
        return False
    if not reached:
        say(f"could not reach the wrist-refined hover pose for {moving_name}")
        return False
    if authorize_event is not None:
        authorize_event("approach", completed=True)

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

    if authorize_event is not None and not authorize_event("grasp"):
        return False
    servo.set_obstacle_enabled(moving_spec["name"], False)
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
        scene.set_gripper_closed(True)
        retained_groups = ("support",) if destination.kind == "on" else None
        _settle_tracker(
            scene,
            perception,
            GRIPPER_SETTLE_STEPS,
            view,
            retain_groups=retained_groups,
        )
        if authorize_event is not None:
            authorize_event("grasp", completed=True)

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
        if destination.kind == "table" and not _table_slot_is_clear(scene, moving_name):
            say(f"table destination for {moving_name} is occupied; not moving the block")
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
        _seed_placed_cube_tracking(scene, perception, moving_name, say, view)
        if authorize_event is not None and not authorize_event("verify_placement"):
            return False
        if authorize_event is not None:
            authorize_event("verify_placement", completed=True)
        say(f"placed {moving_name} at {destination_label}")
        return True
    finally:
        servo.set_obstacle_enabled(moving_spec["name"], True)
        centers = {
            spec["name"]: prim_world_pose(scene.object_paths[spec["name"]])[0]
            for spec in SCENE_OBJECTS
        }
        servo.update_block_positions(centers)


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
    from pseudo_vla.blocks import BLOCKS_BY_COLOR
    from pseudo_vla.perception import (
        ApproachGoal,
        STANDOFF_Z,
        TOP_DOWN_WXYZ,
        prim_world_pose,
        quat_inverse,
        quat_multiply,
        world_to_base,
    )
    from pseudo_vla.scene import TABLE_TOP_Z, object_dims, SCENE_OBJECTS

    specs = {spec["name"].removesuffix("_block"): spec for spec in SCENE_OBJECTS}
    moving_dims = object_dims(specs[moving_name])
    if destination.kind == "table":
        xy = BLOCKS_BY_COLOR[moving_name]["xy"]
        center_world = np.array(
            [xy[0], xy[1], TABLE_TOP_Z + moving_dims[2] / 2.0 + 0.002],
            dtype=np.float64,
        )
    else:
        target_name = destination.target
        if target_name is None or target_name not in specs:
            raise ValueError(f"Unresolved placement target {target_name!r}")
        target_center, _ = prim_world_pose(scene.object_paths[specs[target_name]["name"]])
        target_dims = object_dims(specs[target_name])
        center_world = target_center.copy()
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


def _needs_approach_transit(scene, moving_name: str) -> bool:
    """Route the approach above the block when a nearby block is taller."""
    from pseudo_vla.perception import prim_world_pose
    from pseudo_vla.scene import SCENE_OBJECTS, object_dims

    specs = {spec["name"].removesuffix("_block"): spec for spec in SCENE_OBJECTS}
    moving_center, _ = prim_world_pose(scene.object_paths[specs[moving_name]["name"]])
    moving_top = float(moving_center[2] + object_dims(specs[moving_name])[2] / 2.0)
    for name, spec in specs.items():
        if name == moving_name:
            continue
        center, _ = prim_world_pose(scene.object_paths[spec["name"]])
        if float(np.linalg.norm(center[:2] - moving_center[:2])) > APPROACH_TRANSIT_RADIUS:
            continue
        other_top = float(center[2] + object_dims(spec)[2] / 2.0)
        if other_top > moving_top + APPROACH_TRANSIT_HEIGHT:
            return True
    return False


def _table_slot_is_clear(scene, moving_name: str) -> bool:
    from pseudo_vla.blocks import BLOCKS_BY_COLOR
    from pseudo_vla.perception import prim_world_pose
    from pseudo_vla.scene import SCENE_OBJECTS, object_dims

    target_xy = np.asarray(BLOCKS_BY_COLOR[moving_name]["xy"], dtype=np.float64)
    moving_dims = object_dims(
        next(spec for spec in SCENE_OBJECTS if spec["name"] == f"{moving_name}_block")
    )
    for spec in SCENE_OBJECTS:
        other_name = spec["name"].removesuffix("_block")
        if other_name == moving_name:
            continue
        center, _ = prim_world_pose(scene.object_paths[spec["name"]])
        other_dims = object_dims(spec)
        minimum_clearance = (moving_dims[0] + other_dims[0]) / 2.0 + 0.01
        if np.linalg.norm(center[:2] - target_xy) < minimum_clearance:
            return False
    return True


def _relative_slot_is_clear(scene, moving_name: str, destination) -> bool:
    from pseudo_vla.perception import prim_world_pose
    from pseudo_vla.scene import SCENE_OBJECTS, object_dims

    _, place_goal = _relative_placement_goals(
        scene, moving_name, destination, scene.base_path
    )
    target_center = place_goal.center_world
    moving_spec = next(
        spec for spec in SCENE_OBJECTS if spec["name"] == f"{moving_name}_block"
    )
    moving_dims = object_dims(moving_spec)
    target_name = destination.target
    for spec in SCENE_OBJECTS:
        other_name = spec["name"].removesuffix("_block")
        if other_name in {moving_name, target_name}:
            continue
        center, _ = prim_world_pose(scene.object_paths[spec["name"]])
        other_dims = object_dims(spec)
        overlaps = (
            abs(float(target_center[0] - center[0]))
            < (moving_dims[0] + other_dims[0]) / 2.0 + 0.005
            and abs(float(target_center[1] - center[1]))
            < (moving_dims[1] + other_dims[1]) / 2.0 + 0.005
            and abs(float(target_center[2] - center[2]))
            < (moving_dims[2] + other_dims[2]) / 2.0
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
    perception.segment_groups(rgb, {object_name: (point, None)})
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
