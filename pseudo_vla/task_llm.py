"""Tev1 task routing and callback-informed task-event decisions."""

from __future__ import annotations

from pseudo_vla.task_plan import (
    DESTINATION_KINDS,
    PickPlaceAction,
    TaskPlan,
    validate_task_plan,
)
from pseudo_vla.tev1_client import Tev1Error, ask_boolean, ask_choice

TASK_EVENTS = ("find", "plan", "approach", "grasp", "lift", "place", "verify_placement")
MAX_PLAN_ACTIONS = 6
_MAX_CHOICE_OPTIONS = 24


def interpret_task_prompt(user_prompt: str, scene_state: dict[str, object] | None = None) -> TaskPlan:
    """Route a natural-language request through choices bounded by the current scene."""
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("A non-empty prompt is required")
    instruction = user_prompt.strip()
    state = scene_state or {}
    objects, locations, containers = _scene_catalog(state)
    if not objects:
        raise Tev1Error("Cannot route a task without objects in the current scene")

    decision_state = {
        "instruction": instruction,
        "scene": state,
        "phase": "select_task_intent",
    }
    intent = ask_choice(
        "task_intent",
        decision_state,
        {
            "find": "Find, locate, or identify one object without moving it.",
            "pick_place": "Move one or more listed objects to listed destinations.",
            "unsupported": "The requested operation is not supported by the available handlers.",
        },
    )
    if intent == "unsupported":
        raise Tev1Error("Tev1 classified the instruction as an unsupported task")
    if intent == "find":
        candidates = _option_criteria(objects, "Find the scene object")
        candidates["none"] = "No listed scene object matches the requested object."
        target = ask_choice(
            "find_target",
            {**decision_state, "phase": "select_find_target"},
            candidates,
        )
        if target == "none":
            raise Tev1Error("Tev1 could not match the find request to a scene object")
        try:
            return validate_task_plan(
                "find",
                target,
                [],
                available_objects=objects,
                available_locations=locations,
                available_containers=containers,
            )
        except ValueError as error:
            raise Tev1Error(f"Could not validate Tev1 find decision: {error}") from error

    actions: list[dict[str, object]] = []
    for index in range(MAX_PLAN_ACTIONS + 1):
        state_for_step = {
            **decision_state,
            "phase": "select_next_action",
            "actions_so_far": actions,
            "action_index": index,
        }
        more_actions = ask_boolean("plan_more", state_for_step)
        if not more_actions:
            break
        if index == MAX_PLAN_ACTIONS:
            raise Tev1Error(
                f"Tev1 requested more than the maximum of {MAX_PLAN_ACTIONS} actions"
            )

        moved = {
            action["object"]
            for action in actions
            if isinstance(action.get("object"), str)
        }
        available_to_move = tuple(object_id for object_id in objects if object_id not in moved)
        if not available_to_move:
            raise Tev1Error("Tev1 requested another action but no unmoved objects remain")
        moving_criteria = _option_criteria(available_to_move, "Move this scene object")
        moving_criteria["none"] = "No available object matches the requested action."
        moving = ask_choice(
            "moving_object",
            {**state_for_step, "phase": "select_moving_object"},
            moving_criteria,
        )
        if moving == "none":
            raise Tev1Error("Tev1 could not match an action to an available object")

        target_catalogs = {
            "at": locations,
            "inside": containers,
            "on": tuple(object_id for object_id in objects if object_id != moving),
            "left_of": tuple(object_id for object_id in objects if object_id != moving),
            "right_of": tuple(object_id for object_id in objects if object_id != moving),
        }
        kind_criteria = {
            kind: f"Use destination kind {kind}."
            for kind in DESTINATION_KINDS
            if target_catalogs[kind]
        }
        if not kind_criteria:
            raise Tev1Error("No supported destination is available for this scene object")
        kind_criteria["unsupported"] = "The requested destination is not represented in this scene."
        destination_kind = ask_choice(
            "destination_kind",
            {**state_for_step, "phase": "select_destination_kind", "moving_object": moving},
            kind_criteria,
        )
        if destination_kind == "unsupported":
            raise Tev1Error("Tev1 selected a destination that is not supported in this scene")

        targets = target_catalogs[destination_kind]
        if len(targets) == 1:
            target = targets[0]
        else:
            target_criteria = _option_criteria(targets, "Use this listed destination")
            target_criteria["none"] = "No listed destination matches the user's request."
            target = ask_choice(
                "destination_target",
                {
                    **state_for_step,
                    "phase": "select_destination_target",
                    "moving_object": moving,
                    "destination_kind": destination_kind,
                },
                target_criteria,
            )
            if target == "none":
                raise Tev1Error("Tev1 could not match the destination to the current scene")
        actions.append(
            {
                "object": moving,
                "destination": {"kind": destination_kind, "target": target},
            }
        )
        try:
            validate_task_plan(
                "pick_place",
                "",
                actions,
                available_objects=objects,
                available_locations=locations,
                available_containers=containers,
            )
        except ValueError as error:
            raise Tev1Error(f"Tev1 proposed an invalid action: {error}") from error

    if not actions:
        raise Tev1Error("Tev1 did not select any supported pick-and-place actions")
    try:
        return validate_task_plan(
            "pick_place",
            "",
            actions,
            available_objects=objects,
            available_locations=locations,
            available_containers=containers,
        )
    except ValueError as error:
        raise Tev1Error(f"Could not validate Tev1 task plan: {error}") from error


def _scene_catalog(
    scene_state: dict[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Read object, location, and container identifiers from scene state."""
    raw_objects = scene_state.get("objects", scene_state.get("movable_objects", ()))
    return (
        _catalog_ids(raw_objects, nested_id=True),
        _catalog_ids(scene_state.get("locations", ())),
        _catalog_ids(scene_state.get("containers", ())),
    )


def _catalog_ids(values: object, *, nested_id: bool = False) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    identifiers: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip():
            identifiers.append(value.strip())
        elif nested_id and isinstance(value, dict) and isinstance(value.get("id"), str):
            identifiers.append(value["id"].strip())
    return tuple(dict.fromkeys(identifiers))


def _option_criteria(
    identifiers: tuple[str, ...],
    description: str,
) -> dict[str, str]:
    if len(identifiers) > _MAX_CHOICE_OPTIONS - 1:
        raise Tev1Error(
            f"Tev1 can compare at most {_MAX_CHOICE_OPTIONS - 1} scene candidates at once"
        )
    return {
        identifier: f"{description} {identifier}."
        for identifier in identifiers
    }


def choose_task_event(
    event: str,
    action: PickPlaceAction | None,
    action_index: int,
    last_event_result: dict[str, object] | None,
    plan: TaskPlan | None = None,
) -> str:
    """Ask Tev1 to authorize one expected task event at the current safe boundary."""
    if event not in TASK_EVENTS:
        raise ValueError(f"Unsupported task event {event!r}")
    if action_index < 0:
        raise ValueError("Action index must be non-negative")
    action_state = (
        {
            "object": action.object_id,
            "destination": {
                "kind": action.destination.kind,
                "target": action.destination.target,
            },
        }
        if action is not None
        else None
    )
    validated_plan = (
        [
            {
                "object": item.object_id,
                "destination": {
                    "kind": item.destination.kind,
                    "target": item.destination.target,
                },
            }
            for item in plan.actions
        ]
        if plan is not None and plan.intent == "pick_place"
        else []
    )
    choices = {
        event: f"Proceed with the expected {event} event for action {action_index + 1}.",
        "stop": "Stop before the next stage; do not dispatch another action.",
    }
    return ask_choice(
        "task_event",
        {
            "task": "validated_pick_and_place",
            "phase": f"authorize_{event}",
            "action_index": action_index,
            "action": action_state,
            "validated_plan": validated_plan,
            "last_event_result": last_event_result,
        },
        choices,
    )
