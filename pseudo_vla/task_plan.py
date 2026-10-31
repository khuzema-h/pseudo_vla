"""Validated task plans for pick-and-place operations in the current scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pseudo_vla.blocks import BLOCK_COLORS

DestinationKind = Literal["table", "on", "left_of", "right_of", "inside"]
TaskIntent = Literal["find", "pick_place"]
SCENE_LOCATIONS = ("table",)
KNOWN_CONTAINERS = ("box", "shelf")


class TaskPlanError(ValueError):
    """Raised when a proposed task plan is invalid or unsupported by the scene."""


@dataclass(frozen=True)
class Destination:
    kind: DestinationKind
    target: str | None = None


@dataclass(frozen=True)
class PickPlaceAction:
    object_id: str
    destination: Destination


@dataclass(frozen=True)
class TaskPlan:
    intent: TaskIntent
    query: str | None = None
    actions: tuple[PickPlaceAction, ...] = ()


def validate_task_plan(
    intent: object,
    query: object,
    actions: object,
    *,
    available_objects: tuple[str, ...] = BLOCK_COLORS,
    available_locations: tuple[str, ...] = SCENE_LOCATIONS,
) -> TaskPlan:
    """Validate all source and destination references before any motion is planned."""
    if intent == "find":
        if not isinstance(query, str) or not query.strip() or len(query.strip()) > 200:
            raise TaskPlanError("find requires a non-empty visual query of at most 200 characters")
        if actions != []:
            raise TaskPlanError("find plans must not contain pick-and-place actions")
        return TaskPlan(intent="find", query=query.strip())

    if intent != "pick_place":
        raise TaskPlanError(f"unsupported task intent {intent!r}")
    if not isinstance(query, str) or len(query.strip()) > 200:
        raise TaskPlanError("pick_place query must be a string of at most 200 characters")
    if not isinstance(actions, list) or not 1 <= len(actions) <= 6:
        raise TaskPlanError("pick_place plans must contain between one and six actions")

    validated: list[PickPlaceAction] = []
    moved_objects: set[str] = set()
    on_edges: dict[str, str] = {}
    for index, action in enumerate(actions):
        if not isinstance(action, dict) or set(action) != {"object", "destination"}:
            raise TaskPlanError(
                f"action {index + 1} must contain only object and destination"
            )
        object_id = _identifier(action["object"], "object")
        if object_id not in available_objects:
            raise TaskPlanError(f"object {object_id!r} is not available in this scene")
        if object_id in moved_objects:
            raise TaskPlanError(f"object {object_id!r} is moved more than once in one plan")
        moved_objects.add(object_id)

        raw_destination = action["destination"]
        if not isinstance(raw_destination, dict) or set(raw_destination) != {"kind", "target"}:
            raise TaskPlanError(
                f"action {index + 1} destination must contain only kind and target"
            )
        kind = raw_destination["kind"]
        target_value = raw_destination["target"]
        if kind == "table":
            if target_value != "":
                raise TaskPlanError("table destinations must have an empty target")
            destination = Destination(kind="table")
        elif kind in {"on", "left_of", "right_of", "inside"}:
            target = _identifier(target_value, "destination target")
            if target == object_id:
                raise TaskPlanError("an object cannot be placed relative to itself")
            if kind == "inside":
                if target not in KNOWN_CONTAINERS:
                    raise TaskPlanError(
                        f"inside destinations must name a supported container; got {target!r}"
                    )
                if target not in available_locations:
                    raise TaskPlanError(
                        f"destination {target!r} is not present in the current scene"
                    )
            elif target not in available_objects:
                raise TaskPlanError(
                    f"{kind} destinations must name an object in the current scene"
                )
            if kind == "on":
                on_edges[object_id] = target
            destination = Destination(kind=kind, target=target)
        else:
            raise TaskPlanError(f"unsupported destination kind {kind!r}")
        validated.append(PickPlaceAction(object_id=object_id, destination=destination))

    for start in on_edges:
        visited: set[str] = set()
        current = start
        while current in on_edges:
            if current in visited:
                raise TaskPlanError("stack destination relations contain a cycle")
            visited.add(current)
            current = on_edges[current]

    return TaskPlan(
        intent="pick_place",
        query=query.strip() or None,
        actions=tuple(validated),
    )


def unstack_actions(scene_state: dict[str, object] | None) -> list[dict[str, object]]:
    """Derive a safe top-down unstack sequence from observed support relations."""
    state = scene_state or {}
    stacked_on = state.get("stacked_on")
    if not isinstance(stacked_on, dict) or not stacked_on:
        raise TaskPlanError("no stacked blocks were observed to unstack")
    if any(
        not isinstance(upper, str)
        or upper not in BLOCK_COLORS
        or not isinstance(lower, str)
        or lower not in BLOCK_COLORS
        or upper == lower
        for upper, lower in stacked_on.items()
    ):
        raise TaskPlanError("observed stack relations contain an unknown block")
    if len(set(stacked_on.values())) != len(stacked_on.values()):
        raise TaskPlanError("observed arrangement is branching; unstack order is ambiguous")

    tops = set(stacked_on) - set(stacked_on.values())
    if len(tops) != 1:
        raise TaskPlanError("observed arrangement does not contain one unambiguous tower")
    actions = []
    current = next(iter(tops))
    visited: set[str] = set()
    while current in stacked_on:
        if current in visited:
            raise TaskPlanError("observed stack relations contain a cycle")
        visited.add(current)
        actions.append(
            {"object": current, "destination": {"kind": "table", "target": ""}}
        )
        current = stacked_on[current]
    if visited != set(stacked_on):
        raise TaskPlanError("observed arrangement contains more than one tower")
    return actions


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 40:
        raise TaskPlanError(f"{label} must be a non-empty identifier")
    normalized = value.strip().lower().replace(" ", "_").removesuffix("_block").removesuffix("_cube")
    return normalized
