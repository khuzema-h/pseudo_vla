"""Validated pick-and-place plans.

The plan vocabulary is scene-neutral. Callers supply the object, location, and
container identifiers that exist when the plan is checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DestinationKind = Literal["at", "on", "left_of", "right_of", "inside"]
TaskIntent = Literal["find", "pick_place"]
DESTINATION_KINDS = ("at", "on", "left_of", "right_of", "inside")
RELATIVE_DESTINATIONS = ("on", "left_of", "right_of")


class TaskPlanError(ValueError):
    """Raised when a proposed task plan is invalid for the supplied scene catalog."""


@dataclass(frozen=True)
class Destination:
    kind: DestinationKind
    target: str


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
    available_objects: tuple[str, ...] = (),
    available_locations: tuple[str, ...] = (),
    available_containers: tuple[str, ...] = (),
) -> TaskPlan:
    """Validate source and destination identifiers against the supplied catalog."""
    objects = _catalog(available_objects, "available object")
    locations = _catalog(available_locations, "available location")
    containers = _catalog(available_containers, "available container")

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
        if object_id not in objects:
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
        if kind not in DESTINATION_KINDS:
            raise TaskPlanError(f"unsupported destination kind {kind!r}")
        target = _identifier(raw_destination["target"], "destination target")
        if kind == "at" and target in objects and target not in locations:
            # A named object is a support, not a location. "at the plate" means on it.
            kind = "on"
        if kind == "at":
            if target not in locations:
                raise TaskPlanError(f"location {target!r} is not present in the current scene")
        elif kind == "inside":
            if target not in containers:
                raise TaskPlanError(
                    f"destination {target!r} is not present in the current scene"
                )
        elif target not in objects:
            raise TaskPlanError(
                f"{kind} destinations must name an object in the current scene"
            )
        if kind in RELATIVE_DESTINATIONS and target == object_id:
            raise TaskPlanError("an object cannot be placed relative to itself")
        if kind == "on":
            on_edges[object_id] = target
        validated.append(PickPlaceAction(object_id=object_id, destination=Destination(kind, target)))

    for start in on_edges:
        visited: set[str] = set()
        current = start
        while current in on_edges:
            if current in visited:
                raise TaskPlanError("support destination relations contain a cycle")
            visited.add(current)
            current = on_edges[current]

    return TaskPlan(
        intent="pick_place",
        query=query.strip() or None,
        actions=tuple(validated),
    )


def _catalog(values: tuple[str, ...] | list[str], label: str) -> set[str]:
    if not isinstance(values, (tuple, list)):
        raise TaskPlanError(f"{label} catalog must be a sequence of identifiers")
    return {_identifier(value, label) for value in values}


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 40:
        raise TaskPlanError(f"{label} must be a non-empty identifier")
    return value.strip().lower().replace(" ", "_")
