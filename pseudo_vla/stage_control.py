"""Stage-level instruction parsing, controller state, and dependency logic."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from pseudo_vla.task_plan import Destination


@dataclass(frozen=True)
class StageRequest:
    stage: str  # "approach", "grasp", "lift", "place", "release", "retreat"
    object_id: str | None = None
    destination: Destination | None = None
    instruction: str = ""


class ControllerStageState:
    def __init__(self) -> None:
        self.held_object: str | None = None
        self.stage: str = "home"  # "home", "approached", "grasped", "lifted", "placed", "released"
        self.target_object: str | None = None
        self.context: dict[str, object] | None = None

    def reset(self) -> None:
        self.held_object = None
        self.stage = "home"
        self.target_object = None
        self.context = None


def parse_stage_instruction(
    instruction: str,
    available_objects: tuple[str, ...],
    available_locations: tuple[str, ...] = (),
) -> StageRequest | None:
    """Parse single-stage commands or return None if it is a full multi-action macro or unparsed."""
    raw = instruction.strip().lower()
    if not raw:
        return None

    # Check for release / open gripper and common synonyms
    if raw in {"release", "open gripper", "open the gripper", "release gripper", "drop"}:
        return StageRequest(stage="release", instruction=instruction)
    put_down = re.fullmatch(r"(?:put|place|set)\s+(?:the\s+)?(.+?)\s+down", raw)
    drop_object = re.fullmatch(r"drop\s+(?:the\s+)?(.+)", raw)
    release_match = put_down or drop_object
    if release_match is not None:
        object_text = release_match.group(1)
        object_id = next(
            (
                obj
                for obj in available_objects
                if re.search(rf"\b{re.escape(obj)}\b", object_text)
            ),
            None,
        )
        return StageRequest(stage="release", object_id=object_id, instruction=instruction)

    # Check for retreat
    if raw in {"retreat", "pull back", "back up", "retreat gripper"}:
        return StageRequest(stage="retreat", instruction=instruction)

    # If the instruction contains both a pick/hold and a place destination,
    # let the standard pick-and-place pipeline handle it as a macro.
    has_pick_verb = any(word in raw for word in ("pick", "hold", "take", "grab", "lift"))
    has_place_verb = any(word in raw for word in ("place", "put", "set", "deposit", "stack"))
    destination = _extract_destination(raw, available_objects, available_locations)

    if has_pick_verb and (has_place_verb or (destination is not None and " and " in raw)):
        return None

    # Match target object from available objects, sorted by position in text
    dest_target = destination.target if destination is not None else None
    candidate_matches = []
    for obj in available_objects:
        for m in re.finditer(rf"\b{re.escape(obj)}\b", raw):
            candidate_matches.append((m.start(), obj))

    candidate_matches.sort()
    matched_object = None
    for _, obj in candidate_matches:
        if obj != dest_target:
            matched_object = obj
            break
    if matched_object is None and candidate_matches and dest_target is None:
        matched_object = candidate_matches[0][1]

    # 1. Approach
    if re.search(r"\b(approach|hover|move to|go to|align with)\b", raw):
        if matched_object is not None:
            return StageRequest(stage="approach", object_id=matched_object, instruction=instruction)

    # 2. Lift / Pick
    if re.search(r"\b(lift|pick up|pick|raise)\b", raw):
        return StageRequest(stage="lift", object_id=matched_object, instruction=instruction)

    # 3. Grasp / Hold
    if re.search(r"\b(grasp|grab|grip|close gripper|hold)\b", raw):
        return StageRequest(stage="grasp", object_id=matched_object, instruction=instruction)

    # 4. Place
    if has_place_verb or destination is not None:
        if destination is not None:
            return StageRequest(
                stage="place",
                object_id=matched_object,
                destination=destination,
                instruction=instruction,
            )

    return None


def _extract_destination(
    raw: str,
    available_objects: tuple[str, ...],
    available_locations: tuple[str, ...],
) -> Destination | None:
    # Check "at" / "on" location
    for loc in available_locations:
        if re.search(rf"\b(at|on|onto)\s+(the\s+)?{re.escape(loc)}\b", raw):
            return Destination(kind="at", target=loc)

    # Check relative slots: left_of / right_of
    for obj in available_objects:
        if re.search(rf"\b(to\s+the\s+left\s+of|left\s+of)\s+(the\s+)?{re.escape(obj)}\b", raw):
            return Destination(kind="left_of", target=obj)
        if re.search(rf"\b(to\s+the\s+right\s+of|right\s+of)\s+(the\s+)?{re.escape(obj)}\b", raw):
            return Destination(kind="right_of", target=obj)

    # Check "on" object
    for obj in available_objects:
        if re.search(rf"\b(on|onto)\s+(the\s+)?{re.escape(obj)}\b", raw):
            return Destination(kind="on", target=obj)

    return None


def plan_stage_prerequisites(
    request: StageRequest,
    state: ControllerStageState,
) -> tuple[list[str], str | None]:
    """Return the list of prerequisite stages needed before executing `request.stage`.

    Returns:
        (prerequisite_stages, error_message)
    """
    stage = request.stage

    if stage == "approach":
        target = request.object_id
        if state.held_object and state.held_object != target:
            return [], f"cannot approach {target} because {state.held_object} is already held"
        if state.stage == "approached" and state.target_object == target:
            return [], None
        return [], None

    if stage == "grasp":
        target = request.object_id or state.target_object
        if not target:
            return [], "no target object specified to grasp"
        if state.held_object and state.held_object != target:
            return [], f"cannot grasp {target} because {state.held_object} is already held"
        if state.stage in ("grasped", "lifted") and state.held_object == target:
            return [], None
        if state.stage == "approached" and state.target_object == target:
            return [], None
        return ["approach"], None

    if stage == "lift":
        target = request.object_id or state.held_object or state.target_object
        if not target:
            return [], "no target object specified to lift"
        if state.held_object and state.held_object != target:
            return [], f"cannot lift {target} because {state.held_object} is already held"
        if state.stage == "lifted" and state.held_object == target:
            return [], None
        if state.stage == "grasped" and state.held_object == target:
            return [], None
        if state.stage == "approached" and state.target_object == target:
            return ["grasp"], None
        return ["approach", "grasp"], None

    if stage == "place":
        target = request.object_id or state.held_object or state.target_object
        if not target:
            return [], "no object specified to place"
        if not request.destination:
            return [], "no destination specified for placement"
        if state.stage == "lifted" and state.held_object == target:
            return [], None
        if state.stage == "grasped" and state.held_object == target:
            return ["lift"], None
        if state.stage == "approached" and state.target_object == target:
            return ["grasp", "lift"], None
        return ["approach", "grasp", "lift"], None

    if stage == "release":
        if request.object_id and not state.held_object:
            return [], f"cannot release {request.object_id} because no object is held"
        if request.object_id and state.held_object != request.object_id:
            return [], f"cannot release {request.object_id} because {state.held_object} is held"
        return [], None

    if stage == "retreat":
        return [], None

    return [], f"unsupported stage {stage!r}"
