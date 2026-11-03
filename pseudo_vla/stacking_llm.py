"""Local Ollama prompt parsing and task-event decisions for cube stacking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pseudo_vla.blocks import BLOCK_COLORS
from pseudo_vla.tev1_client import Tev1Error, ask_choice

VALID_COLORS = BLOCK_COLORS
_COLOR_PATTERN = re.compile(
    rf"^({'|'.join(VALID_COLORS)})(?:\s+(?:cube|block)s?)?$",
    re.IGNORECASE,
)
STACK_ACTION_EVENTS = ("approach", "grasp", "lift", "place", "verify_placement")
StackingModelError = Tev1Error


@dataclass(frozen=True)
class UserPrompt:
    intent: Literal["find_object", "stack_blocks"]
    query: str | None = None
    stacking_relations: list["StackingRelation"] | None = None


@dataclass(frozen=True)
class StackingRelation:
    support: str
    moving: str


def parse_stacking_prompt(user_prompt: str) -> list[str]:
    """Parse a natural-language request into a bottom-to-top order."""
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("A non-empty stacking prompt is required")
    mentioned_colors = {
        color for color in VALID_COLORS if re.search(rf"\b{color}\b", user_prompt, re.IGNORECASE)
    }
    if mentioned_colors != set(VALID_COLORS):
        raise ValueError("The prompt must specify red, green, and blue in bottom-to-top order")

    state: dict[str, object] = {
        "instruction": user_prompt.strip(),
        "task": "build_stack",
        "placed_bottom_to_top": [],
    }
    color_criteria = {
        color: f"Use {color} as the next block in the requested bottom-to-top order."
        for color in VALID_COLORS
    }
    base = ask_choice("stack_base", state, color_criteria)
    colors = [_normalize_color(base)]
    while len(colors) < len(VALID_COLORS):
        state["placed_bottom_to_top"] = list(colors)
        remaining = [color for color in VALID_COLORS if color not in colors]
        next_criteria = {
            color: f"Place {color} next, above {colors[-1]}."
            for color in remaining
        }
        next_criteria["none"] = "No remaining block matches the requested next stack position."
        next_color = ask_choice(
            "stack_next",
            state,
            next_criteria,
        )
        if next_color == "none":
            raise StackingModelError("Tev1 could not match the next requested stack block")
        colors.append(_normalize_color(next_color))
    _validate_order(colors)
    return colors


def interpret_user_prompt(user_prompt: str) -> UserPrompt:
    """Parse a block-scene request through the shared Tev1 task router."""
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("A non-empty prompt is required")
    from pseudo_vla.task_llm import interpret_task_prompt

    plan = interpret_task_prompt(
        user_prompt,
        {"objects": list(VALID_COLORS), "locations": ["table"], "containers": []},
    )
    if plan.intent == "find":
        return UserPrompt(intent="find_object", query=plan.query)
    raw_relations = []
    for action in plan.actions:
        if (
            action.object_id not in VALID_COLORS
            or action.destination.kind != "on"
            or action.destination.target not in VALID_COLORS
        ):
            raise StackingModelError(
                "The block-scene stack path supports only placements of one listed block on another"
            )
        raw_relations.append(
            {"upper": action.object_id, "lower": action.destination.target}
        )
    try:
        ordered_relations = _ordered_stacking_relations(raw_relations)
    except ValueError as error:
        raise StackingModelError(f"Could not validate Tev1 stacking relations: {error}") from error
    return UserPrompt(intent="stack_blocks", stacking_relations=ordered_relations)


def _ordered_stacking_relations(relations: list[object]) -> list[StackingRelation]:
    if not 1 <= len(relations) <= len(VALID_COLORS) - 1:
        raise ValueError("provide one or two explicit upper/lower relations")

    above: dict[str, str] = {}
    below: dict[str, str] = {}
    for relation in relations:
        if not isinstance(relation, dict) or set(relation) != {"upper", "lower"}:
            raise ValueError("each relation must contain only upper and lower colors")
        upper = _normalize_color(relation["upper"])
        lower = _normalize_color(relation["lower"])
        if upper == lower:
            raise ValueError("a block cannot be placed on itself")
        if lower in above or upper in below:
            raise ValueError("relations must form one unambiguous three-block chain")
        above[lower] = upper
        below[upper] = lower

    bottoms = [color for color in above if color not in below]
    if len(bottoms) != 1:
        raise ValueError("relations must form one acyclic chain")
    order = [bottoms[0]]
    ordered_pairs = []
    while order[-1] in above:
        next_color = above[order[-1]]
        if next_color in order:
            raise ValueError("stacking relations contain a cycle")
        ordered_pairs.append(StackingRelation(support=order[-1], moving=next_color))
        order.append(next_color)
    if len(ordered_pairs) != len(relations):
        raise ValueError("relations must form one connected chain")
    return ordered_pairs


def normalize_object_prompt(user_prompt: str) -> str:
    """Use Tev1 to map a free-text find request to a listed block identifier."""
    parsed = interpret_user_prompt(user_prompt)
    if parsed.intent != "find_object" or parsed.query is None:
        raise ValueError("The prompt requests stacking, not an object search")
    return parsed.query


def choose_stack_event(
    goal_order: list[str],
    placed: list[str],
    current_support: str | None,
    last_event_result: dict[str, object] | None,
    *,
    expected_event: str | None = None,
    current_pair: dict[str, str] | None = None,
) -> str:
    """Ask Tev1 to choose the next legal high-level stacking event."""
    _validate_order(goal_order)
    if placed != goal_order[: len(placed)]:
        raise ValueError("Placed cubes must be a prefix of the requested stack order")
    if current_support != (placed[-1] if placed else None):
        raise ValueError("Current support does not match the placed stack")

    if expected_event is not None:
        if expected_event not in STACK_ACTION_EVENTS:
            raise ValueError(f"Unsupported stack action event {expected_event!r}")
        if len(placed) >= len(goal_order) or goal_order[len(placed)] != (current_pair or {}).get("moving"):
            raise ValueError("Current pair does not match the next requested placement")
        if current_support != (current_pair or {}).get("support"):
            raise ValueError("Current pair support does not match the placed stack")
        choices = {
            expected_event: (
                f"Authorize {expected_event.replace('_', ' ')} for moving "
                f"{current_pair['moving']} onto {current_pair['support']}."
            ),
            "stop": "Stop at the current safe boundary.",
        }
        phase = f"authorize_{expected_event}"
    elif not placed:
        choices = {
            "establish_base": f"Establish {goal_order[0]} as the table-supported base.",
            "stop": "Stop before changing the scene.",
        }
        phase = "choose_base_event"
    elif len(placed) < len(goal_order):
        choices = {
            "stack_next": (
                f"Place {goal_order[len(placed)]} on the current support {current_support}."
            ),
            "stop": "Stop at the current safe boundary.",
        }
        phase = "choose_placement_event"
    else:
        choices = {
            "finish": "The requested stack is complete; finish the workflow.",
            "stop": "Stop at the current safe boundary.",
        }
        phase = "choose_finish_event"

    state = {
        "task": "validated_block_stacking",
        "phase": phase,
        "goal_order_bottom_to_top": goal_order,
        "placed_bottom_to_top": placed,
        "current_support": current_support,
        "next_cube": goal_order[len(placed)] if len(placed) < len(goal_order) else None,
        "last_event_result": last_event_result,
        "current_pair": current_pair,
    }
    return ask_choice("stack_event", state, choices)


def choose_stack_action(
    support: str,
    moving: str,
    expected_event: str,
    last_event_result: dict[str, object] | None,
) -> str:
    """Ask Tev1 to authorize one stage of a specifically requested block pair."""
    if support not in VALID_COLORS or moving not in VALID_COLORS or support == moving:
        raise ValueError("A stack action requires two distinct supported block colors")
    if expected_event not in STACK_ACTION_EVENTS:
        raise ValueError(f"Unsupported stack action event {expected_event!r}")

    choices = {
        expected_event: (
            f"Authorize {expected_event.replace('_', ' ')} for moving {moving} onto {support}."
        ),
        "stop": "Stop at the current safe boundary.",
    }
    state = {
        "task": "requested_pair_stacking",
        "phase": f"authorize_{expected_event}",
        "requested_pair": {"support": support, "moving": moving},
        "last_event_result": last_event_result,
    }
    return ask_choice("stack_event", state, choices)


def _validate_order(sequence: list[str]) -> None:
    if (
        not isinstance(sequence, list)
        or len(sequence) != len(VALID_COLORS)
        or any(color not in VALID_COLORS for color in sequence)
        or set(sequence) != set(VALID_COLORS)
    ):
        raise ValueError("Stack order must contain red, green, and blue exactly once")


def _normalize_color(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Invalid block color {value!r}")
    normalized = value.strip().lower().rstrip(".,;:")
    match = _COLOR_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(f"Invalid block color {value!r}; expected red, green, or blue")
    return match.group(1).lower()
