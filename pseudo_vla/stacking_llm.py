"""Local Ollama prompt parsing and task-event decisions for cube stacking."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pseudo_vla.blocks import BLOCK_COLORS

VALID_COLORS = BLOCK_COLORS
OLLAMA_TIMEOUT = 120.0
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
_COLOR_PATTERN = re.compile(
    rf"^({'|'.join(VALID_COLORS)})(?:\s+(?:cube|block)s?)?$",
    re.IGNORECASE,
)
STACK_ACTION_EVENTS = ("approach", "grasp", "lift", "place", "verify_placement")


class StackingModelError(RuntimeError):
    """Raised when an Ollama request or model response cannot be used safely."""


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

    config = _load_prompt_config("llama_stacking.json")
    schema = config.get("response_schema")
    if not isinstance(schema, dict):
        raise StackingModelError("Llama prompt config must contain an object response_schema")
    schema_properties = schema.get("properties")
    sequence_schema = schema_properties.get("sequence") if isinstance(schema_properties, dict) else None
    sequence_items = sequence_schema.get("items") if isinstance(sequence_schema, dict) else None
    if not isinstance(sequence_items, dict) or sequence_items.get("enum") != list(VALID_COLORS):
        raise StackingModelError("Llama response_schema colors must match the supported block colors")
    response = _post_json(
        "/api/chat",
        {
            "model": _prompt_config_string(config, "model", "Llama"),
            "messages": [
                {
                    "role": "system",
                    "content": _prompt_config_string(config, "system_prompt", "Llama"),
                },
                {"role": "user", "content": user_prompt.strip()},
            ],
            "format": schema,
            "stream": False,
            "options": {"temperature": 0},
        },
    )
    message = response.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise StackingModelError("Llama response is missing message.content")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise StackingModelError(f"Llama returned invalid JSON: {content!r}") from error
    if not isinstance(parsed, dict) or set(parsed) != {"sequence"}:
        raise ValueError("Llama response must contain only a sequence field")

    sequence = parsed["sequence"]
    if not isinstance(sequence, list) or len(sequence) != len(VALID_COLORS):
        raise ValueError("The stack must contain red, green, and blue exactly once")
    colors = [_normalize_color(color) for color in sequence]
    if set(colors) != set(VALID_COLORS):
        raise ValueError("The stack must contain red, green, and blue exactly once")
    return colors


def interpret_user_prompt(user_prompt: str) -> UserPrompt:
    """Parse an interactive request as either an object search or a stack order."""
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("A non-empty prompt is required")
    config = _load_prompt_config("llama_task.json")
    schema = config.get("response_schema")
    if not isinstance(schema, dict):
        raise StackingModelError("Llama task config must contain an object response_schema")
    properties = schema.get("properties")
    relations_schema = properties.get("stacking_relations") if isinstance(properties, dict) else None
    relation_items = relations_schema.get("items") if isinstance(relations_schema, dict) else None
    relation_properties = (
        relation_items.get("properties") if isinstance(relation_items, dict) else None
    )
    if not isinstance(relation_properties, dict) or any(
        not isinstance(relation_properties.get(key), dict)
        or relation_properties[key].get("enum") != list(VALID_COLORS)
        for key in ("upper", "lower")
    ):
        raise StackingModelError(
            "Llama task relation colors must match the supported block colors"
        )
    response = _post_json(
        "/api/chat",
        {
            "model": _prompt_config_string(config, "model", "Llama"),
            "messages": [
                {
                    "role": "system",
                    "content": _prompt_config_string(config, "system_prompt", "Llama"),
                },
                {"role": "user", "content": user_prompt.strip()},
            ],
            "format": schema,
            "stream": False,
            "options": {"temperature": 0},
        },
    )
    message = response.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise StackingModelError("Llama response is missing message.content")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise StackingModelError(f"Llama returned invalid task JSON: {content!r}") from error
    if not isinstance(parsed, dict) or set(parsed) != {
        "intent",
        "query",
        "stacking_relations",
    }:
        raise StackingModelError(
            "Llama task response must contain only intent, query, and stacking_relations"
        )
    intent = parsed["intent"]
    query = parsed["query"]
    relations = parsed["stacking_relations"]
    if intent == "find_object":
        if not isinstance(query, str) or not query.strip() or len(query.strip()) > 200:
            raise StackingModelError("Llama returned an empty or invalid object query")
        if relations != []:
            raise StackingModelError(
                "Llama object-search response must have empty stacking_relations"
            )
        return UserPrompt(intent="find_object", query=query.strip())
    if intent == "stack_blocks":
        if query != "":
            raise StackingModelError("Llama stack response must have an empty object query")
        if not isinstance(relations, list):
            raise StackingModelError("Llama stacking_relations must be a list")
        try:
            ordered_relations = _ordered_stacking_relations(relations)
        except ValueError as error:
            raise StackingModelError(
                f"Could not validate Llama stacking relations: {error}"
            ) from error
        return UserPrompt(intent="stack_blocks", stacking_relations=ordered_relations)
    raise StackingModelError(f"Llama returned unsupported task intent {intent!r}")


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
    """Use local Llama to turn a free-text request into a visual grounding phrase."""
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
    """Ask d3 to choose the next legal high-level stacking event."""
    _validate_order(goal_order)
    if placed != goal_order[: len(placed)]:
        raise ValueError("Placed cubes must be a prefix of the requested stack order")
    if current_support != (placed[-1] if placed else None):
        raise ValueError("Current support does not match the placed stack")

    config = _load_prompt_config("d3_stacking.json")
    criteria_templates = config.get("criteria_templates")
    if not isinstance(criteria_templates, dict):
        raise StackingModelError("d3 prompt config must contain an object criteria_templates")

    if expected_event is not None:
        if expected_event not in STACK_ACTION_EVENTS:
            raise ValueError(f"Unsupported stack action event {expected_event!r}")
        if len(placed) >= len(goal_order) or goal_order[len(placed)] != (current_pair or {}).get("moving"):
            raise ValueError("Current pair does not match the next requested placement")
        if current_support != (current_pair or {}).get("support"):
            raise ValueError("Current pair support does not match the placed stack")
        choices = {
            expected_event: _criteria_template(
                criteria_templates,
                "authorize",
                "d3",
                event=expected_event.replace("_", " "),
                moving=current_pair["moving"],
                support=current_pair["support"],
            ),
            "stop": _criteria_template(criteria_templates, "stop_authorize", "d3"),
        }
        phase = f"authorize_{expected_event}"
    elif not placed:
        choices = {
            "establish_base": _criteria_template(criteria_templates, "establish_base", "d3"),
            "stop": _criteria_template(criteria_templates, "stop_base", "d3"),
        }
        phase = "choose_base_event"
    elif len(placed) < len(goal_order):
        choices = {
            "stack_next": _criteria_template(
                criteria_templates,
                "stack_next",
                "d3",
                next_cube=goal_order[len(placed)],
                current_support=current_support,
            ),
            "stop": _criteria_template(criteria_templates, "stop_placement", "d3"),
        }
        phase = "choose_placement_event"
    else:
        choices = {
            "finish": _criteria_template(criteria_templates, "finish", "d3"),
            "stop": _criteria_template(criteria_templates, "stop_finish", "d3"),
        }
        phase = "choose_finish_event"

    state = {
        "task": _prompt_config_string(config, "task", "d3"),
        "phase": phase,
        "goal_order_bottom_to_top": goal_order,
        "placed_bottom_to_top": placed,
        "current_support": current_support,
        "next_cube": goal_order[len(placed)] if len(placed) < len(goal_order) else None,
        "last_event_result": last_event_result,
        "current_pair": current_pair,
    }
    response = _post_json(
        "/v1/systemone",
        {
            "model": _prompt_config_string(config, "model", "d3"),
            "state": state,
            "questions": {
                _prompt_config_string(config, "question_id", "d3"): {
                    "type": _prompt_config_string(config, "question_type", "d3"),
                    "instructions": _prompt_config_string(config, "instructions", "d3"),
                    "criteria": choices,
                }
            },
        },
    )
    answers = response.get("answers")
    question_id = _prompt_config_string(config, "question_id", "d3")
    answer = answers.get(question_id) if isinstance(answers, dict) else None
    event = answer.get("choice") if isinstance(answer, dict) else None
    if not isinstance(event, str) or event not in choices:
        raise StackingModelError(
            f"d3 returned an invalid next_event choice {event!r}; expected one of {tuple(choices)}"
        )
    return event


def choose_stack_action(
    support: str,
    moving: str,
    expected_event: str,
    last_event_result: dict[str, object] | None,
) -> str:
    """Ask d3 to authorize one stage of a specifically requested block pair."""
    if support not in VALID_COLORS or moving not in VALID_COLORS or support == moving:
        raise ValueError("A stack action requires two distinct supported block colors")
    if expected_event not in STACK_ACTION_EVENTS:
        raise ValueError(f"Unsupported stack action event {expected_event!r}")

    config = _load_prompt_config("d3_stacking.json")
    criteria_templates = config.get("criteria_templates")
    if not isinstance(criteria_templates, dict):
        raise StackingModelError("d3 prompt config must contain an object criteria_templates")
    choices = {
        expected_event: _criteria_template(
            criteria_templates,
            "authorize",
            "d3",
            event=expected_event.replace("_", " "),
            moving=moving,
            support=support,
        ),
        "stop": _criteria_template(criteria_templates, "stop_authorize", "d3"),
    }
    state = {
        "task": "requested_pair_stacking",
        "phase": f"authorize_{expected_event}",
        "requested_pair": {"support": support, "moving": moving},
        "last_event_result": last_event_result,
    }
    response = _post_json(
        "/v1/systemone",
        {
            "model": _prompt_config_string(config, "model", "d3"),
            "state": state,
            "questions": {
                _prompt_config_string(config, "question_id", "d3"): {
                    "type": _prompt_config_string(config, "question_type", "d3"),
                    "instructions": _prompt_config_string(config, "instructions", "d3"),
                    "criteria": choices,
                }
            },
        },
    )
    answers = response.get("answers")
    question_id = _prompt_config_string(config, "question_id", "d3")
    answer = answers.get(question_id) if isinstance(answers, dict) else None
    event = answer.get("choice") if isinstance(answer, dict) else None
    if not isinstance(event, str) or event not in choices:
        raise StackingModelError(
            f"d3 returned an invalid stack action choice {event!r}; expected one of {tuple(choices)}"
        )
    return event


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


@lru_cache(maxsize=2)
def _load_prompt_config(filename: str) -> dict[str, object]:
    path = _CONFIG_DIR / filename
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StackingModelError(f"Unable to load prompt config {path}: {error}") from error
    if not isinstance(config, dict):
        raise StackingModelError(f"Prompt config {path} must contain a JSON object")
    return config


def _prompt_config_string(config: dict[str, object], key: str, model_name: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise StackingModelError(f"{model_name} prompt config must contain a non-empty {key}")
    return value


def _criteria_template(
    templates: dict[str, object],
    key: str,
    model_name: str,
    **values: str,
) -> str:
    template = templates.get(key)
    if not isinstance(template, str) or not template:
        raise StackingModelError(f"{model_name} prompt config must contain a non-empty {key} template")
    try:
        return template.format(**values)
    except (KeyError, ValueError) as error:
        raise StackingModelError(
            f"{model_name} prompt template {key!r} is invalid: {error}"
        ) from error


def _ollama_base_url() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip().rstrip("/")
    if "://" not in host:
        host = f"http://{host}"
    return host


def _post_json(path: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        f"{_ollama_base_url()}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
            result = json.loads(response.read())
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise StackingModelError(
            f"Ollama {path} returned HTTP {error.code}: {details}"
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise StackingModelError(f"Ollama request to {path} failed: {error}") from error
    except json.JSONDecodeError as error:
        raise StackingModelError(f"Ollama {path} returned invalid JSON") from error
    if not isinstance(result, dict):
        raise StackingModelError(f"Ollama {path} returned a non-object response")
    return result
