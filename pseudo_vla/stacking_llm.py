"""Local Ollama prompt parsing and task-event decisions for cube stacking."""

from __future__ import annotations

import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

VALID_COLORS = ("red", "green", "blue")
LLAMA_MODEL = "llama3.2:3b"
D3_MODEL = "d3:4b"
OLLAMA_TIMEOUT = 120.0
_COLOR_PATTERN = re.compile(r"^(red|green|blue)(?:\s+(?:cube|block)s?)?$", re.IGNORECASE)
STACK_ACTION_EVENTS = ("approach", "grasp", "lift", "place", "verify_placement")


class StackingModelError(RuntimeError):
    """Raised when an Ollama request or model response cannot be used safely."""


def parse_stacking_prompt(user_prompt: str) -> list[str]:
    """Parse a natural-language request into a bottom-to-top order."""
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("A non-empty stacking prompt is required")
    mentioned_colors = {
        color for color in VALID_COLORS if re.search(rf"\b{color}\b", user_prompt, re.IGNORECASE)
    }
    if mentioned_colors != set(VALID_COLORS):
        raise ValueError("The prompt must specify red, green, and blue in bottom-to-top order")

    schema = {
        "type": "object",
        "properties": {
            "sequence": {
                "type": "array",
                "items": {"type": "string", "enum": list(VALID_COLORS)},
            }
        },
        "required": ["sequence"],
        "additionalProperties": False,
    }
    response = _post_json(
        "/api/chat",
        {
            "model": LLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Parse the user's requested bottom-to-top stacking order. "
                        "Return exactly three distinct colors, red, green, and blue, "
                        "in the order requested, as JSON with a sequence array. "
                        "Do not infer a color the user did not request."
                    ),
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

    if expected_event is not None:
        if expected_event not in STACK_ACTION_EVENTS:
            raise ValueError(f"Unsupported stack action event {expected_event!r}")
        if len(placed) >= len(goal_order) or goal_order[len(placed)] != (current_pair or {}).get("moving"):
            raise ValueError("Current pair does not match the next requested placement")
        if current_support != (current_pair or {}).get("support"):
            raise ValueError("Current pair support does not match the placed stack")
        choices = {
            expected_event: (
                f"Authorize the {expected_event.replace('_', ' ')} stage for "
                f"{current_pair['moving']} on {current_pair['support']}."
            ),
            "stop": "Stop at this safe task boundary without executing the next stage.",
        }
        phase = f"authorize_{expected_event}"
    elif not placed:
        choices = {
            "establish_base": "Establish the first requested cube as the table-supported base.",
            "stop": "Stop the stacking workflow without moving the robot.",
        }
        phase = "choose_base_event"
    elif len(placed) < len(goal_order):
        choices = {
            "stack_next": (
                f"Place the next required cube ({goal_order[len(placed)]}) "
                f"on the current support ({current_support})."
            ),
            "stop": "Stop the stacking workflow at the current safe boundary.",
        }
        phase = "choose_placement_event"
    else:
        choices = {
            "finish": "Finish because all requested cubes have been placed.",
            "stop": "Stop the workflow at the current safe boundary.",
        }
        phase = "choose_finish_event"

    state = {
        "task": "three_cube_stacking",
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
            "model": D3_MODEL,
            "state": state,
            "questions": {
                "next_event": {
                    "type": "choice",
                    "instructions": (
                        "Choose the next event for the stacking task. Follow the "
                        "requested bottom-to-top order and choose only an event "
                        "whose conditions are satisfied by the state."
                    ),
                    "criteria": choices,
                }
            },
        },
    )
    answers = response.get("answers")
    answer = answers.get("next_event") if isinstance(answers, dict) else None
    event = answer.get("choice") if isinstance(answer, dict) else None
    if not isinstance(event, str) or event not in choices:
        raise StackingModelError(
            f"d3 returned an invalid next_event choice {event!r}; expected one of {tuple(choices)}"
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
