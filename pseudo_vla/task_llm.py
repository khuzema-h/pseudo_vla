"""Natural-language task plans and bounded d3 event choices."""

from __future__ import annotations

import json

from pseudo_vla.stacking_llm import (
    StackingModelError,
    _load_prompt_config,
    _post_json,
    _prompt_config_string,
)
from pseudo_vla.task_plan import (
    PickPlaceAction,
    TaskPlan,
    validate_task_plan,
)

TASK_EVENTS = ("find", "plan", "approach", "grasp", "place")


def interpret_task_prompt(user_prompt: str, scene_state: dict[str, object] | None = None) -> TaskPlan:
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("A non-empty prompt is required")
    config = _load_prompt_config("llama_task_plan.json")
    schema = config.get("response_schema")
    if not isinstance(schema, dict):
        raise StackingModelError("Llama task-plan config must contain an object response_schema")
    state_text = json.dumps(scene_state or {}, sort_keys=True)
    user_content = (
        f"Instruction: {user_prompt.strip()}\n"
        f"Current scene state (ground truth): {state_text}\n"
        "Use only these current-scene objects and relationships; do not guess hidden or occluded state."
    )
    response = _post_json(
        "/api/chat",
        {
            "model": _prompt_config_string(config, "model", "Llama"),
            "messages": [
                {"role": "system", "content": _prompt_config_string(config, "system_prompt", "Llama")},
                {"role": "user", "content": user_content},
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
        raise StackingModelError(f"Llama returned invalid task plan JSON: {content!r}") from error
    if not isinstance(parsed, dict) or set(parsed) != {"intent", "query", "actions"}:
        raise StackingModelError("Llama task plan must contain only intent, query, and actions")
    objects, locations, containers = _scene_catalog(scene_state)
    try:
        return validate_task_plan(
            parsed["intent"],
            parsed["query"],
            parsed["actions"],
            available_objects=objects,
            available_locations=locations,
            available_containers=containers,
        )
    except ValueError as error:
        raise StackingModelError(f"Could not validate Llama task plan: {error}") from error


def _scene_catalog(
    scene_state: dict[str, object] | None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Read object, location, and container ids from a scene-state payload."""
    state = scene_state or {}
    raw_objects = state.get("objects", state.get("movable_objects", ()))
    return (
        _catalog_ids(raw_objects, nested_id=True),
        _catalog_ids(state.get("locations", ())),
        _catalog_ids(state.get("containers", ())),
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


def choose_task_event(
    event: str,
    action: PickPlaceAction | None,
    action_index: int,
    last_event_result: dict[str, object] | None,
    plan: TaskPlan | None = None,
) -> str:
    if event not in TASK_EVENTS:
        raise ValueError(f"Unsupported task event {event!r}")
    if action_index < 0:
        raise ValueError("Action index must be non-negative")
    config = _load_prompt_config("d3_task.json")
    choices = {
        event: _format_authorize(config, action, event, action_index),
        "stop": _prompt_config_string(config, "stop", "d3"),
    }
    state = {
        "task": _prompt_config_string(config, "task", "d3"),
        "phase": f"authorize_{event}",
        "action_index": action_index,
        "action": {
            "object": action.object_id,
            "destination": {
                "kind": action.destination.kind,
                "target": action.destination.target,
            },
        } if action is not None else None,
        "validated_plan": [
            {
                "object": item.object_id,
                "destination": {
                    "kind": item.destination.kind,
                    "target": item.destination.target,
                },
            }
            for item in plan.actions
        ] if plan is not None and plan.intent == "pick_place" else [],
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
    answer = (
        answers.get(_prompt_config_string(config, "question_id", "d3"))
        if isinstance(answers, dict)
        else None
    )
    selected = answer.get("choice") if isinstance(answer, dict) else None
    if not isinstance(selected, str) or selected not in choices:
        raise StackingModelError(
            f"d3 returned invalid task event {selected!r}; expected one of {tuple(choices)}"
        )
    return selected


def _format_authorize(
    config: dict[str, object],
    action: PickPlaceAction | None,
    event: str,
    action_index: int,
) -> str:
    template = _prompt_config_string(config, "authorize", "d3")
    if action is None:
        try:
            return template.format(
                event=event,
                action_index=action_index + 1,
                object="requested object",
                destination="requested destination",
            )
        except (KeyError, ValueError) as error:
            raise StackingModelError(f"d3 task prompt template is invalid: {error}") from error
    destination = action.destination.kind
    if action.destination.target:
        destination = f"{destination} {action.destination.target}"
    try:
        return template.format(
            event=event,
            action_index=action_index + 1,
            object=action.object_id,
            destination=destination,
        )
    except (KeyError, ValueError) as error:
        raise StackingModelError(f"d3 task prompt template is invalid: {error}") from error
