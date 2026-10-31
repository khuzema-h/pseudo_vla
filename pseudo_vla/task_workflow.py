"""Model-orchestrated execution of validated generic pick-and-place plans."""

from __future__ import annotations

from pseudo_vla.task_plan import TaskPlan


def run_task_plan(plan: TaskPlan, say, run_find, run_action) -> bool:
    """Orchestrate plan-level events and delegate physical actions to the controller."""
    from pseudo_vla.stacking_llm import StackingModelError
    from pseudo_vla.task_llm import choose_task_event

    last_event_result: dict[str, object] | None = None

    def authorize(event, action, index):
        nonlocal last_event_result
        try:
            selected = choose_task_event(event, action, index, last_event_result, plan)
        except (StackingModelError, ValueError) as error:
            say(f"d3 could not authorize {event} safely: {error}")
            return False
        say(f"d3 selected task event: {selected}")
        if selected != event:
            say(f"d3 stopped the task before {event}")
            return False
        return True

    if plan.intent == "find":
        if not authorize("find", None, 0):
            return False
        complete = run_find(plan.query or "")
        last_event_result = {"event": "find", "success": complete}
        return complete

    for index, action in enumerate(plan.actions):
        if not authorize("find", action, index):
            return False
        if not authorize("plan", action, index):
            return False

        def authorize_action(stage, completed=False):
            nonlocal last_event_result
            if completed:
                last_event_result = {
                    "event": stage,
                    "success": True,
                    "action_index": index,
                    "object": action.object_id,
                }
                return True
            if not authorize(stage, action, index):
                return False
            return True

        if not run_action(action, authorize_action):
            last_event_result = {
                "event": "pick_place",
                "success": False,
                "action_index": index,
                "object": action.object_id,
            }
            return False
        last_event_result = {
            "event": "place",
            "success": True,
            "action_index": index,
            "object": action.object_id,
        }
    return True
