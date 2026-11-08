"""Model-orchestrated execution of validated generic pick-and-place plans."""

from __future__ import annotations

from pseudo_vla.task_plan import TaskPlan


def run_task_plan(plan: TaskPlan, say, run_find, run_action, handler=None) -> bool:
    """Orchestrate plan-level events and delegate physical actions to the controller."""
    if plan.intent == "skill_sequence":
        return _run_skill_plan(plan, say, handler)

    from pseudo_vla.tev1_client import Tev1Error
    from pseudo_vla.task_llm import choose_task_event

    last_event_result: dict[str, object] | None = None

    def authorize(event, action, index):
        nonlocal last_event_result
        try:
            selected = choose_task_event(event, action, index, last_event_result, plan)
        except (Tev1Error, ValueError) as error:
            say(f"Tev1 could not authorize {event} safely: {error}")
            return False
        say(f"Tev1 selected task event: {selected}")
        if selected != event:
            say(f"Tev1 stopped the task before {event}")
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
        if (
            last_event_result is None
            or last_event_result.get("event") != "verify_placement"
            or last_event_result.get("success") is not True
        ):
            say("controller reported success without completing placement verification")
            return False
        last_event_result = {
            **(last_event_result or {}),
            "success": True,
            "action_index": index,
            "object": action.object_id,
        }
    return True


def _run_skill_plan(plan: TaskPlan, say, handler) -> bool:
    """Run each primitive skill phase through a handler, then update the world."""
    from pseudo_vla.phase_handlers import FakeBackend
    from pseudo_vla.phases import choose_phase_event, phases_for
    from pseudo_vla.skills import SkillError, World, simulate
    from pseudo_vla.tev1_client import Tev1Error

    backend = handler or FakeBackend()
    world = World()
    for skill_step in plan.steps:
        phases = phases_for(skill_step.skill)
        phase_idx = 0
        retries: dict[str, int] = {}
        last_result: dict[str, object] | None = None
        while phase_idx < len(phases):
            try:
                selected = choose_phase_event(skill_step.skill, phase_idx, last_result, retries)
            except (Tev1Error, ValueError) as error:
                say(f"Tev1 could not authorize {skill_step.skill} safely: {error}")
                return False
            if selected == "stop":
                expected = phases[phase_idx]
                say(f"Tev1 stopped the task before {expected}")
                return False
            failed = isinstance(last_result, dict) and last_result.get("ok") is False
            if failed and selected != phases[phase_idx]:
                outcome = backend.run(skill_step, selected, world)
                if outcome.ok:
                    last_result = {"ok": True, "data": dict(outcome.data)}
                    continue
                failure = str(last_result.get("failure") or "timeout")
                retries[failure] = retries.get(failure, 0) + 1
                last_result = {
                    "ok": False,
                    "failure": outcome.failure or failure,
                    "data": dict(outcome.data),
                }
                continue
            if selected != phases[phase_idx]:
                say(f"Tev1 stopped the task before {phases[phase_idx]}")
                return False
            outcome = backend.run(skill_step, selected, world)
            say(f"Tev1 selected phase: {selected}")
            if not outcome.ok:
                last_result = {
                    "ok": False,
                    "failure": outcome.failure or "timeout",
                    "data": dict(outcome.data),
                }
                continue
            last_result = {"ok": True, "data": dict(outcome.data)}
            phase_idx += 1
        try:
            world = simulate(world, [skill_step])
        except SkillError as error:
            say(f"{skill_step.skill} was not legal after its phases: {error}")
            return False
    return True
