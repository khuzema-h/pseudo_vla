"""Model-orchestrated stacking state machine, independent of robot motion details."""

from __future__ import annotations


def run_prompted_stack_sequence(
    scene,
    perception,
    servo,
    goal_order: list[str],
    commands,
    max_steps: int,
    say,
    log,
    view,
    run_stack_phase,
) -> bool:
    """Orchestrate legal stack events while delegating physical work to the controller."""
    from pseudo_vla.stacking_llm import StackingModelError, choose_stack_event

    placed: list[str] = []
    current_support = None
    last_event_result: dict[str, object] | None = None

    while True:
        try:
            event = choose_stack_event(
                goal_order,
                placed,
                current_support,
                last_event_result,
            )
        except (StackingModelError, ValueError) as error:
            say(f"stack orchestration failed safely: {error}")
            return False

        say(f"d3 selected stack event: {event}")
        if event == "stop":
            say("d3 stopped the stack workflow at a safe boundary")
            return False
        if event == "establish_base":
            if placed:
                say("d3 selected establish_base after the base was already set")
                return False
            current_support = goal_order[0]
            placed.append(current_support)
            last_event_result = {
                "event": event,
                "success": True,
                "base": current_support,
                "detail": "Base cube remains supported by the table.",
            }
            say(f"using {current_support} as the table-supported base")
            continue
        if event == "stack_next":
            if not placed or len(placed) >= len(goal_order) or current_support != placed[-1]:
                say("d3 selected stack_next in an invalid task state")
                return False
            moving_name = goal_order[len(placed)]

            def authorize_event(stage: str, completed: bool = False) -> bool:
                nonlocal last_event_result
                if completed:
                    last_event_result = {
                        "event": stage,
                        "success": True,
                        "support": current_support,
                        "moving": moving_name,
                    }
                    return True
                try:
                    selected = choose_stack_event(
                        goal_order,
                        placed,
                        current_support,
                        last_event_result,
                        expected_event=stage,
                        current_pair={"support": current_support, "moving": moving_name},
                    )
                except (StackingModelError, ValueError) as error:
                    say(f"d3 could not authorize {stage} safely: {error}")
                    return False
                say(f"d3 selected stack event: {selected}")
                if selected != stage:
                    say(f"d3 stopped the stack workflow before {stage}")
                    return False
                return True

            complete = run_stack_phase(
                scene,
                perception,
                servo,
                current_support,
                moving_name,
                commands,
                max_steps,
                say,
                log,
                view,
                authorize_event,
            )
            if not complete:
                last_event_result = {
                    "event": event,
                    "success": False,
                    "support": current_support,
                    "moving": moving_name,
                    "detail": "The controller did not complete the pairwise placement.",
                }
                return False
            placed.append(moving_name)
            current_support = moving_name
            last_event_result = {
                "event": event,
                "success": True,
                "support": placed[-2],
                "moving": moving_name,
            }
            continue
        if event == "finish":
            if placed != goal_order:
                say("d3 selected finish before the requested stack was complete")
                return False
            return True
        say(f"d3 selected unsupported stack event {event!r}")
        return False


def run_requested_stack_relations(
    scene,
    perception,
    servo,
    relations,
    commands,
    max_steps: int,
    say,
    log,
    view,
    run_stack_phase,
) -> bool:
    """Execute only the explicitly requested support/moving pairs, bottom-up."""
    from pseudo_vla.stacking_llm import StackingModelError, choose_stack_action

    last_event_result: dict[str, object] | None = None
    for relation in relations:
        support = relation.support
        moving = relation.moving

        def authorize_event(stage: str, completed: bool = False) -> bool:
            nonlocal last_event_result
            if completed:
                last_event_result = {
                    "event": stage,
                    "success": True,
                    "support": support,
                    "moving": moving,
                }
                return True
            try:
                selected = choose_stack_action(
                    support,
                    moving,
                    stage,
                    last_event_result,
                )
            except (StackingModelError, ValueError) as error:
                say(f"d3 could not authorize {stage} safely: {error}")
                return False
            say(f"d3 selected stack event: {selected}")
            if selected != stage:
                say(f"d3 stopped the stack workflow before {stage}")
                return False
            return True

        say(f"placing {moving} on {support}")
        if not run_stack_phase(
            scene,
            perception,
            servo,
            support,
            moving,
            commands,
            max_steps,
            say,
            log,
            view,
            authorize_event,
        ):
            say(f"stopped before completing requested placement: {moving} on {support}")
            return False
        last_event_result = {
            "event": "placement_complete",
            "success": True,
            "support": support,
            "moving": moving,
        }
    return True
