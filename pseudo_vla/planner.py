"""Choose a skill sequence one slot at a time and repair blocked steps."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pseudo_vla.skills import (
    REGISTRY,
    Catalog,
    SkillError,
    SkillStep,
    World,
    expand,
    simulate,
    step,
)
from pseudo_vla.task_plan import Destination, PickPlaceAction, TaskPlan
from pseudo_vla.tev1_client import Tev1Error, ask_boolean, ask_choice

MAX_PLAN_ACTIONS = 6
MAX_FIXES_PER_STEP = 3
_MAX_CHOICE_OPTIONS = 24


@dataclass(frozen=True)
class _Fix:
    key: str
    description: str
    steps: tuple[SkillStep, ...]


def plan_steps(user_prompt: str, scene_state: Mapping[str, object] | None = None) -> TaskPlan:
    """Fill one registered skill at a time, checking the symbolic world after each."""
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("A non-empty prompt is required")
    state = dict(scene_state or {})
    catalog = Catalog.from_scene_state(state)
    world = World.from_scene_state(state)
    if not catalog.objects:
        raise Tev1Error("Cannot route a task without objects in the current scene")

    planned: list[SkillStep] = []
    base = {"instruction": user_prompt.strip(), "scene": state}
    for index in range(MAX_PLAN_ACTIONS + 1):
        step_state = {
            **base,
            "phase": "select_next_action",
            "action_index": index,
            "steps_so_far": [_public_step(item) for item in planned],
        }
        if not ask_boolean("plan_more", step_state):
            break
        if index == MAX_PLAN_ACTIONS:
            raise Tev1Error(
                f"Tev1 requested more than the maximum of {MAX_PLAN_ACTIONS} actions"
            )
        skill_name = _choose_skill(world, step_state)
        filled = _fill_slots(REGISTRY[skill_name], world, catalog, step_state)
        for primitive in expand(step(skill_name, **filled)):
            prefix, world = resolve_prerequisites(world, primitive, catalog, step_state)
            planned.extend(prefix)
            world = simulate(world, [primitive])
            planned.append(primitive)

    if not planned:
        raise Tev1Error("Tev1 did not select any supported skills")
    return TaskPlan(
        intent="skill_sequence",
        query="",
        actions=actions_from_steps(planned),
        steps=tuple(planned),
    )


def resolve_prerequisites(
    world: World,
    skill_step: SkillStep,
    catalog: Catalog,
    state: Mapping[str, object],
) -> tuple[list[SkillStep], World]:
    """Insert at most three repairs so the requested step becomes legal."""
    inserted: list[SkillStep] = []
    for _ in range(MAX_FIXES_PER_STEP):
        reason = REGISTRY[skill_step.skill].requires(world, skill_step.as_dict())
        if reason is None:
            return inserted, world
        fixes = _legal_fixes(world, skill_step, catalog, reason)
        if not fixes:
            raise Tev1Error(
                f"No legal prerequisite can resolve {reason} before {skill_step.skill}"
            )
        chosen = fixes[0] if len(fixes) == 1 else _choose_fix(fixes, reason, skill_step, state)
        try:
            world = simulate(world, chosen.steps)
        except SkillError as error:
            raise Tev1Error(f"Prerequisite repair {chosen.key} is not legal: {error}") from error
        inserted.extend(chosen.steps)
    if REGISTRY[skill_step.skill].requires(world, skill_step.as_dict()) is not None:
        raise Tev1Error(
            f"Prerequisite repair exceeded {MAX_FIXES_PER_STEP} fixes for {skill_step.skill}"
        )
    return inserted, world


def actions_from_steps(steps: list[SkillStep] | tuple[SkillStep, ...]) -> tuple[PickPlaceAction, ...]:
    """Expose pick-then-place pairs through the older action view."""
    actions: list[PickPlaceAction] = []
    index = 0
    items = list(steps)
    while index < len(items):
        current = items[index]
        following = items[index + 1] if index + 1 < len(items) else None
        if (
            current.skill == "pick"
            and following is not None
            and following.skill == "place"
            and following.get("object") == current.get("object")
        ):
            actions.append(
                PickPlaceAction(
                    current.get("object"),
                    Destination(following.get("kind"), following.get("target")),  # type: ignore[arg-type]
                )
            )
            index += 2
            continue
        index += 1
    return tuple(actions)


def _choose_skill(world: World, state: Mapping[str, object]) -> str:
    del world
    options = {
        name: spec.summary
        for name, spec in REGISTRY.items()
        if spec.offered
    }
    return _choose("skill_next", state, options, "unsupported")


def _fill_slots(spec, world: World, catalog: Catalog, state: Mapping[str, object]) -> dict[str, str]:
    filled: dict[str, str] = {}
    for slot in spec.slots:
        options = slot.options_fn(world, catalog, filled)
        filled[slot.name] = _choose(
            slot.question,
            {**state, "phase": f"select_{slot.name}", "filled": dict(filled)},
            options,
            "none",
        )
    return filled


def _choose(
    question_id: str,
    state: Mapping[str, object],
    options: Mapping[str, str],
    stop_word: str,
) -> str:
    if stop_word in options:
        raise Tev1Error(f"{question_id} options must be computed without {stop_word!r}")
    if not options:
        raise Tev1Error(f"No legal options are available for {question_id}")
    if len(options) == 1:
        return next(iter(options))
    criteria = dict(options)
    criteria[stop_word] = f"Select {stop_word} when none of the listed options applies."
    if len(criteria) > _MAX_CHOICE_OPTIONS:
        raise Tev1Error(
            f"Tev1 can compare at most {_MAX_CHOICE_OPTIONS} options for {question_id}"
        )
    choice = ask_choice(question_id, dict(state), criteria)
    if choice == stop_word:
        raise Tev1Error(f"Tev1 selected {stop_word} for {question_id}")
    if choice not in options:
        raise Tev1Error(f"Tev1 returned {choice!r}, which was not offered for {question_id}")
    return choice


def _choose_fix(fixes: list[_Fix], reason: str, skill_step: SkillStep, state: Mapping[str, object]) -> _Fix:
    criteria = {fix.key: fix.description for fix in fixes}
    criteria["unsupported"] = "None of the listed repairs is appropriate."
    choice = ask_choice(
        "prerequisite_resolution",
        {
            **state,
            "phase": "resolve_prerequisite",
            "failure": reason,
            "skill": skill_step.skill,
            "slots": skill_step.as_dict(),
        },
        criteria,
    )
    if choice == "unsupported":
        raise Tev1Error("Tev1 rejected every prerequisite repair")
    for fix in fixes:
        if fix.key == choice:
            return fix
    raise Tev1Error(f"Tev1 returned repair {choice!r}, which was not offered")


def _legal_fixes(world: World, skill_step: SkillStep, catalog: Catalog, reason: str) -> list[_Fix]:
    slots = skill_step.as_dict()
    obj = slots.get("object") or slots.get("container") or slots.get("tool")
    fixes: list[_Fix] = []
    if reason == "gripper_full" and world.held and catalog.locations:
        parked = step("place", object=world.held, kind="at", target=catalog.locations[0])
        if _runnable(world, (parked,)):
            fixes.append(
                _Fix(
                    "park_held",
                    f"Place the held {world.held} at {catalog.locations[0]}.",
                    (parked,),
                )
            )
    if reason == "blocked" and obj:
        for blocker in world.resting_on(obj):
            if not catalog.locations:
                continue
            clearing = (
                step("pick", object=blocker),
                step("place", object=blocker, kind="at", target=catalog.locations[0]),
            )
            if _runnable(world, clearing):
                fixes.append(
                    _Fix(
                        f"clear:{blocker}",
                        f"Move {blocker} off {obj} and place it at {catalog.locations[0]}.",
                        clearing,
                    )
                )
    if reason == "closed_container" and "open" in REGISTRY:
        container = slots.get("container") or slots.get("target")
        if container is None and obj:
            container = world.container_of(obj)
        if container:
            opening = step("open", container=container)
            if _runnable(world, (opening,)):
                fixes.append(
                    _Fix(
                        f"open:{container}",
                        f"Open {container} before using what is inside.",
                        (opening,),
                    )
                )
    if reason == "not_held" and obj:
        picking = step("pick", object=obj)
        if _runnable(world, (picking,)):
            fixes.append(_Fix("pick_first", f"Pick {obj} before this skill.", (picking,)))
    return fixes


def _runnable(world: World, steps: tuple[SkillStep, ...]) -> bool:
    try:
        simulate(world, steps)
    except SkillError:
        return False
    return True


def _public_step(skill_step: SkillStep) -> dict[str, object]:
    return {"skill": skill_step.skill, **skill_step.as_dict()}
