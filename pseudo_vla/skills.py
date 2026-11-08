"""Symbolic world and skill registry for one arm and one parallel gripper.

Perception and motion stay outside this module. A skill only declares the slots
Tev1 may fill, the phases the executor may run, and how the symbolic world
changes when the skill succeeds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

_DESTINATION_KINDS = ("at", "on", "left_of", "right_of", "inside")

PHASES: dict[str, tuple[str, ...]] = {
    "pick": ("approach", "grasp", "lift", "verify_grasp"),
    "place": ("move_above", "place", "retreat", "verify_placement"),
    "push": ("approach", "contact", "push", "retreat", "verify_effect"),
    "open": ("approach", "contact", "open", "verify_effect"),
    "close": ("approach", "contact", "close", "verify_effect"),
    "insert": ("move_above", "pre_insert", "guarded_insert", "retreat", "verify_effect"),
    "pour": ("move_above", "pour", "retreat", "verify_effect"),
    "press": ("approach", "contact", "press", "retreat", "verify_effect"),
    "rotate": ("approach", "grasp", "rotate", "retreat", "verify_effect"),
}

_RELATIONS = ("on", "left_of", "right_of")
OptionsFn = Callable[["World", "Catalog", Mapping[str, str]], dict[str, str]]
RequireFn = Callable[["World", Mapping[str, str]], str | None]
ApplyFn = Callable[["World", Mapping[str, str]], "World"]


class SkillError(ValueError):
    """Raised when a skill is illegal in the current symbolic world."""


@dataclass(frozen=True)
class Catalog:
    """Identifiers the planner is allowed to offer Tev1."""

    objects: tuple[str, ...] = ()
    locations: tuple[str, ...] = ()
    containers: tuple[str, ...] = ()

    @classmethod
    def from_scene_state(cls, scene_state: Mapping[str, object] | None) -> "Catalog":
        state = scene_state or {}
        return cls(
            objects=_identifiers(state.get("objects", state.get("movable_objects", ()))),
            locations=_identifiers(state.get("locations", ())),
            containers=_identifiers(state.get("containers", ())),
        )


@dataclass(frozen=True)
class World:
    """Facts the skill checker can prove without looking at pixels or joints."""

    held: str | None = None
    on: tuple[tuple[str, str], ...] = ()
    inside: tuple[tuple[str, str], ...] = ()
    opened: tuple[str, ...] = ()
    containers: tuple[str, ...] = ()

    @classmethod
    def from_scene_state(cls, scene_state: Mapping[str, object] | None) -> "World":
        state = scene_state or {}
        return cls(
            held=_optional_name(state.get("held")),
            on=_pairs(state.get("supported_by")),
            inside=_pairs(state.get("inside")),
            opened=_identifiers(state.get("opened", ())),
            containers=_identifiers(state.get("containers", ())),
        )

    def support_of(self, object_id: str) -> str | None:
        for obj, support in self.on:
            if obj == object_id:
                return support
        return None

    def resting_on(self, support_id: str) -> tuple[str, ...]:
        return tuple(obj for obj, support in self.on if support == support_id)

    def container_of(self, object_id: str) -> str | None:
        for obj, container in self.inside:
            if obj == object_id:
                return container
        return None

    def is_open(self, container_id: str) -> bool:
        return container_id in self.opened

    def release(self, object_id: str) -> "World":
        """Drop every support and containment fact about one object."""
        return _replace(
            self,
            held=None if self.held == object_id else self.held,
            on=tuple(pair for pair in self.on if pair[0] != object_id),
            inside=tuple(pair for pair in self.inside if pair[0] != object_id),
        )


@dataclass(frozen=True)
class SkillStep:
    """One concrete skill call after its slots have been filled."""

    skill: str
    slots: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> dict[str, str]:
        return dict(self.slots)

    def get(self, name: str) -> str:
        for key, value in self.slots:
            if key == name:
                return value
        raise KeyError(name)


def step(skill: str, **slots: str) -> SkillStep:
    return SkillStep(skill, tuple(slots.items()))


@dataclass(frozen=True)
class Slot:
    name: str
    question: str
    options_fn: OptionsFn


@dataclass(frozen=True)
class SkillSpec:
    name: str
    slots: tuple[Slot, ...]
    phases: tuple[str, ...]
    requires: RequireFn
    apply: ApplyFn
    summary: str
    expands_to: tuple[str, ...] = ()
    ends_plan: bool = False

    @property
    def offered(self) -> bool:
        return True


def _require_pick(world: World, slots: Mapping[str, str]) -> str | None:
    obj = slots["object"]
    if world.held:
        return "gripper_full"
    if world.resting_on(obj):
        return "blocked"
    container = world.container_of(obj)
    if container and not world.is_open(container):
        return "closed_container"
    return None


def _apply_pick(world: World, slots: Mapping[str, str]) -> World:
    obj = slots["object"]
    released = world.release(obj)
    return _replace(released, held=obj)


def _require_place(world: World, slots: Mapping[str, str]) -> str | None:
    obj = slots["object"]
    kind = slots["kind"]
    target = slots["target"]
    if world.held != obj:
        return "not_held"
    if kind not in _DESTINATION_KINDS:
        return "bad_destination"
    if obj == target:
        return "bad_destination"
    if kind == "inside" and not world.is_open(target):
        return "closed_container"
    return None


def _apply_place(world: World, slots: Mapping[str, str]) -> World:
    obj = slots["object"]
    kind = slots["kind"]
    target = slots["target"]
    released = world.release(obj)
    on = released.on
    inside = released.inside
    if kind == "on":
        on = tuple(sorted(on + ((obj, target),)))
    elif kind == "inside":
        inside = tuple(sorted(inside + ((obj, target),)))
    return _replace(released, held=None, on=on, inside=inside)


def _require_pick_place(world: World, slots: Mapping[str, str]) -> str | None:
    reason = _require_pick(world, {"object": slots["object"]})
    if reason:
        return reason
    held = _apply_pick(world, {"object": slots["object"]})
    return _require_place(
        held,
        {"object": slots["object"], "kind": slots["kind"], "target": slots["target"]},
    )


def _apply_pick_place(world: World, slots: Mapping[str, str]) -> World:
    held = _apply_pick(world, {"object": slots["object"]})
    return _apply_place(
        held,
        {"object": slots["object"], "kind": slots["kind"], "target": slots["target"]},
    )


def _object_options(world: World, catalog: Catalog, _filled: Mapping[str, str]) -> dict[str, str]:
    return {
        object_id: f"Use scene object {object_id}."
        for object_id in catalog.objects
        if object_id != world.held
    }


def _held_or_object_options(
    world: World,
    catalog: Catalog,
    _filled: Mapping[str, str],
) -> dict[str, str]:
    if world.held:
        return {world.held: f"The gripper is holding {world.held}."}
    return _object_options(world, catalog, _filled)


def _kind_options(
    _world: World,
    catalog: Catalog,
    filled: Mapping[str, str],
) -> dict[str, str]:
    moving = filled.get("object")
    criteria: dict[str, str] = {}
    if catalog.locations:
        criteria["at"] = "Place the object at a listed location."
    others = [object_id for object_id in catalog.objects if object_id != moving]
    if others:
        criteria["on"] = "Place the object on another scene object."
        criteria["left_of"] = "Place the object to the left of another scene object."
        criteria["right_of"] = "Place the object to the right of another scene object."
    if catalog.containers:
        criteria["inside"] = "Place the object inside a listed container."
    return criteria


def _target_options(
    _world: World,
    catalog: Catalog,
    filled: Mapping[str, str],
) -> dict[str, str]:
    kind = filled.get("kind")
    moving = filled.get("object")
    if kind == "at":
        identifiers = catalog.locations
    elif kind == "inside":
        identifiers = catalog.containers
    elif kind in _RELATIONS:
        identifiers = tuple(object_id for object_id in catalog.objects if object_id != moving)
    else:
        identifiers = ()
    return {
        identifier: f"Use destination {identifier}."
        for identifier in identifiers
    }


def _replace(world: World, **changes: object) -> World:
    values: dict[str, object] = {
        "held": world.held,
        "on": world.on,
        "inside": world.inside,
        "opened": world.opened,
        "containers": world.containers,
    }
    values.update(changes)
    return World(**values)  # type: ignore[arg-type]


def _require_free(world: World, slots: Mapping[str, str]) -> str | None:
    if world.held:
        return "gripper_full"
    obj = slots.get("object")
    if obj and world.resting_on(obj):
        return "blocked"
    if obj:
        container = world.container_of(obj)
        if container and not world.is_open(container):
            return "closed_container"
    return None


def _require_open(world: World, slots: Mapping[str, str]) -> str | None:
    if world.held:
        return "gripper_full"
    if world.is_open(slots["container"]):
        return "already_open"
    return None


def _apply_open(world: World, slots: Mapping[str, str]) -> World:
    container = slots["container"]
    return _replace(world, opened=tuple(sorted((*world.opened, container))))


def _require_close(world: World, slots: Mapping[str, str]) -> str | None:
    if world.held:
        return "gripper_full"
    if not world.is_open(slots["container"]):
        return "already_closed"
    return None


def _apply_close(world: World, slots: Mapping[str, str]) -> World:
    container = slots["container"]
    return _replace(world, opened=tuple(name for name in world.opened if name != container))


def _apply_push(world: World, slots: Mapping[str, str]) -> World:
    return world.release(slots["object"])


def _require_insert(world: World, slots: Mapping[str, str]) -> str | None:
    obj = slots["object"]
    receptacle = slots["receptacle"]
    if world.held != obj:
        return "not_held"
    if obj == receptacle:
        return "bad_destination"
    if receptacle in world.containers and not world.is_open(receptacle):
        return "closed_container"
    return None


def _apply_insert(world: World, slots: Mapping[str, str]) -> World:
    released = world.release(slots["object"])
    inside = tuple(sorted(released.inside + ((slots["object"], slots["receptacle"]),)))
    return _replace(released, held=None, inside=inside)


def _require_extract(world: World, slots: Mapping[str, str]) -> str | None:
    if world.held:
        return "gripper_full"
    obj = slots["object"]
    receptacle = slots["receptacle"]
    if world.container_of(obj) != receptacle:
        return "not_inside"
    if receptacle in world.containers and not world.is_open(receptacle):
        return "closed_container"
    return None


def _apply_extract(world: World, slots: Mapping[str, str]) -> World:
    return _apply_pick(world, {"object": slots["object"]})


def _require_stack(world: World, slots: Mapping[str, str]) -> str | None:
    if slots["moving"] == slots["base"]:
        return "bad_destination"
    return _require_pick_place(
        world,
        {"object": slots["moving"], "kind": "on", "target": slots["base"]},
    )


def _apply_stack(world: World, slots: Mapping[str, str]) -> World:
    return _apply_pick_place(
        world,
        {"object": slots["moving"], "kind": "on", "target": slots["base"]},
    )


def _require_unstack(world: World, slots: Mapping[str, str]) -> str | None:
    if world.support_of(slots["object"]) is None:
        return "not_supported"
    return _require_pick_place(world, slots)


def _require_held_use(world: World, slots: Mapping[str, str]) -> str | None:
    obj = slots.get("object") or slots.get("tool")
    if not obj or world.held != obj:
        return "not_held"
    target = slots.get("target") or slots.get("receptacle")
    if target and target == obj:
        return "bad_destination"
    return None


def _require_repeat(world: World, slots: Mapping[str, str]) -> str | None:
    del world
    if slots.get("count") not in {"1", "2", "3"}:
        return "bad_count"
    return None


def _identity(world: World, slots: Mapping[str, str]) -> World:
    del slots
    return world


def _closed_container_options(world: World, catalog: Catalog, _filled: Mapping[str, str]) -> dict[str, str]:
    return {
        name: f"Closed container {name}."
        for name in catalog.containers
        if not world.is_open(name)
    }


def _open_container_options(world: World, catalog: Catalog, _filled: Mapping[str, str]) -> dict[str, str]:
    return {
        name: f"Open container {name}."
        for name in catalog.containers
        if world.is_open(name)
    }


def _direction_options(_world: World, _catalog: Catalog, _filled: Mapping[str, str]) -> dict[str, str]:
    return {
        "left": "Push toward world -X.",
        "right": "Push toward world +X.",
        "forward": "Push toward world +Y.",
        "back": "Push toward world -Y.",
    }


def _parallel_grasp(_world: World, _catalog: Catalog, _filled: Mapping[str, str]) -> dict[str, str]:
    return {"parallel": "Use the parallel gripper."}


def _upright(_world: World, _catalog: Catalog, _filled: Mapping[str, str]) -> dict[str, str]:
    return {"upright": "Keep the held object upright."}


def _hang_constraint(_world: World, _catalog: Catalog, _filled: Mapping[str, str]) -> dict[str, str]:
    return {
        "upright": "Hang the object upright.",
        "free": "Hang the object with no orientation constraint.",
    }


def _other_object_options(world: World, catalog: Catalog, filled: Mapping[str, str]) -> dict[str, str]:
    moving = filled.get("object") or filled.get("tool") or filled.get("moving") or world.held
    return {
        object_id: f"Use {object_id}."
        for object_id in catalog.objects
        if object_id != moving
    }


def _receptacle_options(world: World, catalog: Catalog, filled: Mapping[str, str]) -> dict[str, str]:
    moving = filled.get("object") or world.held
    options = {
        object_id: f"Insert into object {object_id}."
        for object_id in catalog.objects
        if object_id != moving
    }
    options.update(
        {
            name: f"Insert into open container {name}."
            for name in catalog.containers
            if world.is_open(name) and name != moving
        }
    )
    return options


def _extract_object_options(world: World, catalog: Catalog, _filled: Mapping[str, str]) -> dict[str, str]:
    contained = {obj for obj, _container in world.inside}
    return {
        object_id: f"Extract {object_id}."
        for object_id in catalog.objects
        if object_id in contained and object_id != world.held
    }


def _extract_receptacle_options(world: World, _catalog: Catalog, filled: Mapping[str, str]) -> dict[str, str]:
    obj = filled.get("object")
    container = world.container_of(obj) if obj else None
    if container is None:
        return {}
    return {container: f"{obj} is inside {container}."}


def _stack_base_options(_world: World, catalog: Catalog, _filled: Mapping[str, str]) -> dict[str, str]:
    return {object_id: f"Use {object_id} as the stack support." for object_id in catalog.objects}


def _stack_next_options(_world: World, catalog: Catalog, filled: Mapping[str, str]) -> dict[str, str]:
    base = filled.get("base")
    return {
        object_id: f"Place {object_id} on {base}."
        for object_id in catalog.objects
        if object_id != base
    }


def _supported_object_options(world: World, catalog: Catalog, _filled: Mapping[str, str]) -> dict[str, str]:
    return {
        object_id: f"Unstack {object_id}."
        for object_id in catalog.objects
        if world.support_of(object_id) and object_id != world.held
    }


def _pour_target_options(world: World, catalog: Catalog, filled: Mapping[str, str]) -> dict[str, str]:
    moving = filled.get("object") or world.held
    options = _other_object_options(world, catalog, {"object": moving or ""})
    options.update(
        {
            name: f"Pour into open container {name}."
            for name in catalog.containers
            if world.is_open(name)
        }
    )
    return options


def _repeat_options(_world: World, _catalog: Catalog, _filled: Mapping[str, str]) -> dict[str, str]:
    return {str(count): f"Repeat the previous skill {count} times." for count in (1, 2, 3)}


def _clarification_options(_world: World, _catalog: Catalog, _filled: Mapping[str, str]) -> dict[str, str]:
    return {
        "missing_object": "The instruction does not identify the object.",
        "missing_destination": "The instruction does not identify the destination.",
        "ambiguous_skill": "More than one skill matches the instruction.",
    }


_OBJECT = Slot("object", "moving_object", _object_options)
_HELD_OBJECT = Slot("object", "moving_object", _held_or_object_options)
_KIND = Slot("kind", "destination_kind", _kind_options)
_TARGET = Slot("target", "destination_target", _target_options)
_CONTAINER = Slot("container", "articulation_action", _closed_container_options)
_OPEN_CONTAINER = Slot("container", "articulation_action", _open_container_options)
_DIRECTION = Slot("direction", "push_direction", _direction_options)
_GRASP = Slot("grasp_family", "grasp_family", _parallel_grasp)
_UPRIGHT = Slot("transport_constraint", "transport_constraint", _upright)
_HANG_CONSTRAINT = Slot("transport_constraint", "transport_constraint", _hang_constraint)
_RECEPTACLE = Slot("receptacle", "insert_receptacle", _receptacle_options)
_POUR_TARGET = Slot("target", "pour_target", _pour_target_options)
_TOOL = Slot("tool", "tool_object", _object_options)
_OTHER = Slot("target", "moving_object", _other_object_options)
_STACK_BASE = Slot("base", "stack_base", _stack_base_options)
_STACK_NEXT = Slot("moving", "stack_next", _stack_next_options)
_SUPPORTED = Slot("object", "moving_object", _supported_object_options)
_EXTRACT_OBJECT = Slot("object", "moving_object", _extract_object_options)
_EXTRACT_RECEPTACLE = Slot("receptacle", "insert_receptacle", _extract_receptacle_options)
_REPEAT = Slot("count", "repeat_count", _repeat_options)
_CLARIFY = Slot("topic", "needs_clarification", _clarification_options)

REGISTRY: dict[str, SkillSpec] = {
    "pick": SkillSpec(
        name="pick",
        slots=(_OBJECT,),
        phases=PHASES["pick"],
        requires=_require_pick,
        apply=_apply_pick,
        summary="Pick up one scene object with the empty gripper.",
    ),
    "place": SkillSpec(
        name="place",
        slots=(_HELD_OBJECT, _KIND, _TARGET),
        phases=PHASES["place"],
        requires=_require_place,
        apply=_apply_place,
        summary="Put the held object at a listed destination.",
    ),
    "pick_place": SkillSpec(
        name="pick_place",
        slots=(_OBJECT, _KIND, _TARGET),
        phases=PHASES["pick"] + PHASES["place"],
        requires=_require_pick_place,
        apply=_apply_pick_place,
        summary="Pick one object and put it at a listed destination.",
        expands_to=("pick", "place"),
    ),
    "open": SkillSpec(
        name="open",
        slots=(_CONTAINER,),
        phases=PHASES["open"],
        requires=_require_open,
        apply=_apply_open,
        summary="Open one closed container.",
    ),
    "close": SkillSpec(
        name="close",
        slots=(_OPEN_CONTAINER,),
        phases=PHASES["close"],
        requires=_require_close,
        apply=_apply_close,
        summary="Close one open container.",
    ),
    "push": SkillSpec(
        name="push",
        slots=(_OBJECT, _DIRECTION),
        phases=PHASES["push"],
        requires=_require_free,
        apply=_apply_push,
        summary="Push one clear object along a listed direction.",
    ),
    "press": SkillSpec(
        name="press",
        slots=(_OBJECT,),
        phases=PHASES["press"],
        requires=_require_free,
        apply=_identity,
        summary="Press one clear object.",
    ),
    "rotate": SkillSpec(
        name="rotate",
        slots=(_OBJECT, _GRASP),
        phases=PHASES["rotate"],
        requires=_require_free,
        apply=_identity,
        summary="Rotate one clear object with the parallel gripper.",
    ),
    "insert": SkillSpec(
        name="insert",
        slots=(_HELD_OBJECT, _RECEPTACLE),
        phases=PHASES["insert"],
        requires=_require_insert,
        apply=_apply_insert,
        summary="Insert the held object into a listed receptacle.",
    ),
    "extract": SkillSpec(
        name="extract",
        slots=(_EXTRACT_OBJECT, _EXTRACT_RECEPTACLE),
        phases=PHASES["pick"],
        requires=_require_extract,
        apply=_apply_extract,
        summary="Take one object out of the container it is inside.",
    ),
    "stack": SkillSpec(
        name="stack",
        slots=(_STACK_BASE, _STACK_NEXT),
        phases=PHASES["pick"] + PHASES["place"],
        requires=_require_stack,
        apply=_apply_stack,
        summary="Pick one object and place it on a chosen support.",
        expands_to=("pick", "place"),
    ),
    "unstack": SkillSpec(
        name="unstack",
        slots=(_SUPPORTED, _KIND, _TARGET),
        phases=PHASES["pick"] + PHASES["place"],
        requires=_require_unstack,
        apply=_apply_pick_place,
        summary="Pick a supported object and place it at a listed destination.",
        expands_to=("pick", "place"),
    ),
    "pour": SkillSpec(
        name="pour",
        slots=(_HELD_OBJECT, _POUR_TARGET, _UPRIGHT),
        phases=PHASES["pour"],
        requires=_require_held_use,
        apply=_identity,
        summary="Pour from the held object into a listed target.",
    ),
    "use_tool": SkillSpec(
        name="use_tool",
        slots=(_TOOL, _OTHER, _GRASP),
        phases=("approach", "contact", "verify_effect"),
        requires=_require_held_use,
        apply=_identity,
        summary="Use one held tool on another object.",
    ),
    "hang": SkillSpec(
        name="hang",
        slots=(_HELD_OBJECT, _OTHER, _HANG_CONSTRAINT),
        phases=("move_above", "place", "retreat", "verify_effect"),
        requires=_require_held_use,
        apply=lambda world, slots: _apply_place(
            world, {"object": slots["object"], "kind": "on", "target": slots["target"]}
        ),
        summary="Hang the held object on a listed support.",
    ),
    "wait": SkillSpec(
        name="wait",
        slots=(),
        phases=("wait",),
        requires=lambda world, slots: None,
        apply=_identity,
        summary="Wait without moving the arm.",
    ),
    "go_home": SkillSpec(
        name="go_home",
        slots=(),
        phases=("retreat",),
        requires=lambda world, slots: None,
        apply=_identity,
        summary="Move the arm back to its home pose.",
    ),
    "repeat": SkillSpec(
        name="repeat",
        slots=(_REPEAT,),
        phases=("wait",),
        requires=_require_repeat,
        apply=_identity,
        summary="Repeat the previous skill one, two, or three times.",
    ),
    "ask_clarification": SkillSpec(
        name="ask_clarification",
        slots=(_CLARIFY,),
        phases=("ask",),
        requires=lambda world, slots: None,
        apply=_identity,
        summary="Stop and ask the user to clarify the instruction.",
        ends_plan=True,
    ),
}


def expand(skill_step: SkillStep) -> tuple[SkillStep, ...]:
    """Turn a macro into the primitive skills the executor runs."""
    spec = REGISTRY[skill_step.skill]
    if not spec.expands_to:
        return (skill_step,)
    slots = skill_step.as_dict()
    if skill_step.skill == "pick_place":
        return (
            step("pick", object=slots["object"]),
            step("place", object=slots["object"], kind=slots["kind"], target=slots["target"]),
        )
    if skill_step.skill == "stack":
        return (
            step("pick", object=slots["moving"]),
            step("place", object=slots["moving"], kind="on", target=slots["base"]),
        )
    if skill_step.skill == "unstack":
        return (
            step("pick", object=slots["object"]),
            step("place", object=slots["object"], kind=slots["kind"], target=slots["target"]),
        )
    raise SkillError(f"{skill_step.skill} has no expansion")


def simulate(world: World, steps: tuple[SkillStep, ...] | list[SkillStep]) -> World:
    """Fold each skill's apply in order, rejecting the first illegal step."""
    current = world
    for skill_step in steps:
        for primitive in expand(skill_step):
            spec = REGISTRY[primitive.skill]
            slots = primitive.as_dict()
            reason = spec.requires(current, slots)
            if reason:
                raise SkillError(reason)
            current = spec.apply(current, slots)
    return current


def _optional_name(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _identifiers(values: object) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        return ()
    names: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip():
            names.append(value.strip())
        elif isinstance(value, dict) and isinstance(value.get("id"), str) and value["id"].strip():
            names.append(value["id"].strip())
    return tuple(dict.fromkeys(names))


def _pairs(values: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, dict):
        return ()
    pairs = [
        (str(obj).strip(), str(support).strip())
        for obj, support in values.items()
        if isinstance(obj, str) and obj.strip() and isinstance(support, str) and support.strip()
    ]
    return tuple(sorted(pairs))
