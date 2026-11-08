"""Symbolic world and skill registry for one arm and one parallel gripper.

Perception and motion stay outside this module. A skill only declares the slots
Tev1 may fill, the phases the executor may run, and how the symbolic world
changes when the skill succeeds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pseudo_vla.task_plan import DESTINATION_KINDS

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

    @classmethod
    def from_scene_state(cls, scene_state: Mapping[str, object] | None) -> "World":
        state = scene_state or {}
        return cls(
            held=_optional_name(state.get("held")),
            on=_pairs(state.get("supported_by")),
            inside=_pairs(state.get("inside")),
            opened=_identifiers(state.get("opened", ())),
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
        return World(
            held=None if self.held == object_id else self.held,
            on=tuple(pair for pair in self.on if pair[0] != object_id),
            inside=tuple(pair for pair in self.inside if pair[0] != object_id),
            opened=self.opened,
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
    return World(
        held=obj,
        on=released.on,
        inside=released.inside,
        opened=released.opened,
    )


def _require_place(world: World, slots: Mapping[str, str]) -> str | None:
    obj = slots["object"]
    kind = slots["kind"]
    target = slots["target"]
    if world.held != obj:
        return "not_held"
    if kind not in DESTINATION_KINDS:
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
    return World(held=None, on=on, inside=inside, opened=released.opened)


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


_OBJECT = Slot("object", "moving_object", _object_options)
_HELD_OBJECT = Slot("object", "moving_object", _held_or_object_options)
_KIND = Slot("kind", "destination_kind", _kind_options)
_TARGET = Slot("target", "destination_target", _target_options)

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
