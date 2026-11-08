"""Phase handlers. Perception and motion stay behind the callables injected here.

Nothing in this module imports SAM 2, CoTracker, cuRobo, or OOAL. The controller
passes those systems in as callables when a real scene is running. Tests pass
a FakeBackend instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from pseudo_vla.skills import SkillStep, World

BackendCall = Callable[[SkillStep, str, World], "PhaseResult"]

SEGMENT_PHASES = {"find", "verify_placement"}
TRACKER_PHASES = {"approach", "contact", "verify_effect"}
CUROBO_PHASES = {"plan", "move_above", "retreat", "pre_insert"}
GRASP_PHASES = {"grasp", "rotate"}
GRIPPER_PHASES = {"verify_grasp", "guarded_insert"}


@dataclass(frozen=True)
class PhaseResult:
    """The only result shape a phase handler may return."""

    ok: bool
    failure: str | None = None
    data: dict[str, object] = field(default_factory=dict)


class PhaseHandler(Protocol):
    def run(self, step: SkillStep, phase: str, world: World) -> PhaseResult:
        """Run one phase of a skill against the current symbolic world."""


def backend_for(phase: str) -> str:
    """Name the system that owns a phase, without calling that system."""
    if phase in GRIPPER_PHASES:
        return "gripper"
    if phase in GRASP_PHASES:
        return "ooal"
    if phase in CUROBO_PHASES:
        return "curobo"
    if phase in TRACKER_PHASES:
        return "tracker"
    if phase in SEGMENT_PHASES or phase.startswith("verify_"):
        return "segment"
    return "scripted"


class FakeBackend:
    """A phase handler that succeeds, or replays a script of results."""

    def __init__(self, script: list[PhaseResult] | None = None) -> None:
        self.script = list(script or [])
        self.calls: list[tuple[str, str]] = []

    def run(self, step: SkillStep, phase: str, world: World) -> PhaseResult:
        del world
        self.calls.append((step.skill, phase))
        if self.script:
            return self.script.pop(0)
        return PhaseResult(ok=True, data={"phase": phase, "backend": "fake"})


class PhaseAdapters:
    """Route a phase to an injected backend. Missing backends fail closed."""

    def __init__(
        self,
        *,
        segment: BackendCall | None = None,
        tracker: BackendCall | None = None,
        curobo: BackendCall | None = None,
        grasp: BackendCall | None = None,
        gripper: BackendCall | None = None,
    ) -> None:
        self._backends = {
            "segment": segment,
            "tracker": tracker,
            "curobo": curobo,
            "ooal": grasp,
            "gripper": gripper,
        }

    def run(self, step: SkillStep, phase: str, world: World) -> PhaseResult:
        group = backend_for(phase)
        call = self._backends.get(group)
        if call is None:
            return PhaseResult(
                ok=False,
                failure="timeout",
                data={"phase": phase, "backend": group},
            )
        return call(step, phase, world)
