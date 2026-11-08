"""Phase authorization and the fixed failure recovery table.

Tev1 may choose the next phase of the current skill, one recovery from the
table below, or stop. Retry caps live here so a model answer cannot extend them.
"""

from __future__ import annotations

from collections.abc import Mapping

from pseudo_vla.skills import PHASES, REGISTRY
from pseudo_vla.tev1_client import Tev1Error, ask_choice

FAILURES = ("grasp_miss", "ik_fail", "jam", "slip", "drop", "timeout")

RECOVERY: dict[str, tuple[str, ...]] = {
    "grasp_miss": ("regrasp",),
    "ik_fail": ("replan",),
    "jam": ("retreat",),
    "slip": ("regrasp",),
    "drop": ("reapproach",),
    "timeout": ("retry",),
}

MAX_RETRIES: dict[str, int] = {
    "grasp_miss": 2,
    "ik_fail": 2,
    "jam": 1,
    "slip": 1,
    "drop": 1,
    "timeout": 1,
}


def choose_phase_event(
    skill: str,
    phase_idx: int,
    last_result: Mapping[str, object] | None,
    retries: Mapping[str, int] | None,
) -> str:
    """Ask Tev1 for the next phase, one legal recovery, or stop."""
    used = dict(retries or {})
    failure = _failure(last_result)
    if failure:
        if used.get(failure, 0) >= MAX_RETRIES[failure]:
            return "stop"
        options = {
            name: f"Recover from {failure} with {name}."
            for name in RECOVERY[failure]
        }
    else:
        phases = phases_for(skill)
        if not 0 <= phase_idx < len(phases):
            return "stop"
        phase = phases[phase_idx]
        options = {phase: f"Proceed with the {phase} phase of {skill}."}
    options["stop"] = "Stop before the next phase."
    return ask_choice(
        "phase_event",
        {
            "skill": skill,
            "phase_index": phase_idx,
            "expected_phase": None if failure else phases_for(skill)[phase_idx],
            "last_result": None if last_result is None else dict(last_result),
            "retries": used,
        },
        options,
    )


def phases_for(skill: str) -> tuple[str, ...]:
    spec = REGISTRY.get(skill)
    if spec is not None:
        return spec.phases
    template = PHASES.get(skill)
    if template is None:
        raise ValueError(f"Unsupported skill {skill!r}")
    return template


def _failure(last_result: Mapping[str, object] | None) -> str | None:
    if not isinstance(last_result, Mapping) or last_result.get("ok") is not False:
        return None
    failure = last_result.get("failure")
    if not isinstance(failure, str) or failure not in RECOVERY:
        raise Tev1Error(f"Phase result has unknown failure {failure!r}")
    return failure
