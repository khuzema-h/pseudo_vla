"""Replay a prompt suite through the planner and score exact step matches."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from unittest.mock import patch

from pseudo_vla.task_llm import interpret_task_prompt
from pseudo_vla.tev1_client import Tev1Error


def evaluate(cases, log_path: str | None = None) -> dict[str, object]:
    """Return exact-match accuracy and the number of Tev1 calls."""
    matches = 0
    calls = 0
    mismatches = []
    previous = os.environ.get("TEV1_CALL_LOG")
    if log_path:
        destination = Path(log_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("", encoding="utf-8")
        os.environ["TEV1_CALL_LOG"] = str(destination)
    try:
        for index, case in enumerate(cases):
            matched, used, detail = _run_case(case)
            calls += used
            if matched:
                matches += 1
            else:
                mismatches.append({"index": index, "instruction": case["instruction"], "detail": detail})
    finally:
        if previous is None:
            os.environ.pop("TEV1_CALL_LOG", None)
        else:
            os.environ["TEV1_CALL_LOG"] = previous
    total = len(cases)
    return {
        "exact_match": matches,
        "total": total,
        "tev1_calls": calls,
        "mean_calls_per_plan": round(calls / total, 2) if total else 0.0,
        "mismatches": mismatches,
    }


def format_baseline(report: dict[str, object]) -> str:
    return (
        f"exact_match: {report['exact_match']}/{report['total']}\n"
        f"tev1_calls: {report['tev1_calls']}\n"
        f"mean_calls_per_plan: {report['mean_calls_per_plan']}\n"
    )


def _run_case(case: dict[str, object]) -> tuple[bool, int, str]:
    script = list(case["answers"])
    seen: list[str] = []

    def fake_urlopen(request, timeout=None):
        del timeout
        payload = json.loads(request.data.decode("utf-8"))
        question_id = next(iter(payload["questions"]))
        spec = payload["questions"][question_id]
        if not script:
            raise AssertionError(f"unexpected question {question_id}")
        expected_id, expected_value = script.pop(0)
        if question_id != expected_id:
            raise AssertionError(
                f"expected {expected_id} but asked {question_id} "
                f"with options {list(spec.get('criteria') or [])}"
            )
        seen.append(question_id)
        if spec["type"] == "noul":
            answer = {"type": "noul", "noul": 0.99 if expected_value else 0.01}
        else:
            criteria = spec.get("criteria") or {}
            if expected_value not in criteria:
                raise AssertionError(
                    f"{expected_value!r} was not offered for {question_id}: {list(criteria)}"
                )
            answer = {"type": "choice", "choice": expected_value}
        return io.BytesIO(json.dumps({"answers": {question_id: answer}}).encode("utf-8"))

    expected = case["expected"]
    try:
        with patch("pseudo_vla.tev1_client.urlopen", side_effect=fake_urlopen):
            plan = interpret_task_prompt(case["instruction"], case["scene"])
    except (Tev1Error, AssertionError, ValueError) as error:
        if script:
            return False, len(seen), f"stopped early: {error}"
        token = expected.get("error")
        if token and token in str(error):
            return True, len(seen), ""
        return False, len(seen), str(error)
    if script:
        return False, len(seen), f"unused answers: {script}"
    if expected.get("error"):
        return False, len(seen), f"expected {expected['error']} but produced a plan"
    actual = {
        "intent": plan.intent,
        "steps": [{"skill": item.skill, **item.as_dict()} for item in plan.steps],
    }
    if expected.get("intent") == "find":
        actual["query"] = plan.query
    wanted = {key: value for key, value in expected.items() if key != "error"}
    if actual != wanted:
        return False, len(seen), f"got {actual}"
    return True, len(seen), ""
