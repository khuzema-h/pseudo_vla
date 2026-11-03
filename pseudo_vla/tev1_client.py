"""Validated client for Ollama's Tev1 decision-model endpoint."""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TEV1_TIMEOUT = 120.0
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "tev1_questions.json"
_QUESTION_TYPES = {"choice", "noul", "score"}
_MAX_CHOICES = 24


class Tev1Error(RuntimeError):
    """Raised when a Tev1 request or answer cannot be used safely."""


def ask_choice(
    question_id: str,
    state: dict[str, object],
    criteria: dict[str, str | None],
) -> str:
    answer = ask_questions(state, [question_id], {question_id: criteria})[question_id]
    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in criteria:
        raise Tev1Error(
            f"Tev1 returned invalid choice {choice!r} for {question_id!r}; "
            f"expected one of {tuple(criteria)}"
        )
    return choice


def ask_boolean(question_id: str, state: dict[str, object]) -> bool:
    answer = ask_questions(state, [question_id])[question_id]
    probability = answer.get("noul")
    if isinstance(probability, bool):
        return probability
    if (
        not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise Tev1Error(
            f"Tev1 returned invalid boolean probability {probability!r} "
            f"for {question_id!r}"
        )
    if float(probability) == 0.5:
        raise Tev1Error(f"Tev1 was undecided on boolean question {question_id!r}")
    return float(probability) > 0.5


def ask_score(
    question_id: str,
    state: dict[str, object],
) -> float:
    answer = ask_questions(state, [question_id])[question_id]
    score = answer.get("score")
    question = _question_bank()["questions"][question_id]
    criteria = question.get("criteria")
    if (
        not isinstance(criteria, list)
        or not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= len(criteria) - 1
    ):
        raise Tev1Error(
            f"Tev1 returned invalid score {score!r} for {question_id!r}"
        )
    return float(score)


def ask_questions(
    state: dict[str, object],
    question_ids: list[str],
    criteria_overrides: dict[str, dict[str, str | None] | list[str]] | None = None,
) -> dict[str, dict[str, object]]:
    """Ask one or more bank questions about a single current state."""
    bank = _question_bank()
    specs = bank["questions"]
    if not isinstance(question_ids, list) or not 1 <= len(question_ids) <= 64:
        raise ValueError("A Tev1 request must include between one and 64 questions")
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Tev1 question ids must be unique within a request")
    if not isinstance(state, dict):
        raise ValueError("Tev1 state must be an object")

    overrides = criteria_overrides or {}
    if set(overrides) - set(question_ids):
        raise ValueError("Criteria overrides must correspond to requested questions")
    questions: dict[str, dict[str, object]] = {}
    for question_id in question_ids:
        spec = specs.get(question_id)
        if not isinstance(spec, dict):
            raise ValueError(f"Unknown Tev1 question id {question_id!r}")
        question = {
            "type": spec["type"],
            "instructions": spec["instructions"],
        }
        criteria = overrides.get(question_id, spec.get("criteria"))
        if criteria is not None:
            question["criteria"] = criteria
        _validate_question(question_id, question)
        questions[question_id] = question

    response = _post_json(
        "/v1/systemone",
        {
            "model": bank["model"],
            "state": state,
            "questions": questions,
        },
    )
    answers = response.get("answers")
    if not isinstance(answers, dict):
        raise Tev1Error("Tev1 response is missing an answers object")

    validated: dict[str, dict[str, object]] = {}
    for question_id, question in questions.items():
        answer = answers.get(question_id)
        if not isinstance(answer, dict):
            raise Tev1Error(f"Tev1 response is missing answer {question_id!r}")
        answer_type = answer.get("type")
        if answer_type is not None and answer_type != question["type"]:
            raise Tev1Error(
                f"Tev1 answer {question_id!r} has type {answer_type!r}; "
                f"expected {question['type']!r}"
            )
        validated[question_id] = answer
    return validated


def _validate_question(
    question_id: str,
    question: dict[str, object],
    *,
    allow_dynamic_criteria: bool = False,
) -> None:
    question_type = question.get("type")
    instructions = question.get("instructions")
    if question_type not in _QUESTION_TYPES or not isinstance(instructions, str) or not instructions:
        raise Tev1Error(f"Invalid Tev1 question definition for {question_id!r}")
    criteria = question.get("criteria")
    if question_type == "choice":
        if criteria is None and allow_dynamic_criteria:
            return
        if (
            not isinstance(criteria, dict)
            or not 2 <= len(criteria) <= _MAX_CHOICES
            or any(
                not isinstance(key, str)
                or not key
                or (description is not None and not isinstance(description, str))
                for key, description in criteria.items()
            )
        ):
            raise Tev1Error(
                f"Choice question {question_id!r} must have 2 to {_MAX_CHOICES} options"
            )
    elif question_type == "score":
        if (
            not isinstance(criteria, list)
            or not 2 <= len(criteria) <= _MAX_CHOICES
            or any(not isinstance(level, str) or not level for level in criteria)
        ):
            raise Tev1Error(
                f"Score question {question_id!r} must have 2 to {_MAX_CHOICES} levels"
            )
    elif criteria is not None and (
        not isinstance(criteria, dict)
        or set(criteria) != {"true", "false"}
        or any(not isinstance(value, str) for value in criteria.values())
    ):
        raise Tev1Error(
            f"Boolean question {question_id!r} criteria must describe true and false"
        )


@lru_cache(maxsize=1)
def _question_bank() -> dict[str, object]:
    try:
        bank = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Tev1Error(f"Unable to load Tev1 question bank {_CONFIG_PATH}: {error}") from error
    if not isinstance(bank, dict):
        raise Tev1Error(f"Tev1 question bank {_CONFIG_PATH} must contain a JSON object")
    model = bank.get("model")
    max_questions = bank.get("max_questions")
    questions = bank.get("questions")
    if (
        not isinstance(model, str)
        or not model
        or max_questions != 64
        or not isinstance(questions, dict)
        or not 1 <= len(questions) <= 64
    ):
        raise Tev1Error(
            "Tev1 question bank must define a model, max_questions=64, and 1 to 64 questions"
        )
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not isinstance(question, dict):
            raise Tev1Error("Tev1 question bank entries must be named objects")
        _validate_question(question_id, question, allow_dynamic_criteria=True)
    return bank


def _ollama_base_url() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip().rstrip("/")
    if "://" not in host:
        host = f"http://{host}"
    return host


def _post_json(path: str, payload: dict[str, object]) -> dict[str, object]:
    try:
        body = json.dumps(payload).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Tev1Error(f"Tev1 request payload is not valid JSON: {error}") from error
    request = Request(
        f"{_ollama_base_url()}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=TEV1_TIMEOUT) as response:
            result = json.loads(response.read())
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise Tev1Error(f"Ollama {path} returned HTTP {error.code}: {details}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise Tev1Error(f"Ollama request to {path} failed: {error}") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise Tev1Error(f"Ollama {path} returned invalid JSON") from error
    if not isinstance(result, dict):
        raise Tev1Error(f"Ollama {path} returned a non-object response")
    return result
