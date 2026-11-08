import unittest
from unittest.mock import patch

from pseudo_vla import phases, task_workflow
from pseudo_vla.phase_handlers import FakeBackend, PhaseAdapters, PhaseResult, backend_for
from pseudo_vla.skills import step
from pseudo_vla.task_plan import TaskPlan


class PhaseHandlerTests(unittest.TestCase):
    def test_phase_names_stay_behind_their_backends(self):
        self.assertEqual(backend_for("find"), "segment")
        self.assertEqual(backend_for("verify_placement"), "segment")
        self.assertEqual(backend_for("approach"), "tracker")
        self.assertEqual(backend_for("contact"), "tracker")
        self.assertEqual(backend_for("verify_effect"), "tracker")
        self.assertEqual(backend_for("plan"), "curobo")
        self.assertEqual(backend_for("move_above"), "curobo")
        self.assertEqual(backend_for("retreat"), "curobo")
        self.assertEqual(backend_for("pre_insert"), "curobo")
        self.assertEqual(backend_for("grasp"), "ooal")
        self.assertEqual(backend_for("rotate"), "ooal")
        self.assertEqual(backend_for("verify_grasp"), "gripper")
        self.assertEqual(backend_for("guarded_insert"), "gripper")

    def test_adapters_call_only_the_injected_backend(self):
        seen = []

        def grasp(step, phase, world):
            seen.append((phase, step.skill))
            return PhaseResult(ok=True, data={"backend": "ooal"})

        adapters = PhaseAdapters(grasp=grasp)
        result = adapters.run(step("rotate", object="knob"), "rotate", None)
        self.assertTrue(result.ok)
        self.assertEqual(seen, [("rotate", "rotate")])
        blocked = adapters.run(step("pick", object="mug"), "approach", None)
        self.assertFalse(blocked.ok)
        self.assertEqual(blocked.failure, "timeout")

    def test_six_skills_run_end_to_end_on_a_fake_backend(self):
        plan = TaskPlan(
            intent="skill_sequence",
            steps=(
                step("open", container="drawer"),
                step("pick", object="mug"),
                step("pour", object="mug", target="bowl", transport_constraint="upright"),
                step("place", object="mug", kind="at", target="counter"),
                step("push", object="mug", direction="left"),
                step("wait"),
            ),
        )
        backend = FakeBackend()

        def ask_choice(question_id, state, criteria):
            self.assertEqual(question_id, "phase_event")
            self.assertIn(state["expected_phase"], criteria)
            return state["expected_phase"]

        with patch.object(phases, "ask_choice", side_effect=ask_choice):
            complete = task_workflow.run_task_plan(
                plan,
                lambda _message: None,
                lambda _query: self.fail("find must stay on the legacy path"),
                lambda *_args: self.fail("legacy motion must not run"),
                backend,
            )
        self.assertTrue(complete)
        expected = [
            (skill_step.skill, phase)
            for skill_step in plan.steps
            for phase in phases.phases_for(skill_step.skill)
        ]
        self.assertEqual(backend.calls, expected)
        self.assertEqual(
            {skill for skill, _phase in expected},
            {"open", "pick", "pour", "place", "push", "wait"},
        )


if __name__ == "__main__":
    unittest.main()
