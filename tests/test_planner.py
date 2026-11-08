import unittest
from unittest.mock import patch

from pseudo_vla import planner, task_llm
from pseudo_vla.skills import step
from pseudo_vla.task_plan import Destination, PickPlaceAction
from pseudo_vla.tev1_client import Tev1Error


def _script(test, answers):
    calls = []

    def ask_choice(question_id, state, criteria):
        choice = answers[question_id].pop(0)
        calls.append(question_id)
        test.assertIn(choice, criteria)
        test.assertLessEqual(len(criteria), 24)
        return choice

    def ask_boolean(question_id, state):
        choice = answers[question_id].pop(0)
        calls.append(question_id)
        return choice

    return calls, ask_choice, ask_boolean


class PlannerTests(unittest.TestCase):
    def _plan(self, scene, answers, instruction="pick up the plate"):
        calls, ask_choice, ask_boolean = _script(self, answers)
        with (
            patch.object(planner, "ask_choice", side_effect=ask_choice),
            patch.object(planner, "ask_boolean", side_effect=ask_boolean),
        ):
            plan = planner.plan_steps(instruction, scene)
        return plan, calls

    def test_blocked_pick_inserts_a_clear_without_asking(self):
        plan, calls = self._plan(
            {
                "objects": ["mug", "plate"],
                "locations": ["counter"],
                "supported_by": {"mug": "plate"},
            },
            {
                "plan_more": [True, False],
                "skill_next": ["pick"],
                "moving_object": ["plate"],
            },
        )
        self.assertEqual(
            plan.steps,
            (
                step("pick", object="mug"),
                step("place", object="mug", kind="at", target="counter"),
                step("pick", object="plate"),
            ),
        )
        self.assertNotIn("prerequisite_resolution", calls)

    def test_busy_gripper_parks_the_held_object(self):
        plan, calls = self._plan(
            {
                "objects": ["mug", "plate"],
                "locations": ["counter"],
                "held": "mug",
            },
            {
                "plan_more": [True, False],
                "skill_next": ["pick"],
            },
            instruction="pick up the plate",
        )
        self.assertEqual(
            plan.steps,
            (
                step("place", object="mug", kind="at", target="counter"),
                step("pick", object="plate"),
            ),
        )
        self.assertNotIn("prerequisite_resolution", calls)

    def test_two_blockers_ask_which_one_to_clear_first(self):
        plan, calls = self._plan(
            {
                "objects": ["mug", "bowl", "plate"],
                "locations": ["counter"],
                "supported_by": {"mug": "plate", "bowl": "plate"},
            },
            {
                "plan_more": [True, False],
                "skill_next": ["pick"],
                "moving_object": ["plate"],
                "prerequisite_resolution": ["clear:mug"],
            },
        )
        self.assertEqual(plan.steps[0], step("pick", object="mug"))
        self.assertEqual(plan.steps[-1], step("pick", object="plate"))
        self.assertEqual(calls.count("prerequisite_resolution"), 1)
        self.assertEqual(plan.steps[2], step("pick", object="bowl"))

    def test_none_and_unsupported_stop_the_plan(self):
        with self.assertRaisesRegex(Tev1Error, "unsupported"):
            self._plan(
                {"objects": ["mug"], "locations": ["counter"]},
                {"plan_more": [True], "skill_next": ["unsupported"]},
            )
        with self.assertRaisesRegex(Tev1Error, "none"):
            self._plan(
                {"objects": ["mug", "plate"], "locations": ["counter"]},
                {
                    "plan_more": [True],
                    "skill_next": ["pick"],
                    "moving_object": ["none"],
                },
            )

    def test_a_single_real_option_skips_the_call(self):
        plan, calls = self._plan(
            {"objects": ["mug"], "locations": ["counter"]},
            {
                "plan_more": [True, False],
                "skill_next": ["place"],
            },
            instruction="put it on the counter",
        )
        self.assertNotIn("moving_object", calls)
        self.assertNotIn("destination_target", calls)
        self.assertEqual(plan.steps[0].skill, "pick")
        self.assertEqual(
            plan.steps[1],
            step("place", object="mug", kind="at", target="counter"),
        )

    def test_legacy_pick_place_plans_keep_their_actions_and_record_steps(self):
        answers = {
            "task_intent": ["pick_place"],
            "plan_more": [True, False],
            "moving_object": ["plate"],
            "destination_kind": ["on"],
        }
        calls, ask_choice, ask_boolean = _script(self, answers)
        scene = {
            "objects": [{"id": "plate"}, {"id": "mug"}],
            "locations": ["counter"],
            "containers": [],
        }
        with (
            patch.object(task_llm, "ask_choice", side_effect=ask_choice),
            patch.object(task_llm, "ask_boolean", side_effect=ask_boolean),
        ):
            plan = task_llm.interpret_task_prompt("hold plate and place it on the mug", scene)
        self.assertEqual(plan.actions, (PickPlaceAction("plate", Destination("on", "mug")),))
        self.assertEqual(
            plan.steps,
            (
                step("pick", object="plate"),
                step("place", object="plate", kind="on", target="mug"),
            ),
        )
        self.assertEqual(
            calls,
            ["task_intent", "plan_more", "moving_object", "destination_kind", "plan_more"],
        )


if __name__ == "__main__":
    unittest.main()
