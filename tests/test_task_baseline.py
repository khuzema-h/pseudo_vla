"""Characterization tests for the Tev1 router before the skill registry.

These freeze the plans and the question order of the current router.
Later milestones must keep them passing.
"""

import unittest
from unittest.mock import patch

from pseudo_vla import task_llm
from pseudo_vla.task_plan import Destination, PickPlaceAction
from pseudo_vla.tev1_client import Tev1Error

KITCHEN = {
    "objects": [{"id": "plate"}, {"id": "mug"}],
    "locations": ["counter"],
    "containers": [],
    "supported_by": {},
}
BLOCKS = {
    "objects": ["blue", "red", "green"],
    "locations": ["table"],
    "containers": [],
    "supported_by": {"blue": "red"},
}


class RouterBaselineTests(unittest.TestCase):
    def _script(self, answers):
        calls = []

        def ask_choice(question_id, state, criteria):
            choice = answers[question_id].pop(0)
            calls.append((question_id, choice, tuple(criteria)))
            self.assertIn(choice, criteria)
            self.assertLessEqual(len(criteria), 24)
            return choice

        def ask_boolean(question_id, state):
            choice = answers[question_id].pop(0)
            calls.append((question_id, choice, ()))
            return choice

        return calls, ask_choice, ask_boolean

    def test_find_selects_only_the_named_object(self):
        calls, ask_choice, ask_boolean = self._script(
            {
                "task_intent": ["find"],
                "find_target": ["mug"],
            }
        )
        with (
            patch.object(task_llm, "ask_choice", side_effect=ask_choice),
            patch.object(task_llm, "ask_boolean", side_effect=ask_boolean),
        ):
            plan = task_llm.interpret_task_prompt("where is the mug", KITCHEN)
        self.assertEqual(plan.intent, "find")
        self.assertEqual(plan.query, "mug")
        self.assertEqual(plan.actions, ())
        self.assertEqual([question for question, *_rest in calls], ["task_intent", "find_target"])

    def test_hold_plate_and_place_it_on_the_mug(self):
        calls, ask_choice, ask_boolean = self._script(
            {
                "task_intent": ["pick_place"],
                "plan_more": [True, False],
                "moving_object": ["plate"],
                "destination_kind": ["on"],
            }
        )
        with (
            patch.object(task_llm, "ask_choice", side_effect=ask_choice),
            patch.object(task_llm, "ask_boolean", side_effect=ask_boolean),
        ):
            plan = task_llm.interpret_task_prompt("hold plate and place it on the mug", KITCHEN)
        self.assertEqual(
            plan.actions,
            (PickPlaceAction("plate", Destination("on", "mug")),),
        )
        self.assertEqual(
            [question for question, *_rest in calls],
            ["task_intent", "plan_more", "moving_object", "destination_kind", "plan_more"],
        )
        kind_options = calls[3][2]
        self.assertIn("none", calls[2][2])
        self.assertIn("unsupported", kind_options)
        self.assertNotIn("destination_target", [question for question, *_rest in calls])

    def test_multi_action_plan_keeps_order_and_drops_moved_objects(self):
        calls, ask_choice, ask_boolean = self._script(
            {
                "task_intent": ["pick_place"],
                "plan_more": [True, True, False],
                "moving_object": ["blue", "green"],
                "destination_kind": ["on", "left_of"],
                "destination_target": ["red", "red"],
            }
        )
        with (
            patch.object(task_llm, "ask_choice", side_effect=ask_choice),
            patch.object(task_llm, "ask_boolean", side_effect=ask_boolean),
        ):
            plan = task_llm.interpret_task_prompt("stack blue on red then move green left", BLOCKS)
        self.assertEqual(
            plan.actions,
            (
                PickPlaceAction("blue", Destination("on", "red")),
                PickPlaceAction("green", Destination("left_of", "red")),
            ),
        )
        moving_calls = [options for question, _choice, options in calls if question == "moving_object"]
        self.assertIn("blue", moving_calls[0])
        self.assertNotIn("blue", moving_calls[1])
        self.assertIn("green", moving_calls[1])

    def test_unsupported_intent_stops_before_any_action_question(self):
        calls, ask_choice, ask_boolean = self._script(
            {"task_intent": ["unsupported"]}
        )
        with (
            patch.object(task_llm, "ask_choice", side_effect=ask_choice),
            patch.object(task_llm, "ask_boolean", side_effect=ask_boolean),
        ):
            with self.assertRaisesRegex(Tev1Error, "unsupported"):
                task_llm.interpret_task_prompt("fly the mug to the moon", KITCHEN)
        self.assertEqual([question for question, *_rest in calls], ["task_intent"])

    def test_choose_task_event_proceeds_or_stops(self):
        action = PickPlaceAction("plate", Destination("on", "mug"))
        seen = {}

        def ask_choice(question_id, state, criteria):
            seen["criteria"] = criteria
            seen["event"] = state["phase"]
            return criteria_choice

        for criteria_choice, expected in (("grasp", "grasp"), ("stop", "stop")):
            with patch.object(task_llm, "ask_choice", side_effect=ask_choice):
                decision = task_llm.choose_task_event("grasp", action, 0, {"ok": True})
            self.assertEqual(decision, expected)
            self.assertEqual(set(seen["criteria"]), {"grasp", "stop"})
            self.assertEqual(seen["event"], "authorize_grasp")


if __name__ == "__main__":
    unittest.main()
