import unittest

from pseudo_vla.task_plan import (
    Destination,
    PickPlaceAction,
    TaskPlanError,
    unstack_actions,
    validate_task_plan,
)


class TaskPlanTests(unittest.TestCase):
    def test_validates_find_query(self):
        plan = validate_task_plan("find", "the blue block", [])
        self.assertEqual(plan.intent, "find")
        self.assertEqual(plan.query, "the blue block")
        self.assertEqual(plan.actions, ())

    def test_validates_stack_and_relative_placement_actions(self):
        plan = validate_task_plan(
            "pick_place",
            "",
            [
                {"object": "blue", "destination": {"kind": "on", "target": "red"}},
                {"object": "green", "destination": {"kind": "left_of", "target": "red"}},
            ],
        )
        self.assertEqual(
            plan.actions,
            (
                PickPlaceAction("blue", Destination("on", "red")),
                PickPlaceAction("green", Destination("left_of", "red")),
            ),
        )

    def test_validates_unstack_actions_in_given_order(self):
        plan = validate_task_plan(
            "pick_place",
            "",
            [
                {"object": "blue", "destination": {"kind": "table", "target": ""}},
                {"object": "green", "destination": {"kind": "table", "target": ""}},
            ],
        )
        self.assertEqual([action.object_id for action in plan.actions], ["blue", "green"])

    def test_derives_top_down_unstack_order_from_observed_tower(self):
        actions = unstack_actions(
            {"stacked_on": {"blue": "green", "red": "blue"}}
        )
        self.assertEqual(
            [action["object"] for action in actions],
            ["red", "blue"],
        )

    def test_rejects_ambiguous_unstack_scene(self):
        with self.assertRaisesRegex(TaskPlanError, "branching"):
            unstack_actions({"stacked_on": {"blue": "red", "green": "red"}})

    def test_rejects_unsupported_container_before_motion(self):
        with self.assertRaisesRegex(TaskPlanError, "not present in the current scene"):
            validate_task_plan(
                "pick_place",
                "",
                [
                    {
                        "object": "green",
                        "destination": {"kind": "inside", "target": "box"},
                    }
                ],
            )

    def test_rejects_invalid_and_ambiguous_plans(self):
        invalid = (
            ("find", "", []),
            ("pick_place", "extra query", []),
            (
                "pick_place",
                "",
                [{"object": "purple", "destination": {"kind": "table", "target": ""}}],
            ),
            (
                "pick_place",
                "",
                [{"object": "red", "destination": {"kind": "on", "target": "red"}}],
            ),
            (
                "pick_place",
                "",
                [
                    {"object": "red", "destination": {"kind": "on", "target": "blue"}},
                    {"object": "blue", "destination": {"kind": "on", "target": "red"}},
                ],
            ),
        )
        for intent, query, actions in invalid:
            with self.subTest(intent=intent, query=query, actions=actions):
                with self.assertRaises(TaskPlanError):
                    validate_task_plan(intent, query, actions)


if __name__ == "__main__":
    unittest.main()
