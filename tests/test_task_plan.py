import unittest

from pseudo_vla.task_plan import (
    Destination,
    PickPlaceAction,
    TaskPlanError,
    validate_task_plan,
)

BLOCKS = ("red", "green", "blue")
TABLE = ("table",)


class TaskPlanTests(unittest.TestCase):
    def _validate(self, intent, query, actions, **catalog):
        return validate_task_plan(
            intent,
            query,
            actions,
            available_objects=catalog.get("available_objects", BLOCKS),
            available_locations=catalog.get("available_locations", TABLE),
            available_containers=catalog.get("available_containers", ()),
        )

    def test_validates_find_query(self):
        plan = self._validate("find", "the blue block", [])
        self.assertEqual(plan.intent, "find")
        self.assertEqual(plan.query, "the blue block")
        self.assertEqual(plan.actions, ())

    def test_validates_support_and_relative_placement_actions(self):
        plan = self._validate(
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

    def test_validates_location_placement_in_given_order(self):
        plan = self._validate(
            "pick_place",
            "",
            [
                {"object": "blue", "destination": {"kind": "at", "target": "table"}},
                {"object": "green", "destination": {"kind": "at", "target": "table"}},
            ],
        )
        self.assertEqual([action.object_id for action in plan.actions], ["blue", "green"])
        self.assertTrue(all(action.destination == Destination("at", "table") for action in plan.actions))

    def test_accepts_identifiers_from_the_supplied_catalog(self):
        plan = validate_task_plan(
            "pick_place",
            "",
            [{"object": "mug", "destination": {"kind": "inside", "target": "bin"}}],
            available_objects=("mug",),
            available_containers=("bin",),
        )
        self.assertEqual(plan.actions, (PickPlaceAction("mug", Destination("inside", "bin")),))

    def test_rejects_container_absent_from_the_catalog(self):
        with self.assertRaisesRegex(TaskPlanError, "not present in the current scene"):
            self._validate(
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
                [{"object": "purple", "destination": {"kind": "at", "target": "table"}}],
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
            (
                "pick_place",
                "",
                [{"object": "red", "destination": {"kind": "at", "target": "shelf"}}],
            ),
        )
        for intent, query, actions in invalid:
            with self.subTest(intent=intent, query=query, actions=actions):
                with self.assertRaises(TaskPlanError):
                    self._validate(intent, query, actions)


if __name__ == "__main__":
    unittest.main()
