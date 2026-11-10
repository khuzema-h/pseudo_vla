import unittest
from types import SimpleNamespace

from pseudo_vla.stage_control import (
    ControllerStageState,
    StageRequest,
    parse_stage_instruction,
    plan_stage_prerequisites,
)
from pseudo_vla.task_plan import Destination


class StageControlTests(unittest.TestCase):
    def setUp(self):
        self.objects = ("mug", "plate")
        self.locations = ("counter",)

    def test_parse_stage_commands(self):
        # Lift / pick commands
        req = parse_stage_instruction("lift mug", self.objects, self.locations)
        self.assertIsNotNone(req)
        self.assertEqual(req.stage, "lift")
        self.assertEqual(req.object_id, "mug")

        req = parse_stage_instruction("hold and lift mug", self.objects, self.locations)
        self.assertIsNotNone(req)
        self.assertEqual(req.stage, "lift")
        self.assertEqual(req.object_id, "mug")

        req = parse_stage_instruction("pick up the mug", self.objects, self.locations)
        self.assertIsNotNone(req)
        self.assertEqual(req.stage, "lift")
        self.assertEqual(req.object_id, "mug")

        req = parse_stage_instruction("lift", self.objects, self.locations)
        self.assertIsNotNone(req)
        self.assertEqual(req.stage, "lift")
        self.assertIsNone(req.object_id)

        # Approach commands
        req = parse_stage_instruction("approach mug", self.objects, self.locations)
        self.assertIsNotNone(req)
        self.assertEqual(req.stage, "approach")
        self.assertEqual(req.object_id, "mug")

        # Grasp commands
        req = parse_stage_instruction("grasp mug", self.objects, self.locations)
        self.assertIsNotNone(req)
        self.assertEqual(req.stage, "grasp")
        self.assertEqual(req.object_id, "mug")

        req = parse_stage_instruction("hold mug", self.objects, self.locations)
        self.assertIsNotNone(req)
        self.assertEqual(req.stage, "grasp")
        self.assertEqual(req.object_id, "mug")

        # Place commands
        req = parse_stage_instruction("place on the plate", self.objects, self.locations)
        self.assertIsNotNone(req)
        self.assertEqual(req.stage, "place")
        self.assertEqual(req.destination, Destination(kind="on", target="plate"))

        req = parse_stage_instruction("place at counter", self.objects, self.locations)
        self.assertIsNotNone(req)
        self.assertEqual(req.stage, "place")
        self.assertEqual(req.destination, Destination(kind="at", target="counter"))

        req = parse_stage_instruction("place mug to the left of plate", self.objects, self.locations)
        self.assertIsNotNone(req)
        self.assertEqual(req.stage, "place")
        self.assertEqual(req.object_id, "mug")
        self.assertEqual(req.destination, Destination(kind="left_of", target="plate"))

        # Release / retreat
        req = parse_stage_instruction("open gripper", self.objects, self.locations)
        self.assertIsNotNone(req)
        self.assertEqual(req.stage, "release")

        req = parse_stage_instruction("put mug down", self.objects, self.locations)
        self.assertIsNotNone(req)
        self.assertEqual(req.stage, "release")
        self.assertEqual(req.object_id, "mug")

        req = parse_stage_instruction("place the mug down", self.objects, self.locations)
        self.assertIsNotNone(req)
        self.assertEqual(req.stage, "release")
        self.assertEqual(req.object_id, "mug")

        req = parse_stage_instruction("retreat", self.objects, self.locations)
        self.assertIsNotNone(req)
        self.assertEqual(req.stage, "retreat")

        # Multi-stage pick-and-place should return None so existing macro handles it
        req = parse_stage_instruction("hold mug and place on plate", self.objects, self.locations)
        self.assertIsNone(req)

        req = parse_stage_instruction("hold mug and place it on the plate", self.objects, self.locations)
        self.assertIsNone(req)

    def test_block_scene_commands(self):
        blocks = ("blue", "red", "green")
        locations = ("table",)

        req = parse_stage_instruction("lift blue", blocks, locations)
        self.assertIsNotNone(req)
        self.assertEqual(req.stage, "lift")
        self.assertEqual(req.object_id, "blue")

        req = parse_stage_instruction("place on red", blocks, locations)
        self.assertIsNotNone(req)
        self.assertEqual(req.stage, "place")
        self.assertEqual(req.destination, Destination(kind="on", target="red"))

        req = parse_stage_instruction("place green to the right of red", blocks, locations)
        self.assertIsNotNone(req)
        self.assertEqual(req.stage, "place")
        self.assertEqual(req.object_id, "green")
        self.assertEqual(req.destination, Destination(kind="right_of", target="red"))

    def test_prerequisite_planning_from_home(self):
        state = ControllerStageState()

        # Lift mug from home: requires approach and grasp first
        req = StageRequest(stage="lift", object_id="mug")
        prereqs, error = plan_stage_prerequisites(req, state)
        self.assertIsNone(error)
        self.assertEqual(prereqs, ["approach", "grasp"])

        # Grasp mug from home: requires approach first
        req = StageRequest(stage="grasp", object_id="mug")
        prereqs, error = plan_stage_prerequisites(req, state)
        self.assertIsNone(error)
        self.assertEqual(prereqs, ["approach"])

        # Approach mug from home: no prerequisites
        req = StageRequest(stage="approach", object_id="mug")
        prereqs, error = plan_stage_prerequisites(req, state)
        self.assertIsNone(error)
        self.assertEqual(prereqs, [])

    def test_prerequisite_planning_when_already_approached(self):
        state = ControllerStageState()
        state.stage = "approached"
        state.target_object = "mug"

        # Lift mug: only needs grasp (already approached)
        req = StageRequest(stage="lift", object_id="mug")
        prereqs, error = plan_stage_prerequisites(req, state)
        self.assertIsNone(error)
        self.assertEqual(prereqs, ["grasp"])

        # Grasp mug: no prerequisites (already approached)
        req = StageRequest(stage="grasp", object_id="mug")
        prereqs, error = plan_stage_prerequisites(req, state)
        self.assertIsNone(error)
        self.assertEqual(prereqs, [])

    def test_prerequisite_planning_when_already_grasped(self):
        state = ControllerStageState()
        state.stage = "grasped"
        state.held_object = "mug"
        state.target_object = "mug"

        # Lift mug: no prerequisites (already grasped)
        req = StageRequest(stage="lift", object_id="mug")
        prereqs, error = plan_stage_prerequisites(req, state)
        self.assertIsNone(error)
        self.assertEqual(prereqs, [])

        # Place on plate: only needs lift
        req = StageRequest(stage="place", object_id="mug", destination=Destination("on", "plate"))
        prereqs, error = plan_stage_prerequisites(req, state)
        self.assertIsNone(error)
        self.assertEqual(prereqs, ["lift"])

    def test_prerequisite_planning_when_already_lifted(self):
        state = ControllerStageState()
        state.stage = "lifted"
        state.held_object = "mug"
        state.target_object = "mug"

        # Place on plate: no prerequisites (already held and lifted)
        req = StageRequest(stage="place", destination=Destination("on", "plate"))
        prereqs, error = plan_stage_prerequisites(req, state)
        self.assertIsNone(error)
        self.assertEqual(prereqs, [])

        req = StageRequest(stage="release", object_id="mug")
        prereqs, error = plan_stage_prerequisites(req, state)
        self.assertIsNone(error)
        self.assertEqual(prereqs, [])

        req = StageRequest(stage="release", object_id="plate")
        prereqs, error = plan_stage_prerequisites(req, state)
        self.assertEqual(prereqs, [])
        self.assertIn("mug is held", error)

        # Trying to approach or lift another object fails because gripper is full
        req = StageRequest(stage="approach", object_id="plate")
        prereqs, error = plan_stage_prerequisites(req, state)
        self.assertIn("already held", error)

    def test_stage_history_preserves_transition_context_across_reset(self):
        state = ControllerStageState()
        state.stage = "lifted"
        state.held_object = "mug"
        state.context = {
            "moving_item": SimpleNamespace(center_world=[0.1, 0.2, 0.3]),
            "known_objects": {"plate": object()},
        }

        state.record_transition("lift", "mug", "grasped")
        state.reset()

        self.assertEqual(len(state.history), 1)
        transition = state.history[0]
        self.assertEqual(transition["stage_before"], "grasped")
        self.assertEqual(transition["stage_after"], "lifted")
        self.assertEqual(transition["held_object"], "mug")
        self.assertEqual(transition["moving_center_world"], [0.1, 0.2, 0.3])
        self.assertEqual(transition["known_objects"], ("plate",))


if __name__ == "__main__":
    unittest.main()
