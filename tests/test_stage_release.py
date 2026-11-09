import queue
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from pseudo_vla import controller
from pseudo_vla.stage_control import ControllerStageState


class StageReleaseTests(unittest.TestCase):
    def test_release_surface_chooses_highest_overlapping_surface_below_object(self):
        scene = SimpleNamespace(
            object_paths={
                "held": "/World/held",
                "support": "/World/support",
                "far": "/World/far",
            }
        )
        bounds = {
            "/World/support": (
                np.array([-0.10, -0.10, 0.10]),
                np.array([0.10, 0.10, 0.30]),
            ),
            "/World/far": (
                np.array([0.50, 0.50, 0.10]),
                np.array([0.70, 0.70, 0.40]),
            ),
        }

        with patch("pseudo_vla.scene.world_collision_bounds", side_effect=bounds.__getitem__, create=True):
            support = controller._release_surface_below(
                scene,
                "held",
                np.array([0.0, 0.0, 0.50]),
                np.array([0.10, 0.10, 0.10]),
                Mock(),
            )

        self.assertEqual(support, ("support", 0.30))

    def test_release_moves_above_and_to_surface_before_opening(self):
        scene = SimpleNamespace(
            gripper_closed=True,
            hand_path="/World/hand",
            base_path="/World/base",
            object_paths={},
            last_observed={},
        )
        state = ControllerStageState()
        state.held_object = "mug"
        state.target_object = "mug"
        state.stage = "lifted"
        state.context = {
            "moving_item": SimpleNamespace(center_world=np.array([0.0, 0.0, 0.20])),
            "moving_dims": np.array([0.10, 0.10, 0.10]),
            "grasp_world": np.array([0.0, 0.0, 0.30]),
            "grasp_goal": SimpleNamespace(quaternion=np.array([1.0, 0.0, 0.0, 0.0])),
            "moving_spec": {"name": "mug"},
        }
        servo = SimpleNamespace(
            update_block_positions=Mock(),
            set_obstacle_enabled=Mock(),
        )
        actions = []

        def execute_pose(_scene, _servo, goal, *_args, **kwargs):
            actions.append(("move", goal, kwargs.get("release_when_supported", False)))
            return True, None

        def open_gripper(_scene, *_args):
            actions.append(("open",))
            scene.gripper_closed = False

        def world_pose(path):
            if path == scene.hand_path:
                return np.array([0.0, 0.0, 0.50]), np.array([1.0, 0.0, 0.0, 0.0])
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

        with (
            patch("pseudo_vla.controller._release_surface_below", return_value=("table", 0.01)),
            patch("pseudo_vla.controller._execute_pose", side_effect=execute_pose),
            patch("pseudo_vla.controller._open_gripper_for_next_command", side_effect=open_gripper),
            patch("pseudo_vla.perception.prim_world_pose", side_effect=world_pose),
            patch(
                "pseudo_vla.perception.world_to_base",
                side_effect=lambda points, _base, _quat: points,
            ),
        ):
            result = controller._stage_release(
                scene, Mock(), servo, queue.Queue(), 10, Mock(), Mock(), None, state
            )

        self.assertTrue(result)
        self.assertEqual([action[0] for action in actions], ["move", "move", "open", "move"])
        self.assertTrue(actions[1][2])
        self.assertEqual(state.stage, "released")
        self.assertIsNone(state.held_object)
        self.assertIsNone(state.context)
        self.assertFalse(scene.gripper_closed)


if __name__ == "__main__":
    unittest.main()
