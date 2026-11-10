import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pseudo_vla import controller
from pseudo_vla.motion import close_gripper_in_place
from pseudo_vla.motion import execute_pose
from pseudo_vla.scene import FINGER_GRIP_EFFORT, FINGER_GRIP_STIFFNESS, strengthen_finger_grip
from pseudo_vla.servo import GripperRelativeObjectTracker


class GripperRelativeObjectTrackerTests(unittest.TestCase):
    def test_tracks_object_motion_relative_to_hand(self):
        tracker = GripperRelativeObjectTracker(max_relative_drift=0.05, lost_frames=2)
        hand = np.array([0.0, 0.0, 0.3])
        object_points = np.array(
            [
                [0.01, 0.0, 0.3],
                [0.02, 0.0, 0.3],
                [0.01, 0.01, 0.3],
                [0.02, 0.01, 0.3],
            ]
        )

        self.assertTrue(tracker.seed(hand, object_points, observation_id=1))
        self.assertTrue(tracker.update(hand + [0.0, 0.0, 0.1], object_points + [0.0, 0.0, 0.1], 2))
        self.assertIsNone(tracker.update(hand + [0.0, 0.0, 0.2], object_points + [0.0, 0.0, 0.2], 2))

    def test_reports_lost_only_after_repeated_relative_drift(self):
        tracker = GripperRelativeObjectTracker(max_relative_drift=0.05, lost_frames=2)
        object_points = np.array(
            [
                [0.01, 0.0, 0.3],
                [0.02, 0.0, 0.3],
                [0.01, 0.01, 0.3],
                [0.02, 0.01, 0.3],
            ]
        )
        self.assertTrue(tracker.seed(np.array([0.0, 0.0, 0.3]), object_points))

        self.assertIsNone(tracker.update(np.array([0.0, 0.0, 0.4]), object_points))
        self.assertFalse(tracker.lost)
        self.assertFalse(tracker.update(np.array([0.0, 0.0, 0.5]), object_points))
        self.assertTrue(tracker.lost)
        self.assertTrue(tracker.update(np.array([0.0, 0.0, 0.5]), object_points + [0.0, 0.0, 0.2]))
        self.assertFalse(tracker.lost)

    def test_execute_pose_stops_when_gripper_tracking_reports_loss(self):
        class _Servo:
            trajectory = object()

            def clear(self):
                self.trajectory = None

            def update_goal(self, *_args):
                self.trajectory = object()

            def command(self):
                return None

            def hand_position(self, _joints):
                return np.array([0.0, 0.0, 0.0])

        targets = []
        scene = SimpleNamespace(
            joint_positions=lambda: {},
            step=lambda: None,
            read_sensors=lambda: (np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2)), {}),
            set_joint_targets=lambda joints: targets.append(joints),
        )
        servo = _Servo()
        result = execute_pose(
            scene,
            servo,
            SimpleNamespace(position=np.array([1.0, 0.0, 0.0])),
            SimpleNamespace(push_frame=lambda _rgb: None),
            Queue(),
            2,
            lambda _message: None,
            None,
            tracking_check=lambda *_args: False,
        )

        self.assertEqual(result, (False, None))
        self.assertIsNone(servo.trajectory)
        self.assertEqual(targets, [{}])

    def test_controller_checks_tracked_points_against_the_current_hand_pose(self):
        tracker = GripperRelativeObjectTracker(max_relative_drift=0.05, lost_frames=2)
        servo = SimpleNamespace(gripper_object_tracker=tracker)
        object_points = np.array(
            [
                [0.01, 0.0, 0.3],
                [0.02, 0.0, 0.3],
                [0.01, 0.01, 0.3],
                [0.02, 0.01, 0.3],
            ]
        )
        scene = SimpleNamespace(
            hand_path="/World/hand",
            camera_path="/World/camera",
            read_sensors=lambda: (np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2)), {}),
        )
        hand_position = np.array([0.0, 0.0, 0.3])
        perception = SimpleNamespace(
            query_groups={"moving": slice(0, 4)},
            tracking_generation=1,
            push_frame=lambda _rgb: (np.zeros((4, 2)), np.ones(4, dtype=bool)),
            latest_tracks=lambda: (np.zeros((4, 2)), np.ones(4, dtype=bool)),
            tracked_group_points=lambda *_args: object_points.copy(),
        )

        with patch(
            "pseudo_vla.perception.prim_world_pose",
            side_effect=lambda _path: (
                hand_position.copy(),
                np.array([1.0, 0.0, 0.0, 0.0]),
            ),
        ):
            self.assertTrue(controller._seed_gripper_object_tracking(scene, perception, servo, lambda _: None))
            hand_position[2] += 0.1
            perception.tracking_generation = 2
            self.assertIsNone(
                controller._check_gripper_object_tracking(
                    scene, perception, servo, np.ones((2, 2)), {}
                )
            )
            perception.tracking_generation = 3
            self.assertFalse(
                controller._check_gripper_object_tracking(
                    scene, perception, servo, np.ones((2, 2)), {}
                )
            )


class CloseGripperTests(unittest.TestCase):
    def test_fingers_finish_closing_before_the_arm_can_move(self):
        arm = {f"panda_joint{index}": 0.1 * index for index in range(1, 8)}
        state = {**arm, "panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04}
        commands = []

        def set_joint_targets(command):
            commands.append(dict(command))
            for name, value in command.items():
                if name.startswith("panda_finger"):
                    state[name] = max(0.012, state[name] - 0.01)
                else:
                    state[name] = value

        scene = SimpleNamespace(
            joint_positions=lambda: dict(state),
            set_joint_targets=set_joint_targets,
            step=lambda: None,
            read_sensors=lambda: (np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2)), {}),
            gripper_closed=False,
        )
        perception = SimpleNamespace(push_frame=lambda _rgb: None)

        result = close_gripper_in_place(scene, perception, Queue(), None, max_steps=80)

        self.assertIsNone(result)
        self.assertTrue(scene.gripper_closed)
        self.assertLess(len(commands), 80)
        self.assertLess(state["panda_finger_joint1"], 0.02)
        for command in commands:
            for name, value in arm.items():
                self.assertAlmostEqual(command[name], value)


class FingerDriveTests(unittest.TestCase):
    def test_finger_drive_uses_a_firm_force_limit(self):
        calls = {}

        class _Arm:
            dof_names = [
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
                "panda_finger_joint1",
                "panda_finger_joint2",
            ]

            def set_dof_gains(self, stiffnesses, dampings, dof_indices):
                calls["gains"] = (stiffnesses, dampings, list(dof_indices))

            def set_dof_max_efforts(self, max_efforts, dof_indices):
                calls["efforts"] = (max_efforts, list(dof_indices))

        strengthen_finger_grip(_Arm())

        self.assertEqual(calls["gains"][2], [7, 8])
        self.assertEqual(calls["gains"][0][0], FINGER_GRIP_STIFFNESS)
        self.assertEqual(calls["efforts"][0], [FINGER_GRIP_EFFORT, FINGER_GRIP_EFFORT])
        self.assertGreaterEqual(FINGER_GRIP_EFFORT, 15.0)
        self.assertLessEqual(FINGER_GRIP_EFFORT, 40.0)
        self.assertLess(FINGER_GRIP_STIFFNESS * 0.01, 40.0)


if __name__ == "__main__":
    unittest.main()
