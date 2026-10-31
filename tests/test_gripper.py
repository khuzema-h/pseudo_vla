import unittest
from queue import Queue
from types import SimpleNamespace

import numpy as np

from pseudo_vla.motion import close_gripper_in_place
from pseudo_vla.scene import FINGER_GRIP_EFFORT, FINGER_GRIP_STIFFNESS, strengthen_finger_grip


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
