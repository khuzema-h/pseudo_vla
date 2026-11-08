import unittest
from unittest.mock import patch

import numpy as np

from pseudo_vla.geometry import (
    TOP_DOWN_WXYZ,
    adaptive_approach_pose,
    adaptive_grasp_orientation,
    base_to_world,
    base_to_world_quat,
    collision_vertical_gap,
    descent_between_masks,
    descent_between_points,
    handle_pinch_target,
    side_grasp_placement_world,
    slide_grasp_toward_handle,
    matrix_to_quat_wxyz,
    quat_inverse,
    quat_multiply,
    quat_to_matrix,
    tracked_surface_top,
    world_to_base,
    world_to_base_quat,
)


class TestAdaptiveGrasp(unittest.TestCase):
    def test_matrix_to_quat_roundtrip(self):
        # Identity
        I = np.eye(3)
        qI = matrix_to_quat_wxyz(I)
        np.testing.assert_allclose(qI, [1.0, 0.0, 0.0, 0.0], atol=1e-6)

        # 180 deg around X
        Rx_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        qRx = matrix_to_quat_wxyz(Rx_180)
        np.testing.assert_allclose(np.abs(qRx), [0.0, 1.0, 0.0, 0.0], atol=1e-6)

    def test_quat_base_world_roundtrip(self):
        base_quat = np.array([0.7071, 0.0, 0.0, 0.7071])
        base_quat /= np.linalg.norm(base_quat)
        world_quat = TOP_DOWN_WXYZ.copy()
        
        base_q = world_to_base_quat(world_quat, base_quat)
        restored = base_to_world_quat(base_q, base_quat)
        np.testing.assert_allclose(np.abs(restored), np.abs(world_quat), atol=1e-6)

    def test_mug_tilted_grasp(self):
        # Mug center at (-0.04, -0.28, 0.95)
        center = np.array([-0.04, -0.28, 0.95])
        # Handle points at (-0.04, -0.23, 0.95), dy = +0.05
        handle_pts = np.array([
            [-0.04, -0.23, 0.94],
            [-0.04, -0.23, 0.95],
            [-0.04, -0.23, 0.96],
            [-0.04, -0.228, 0.95],
        ])
        q = adaptive_grasp_orientation(handle_pts, object_center=center, grasp_style="tilted", tilt_angle_deg=35.0)
        R = quat_to_matrix(q)
        z_hand = R[:, 2]
        y_hand = R[:, 1]
        x_hand = R[:, 0]

        # z_hand should point towards -Y (inward to mug) and -Z (downward)
        self.assertLess(z_hand[1], -0.5)
        self.assertLess(z_hand[2], -0.4)
        self.assertAlmostEqual(z_hand[0], 0.0, places=2)

        # y_hand (finger opening) should be tangent, i.e. along X
        self.assertAlmostEqual(abs(y_hand[0]), 1.0, places=2)
        self.assertAlmostEqual(y_hand[1], 0.0, places=2)
        self.assertAlmostEqual(y_hand[2], 0.0, places=2)

        # x_hand (palm normal) should have positive Z component (wrist camera looking down at handle)
        self.assertGreater(x_hand[2], 0.5)

    def test_mug_side_grasp_closes_across_the_handle(self):
        center = np.array([-0.04, -0.28, 0.95])
        handle_pts = np.array([
            [-0.04, -0.23, 0.94],
            [-0.04, -0.23, 0.95],
            [-0.04, -0.23, 0.96],
            [-0.04, -0.228, 0.95],
        ])
        quaternion = adaptive_grasp_orientation(handle_pts, object_center=center, grasp_style="side")
        rotation = quat_to_matrix(quaternion)
        z_hand = rotation[:, 2]
        y_hand = rotation[:, 1]

        self.assertLess(z_hand[1], -0.9)
        self.assertAlmostEqual(z_hand[2], -np.sin(np.radians(15.0)), places=2)
        self.assertAlmostEqual(abs(y_hand[0]), 1.0, places=2)
        self.assertAlmostEqual(y_hand[2], 0.0, places=2)

    def test_handle_pinch_target_moves_the_surface_inward(self):
        center = np.array([-0.04, -0.28, 0.95])
        surface = np.array([[-0.04, -0.23, 0.95]])

        pinch = handle_pinch_target(surface, center, inset=0.008)

        np.testing.assert_allclose(pinch, [-0.04, -0.238, 0.95], atol=1e-6)

        raised = handle_pinch_target(surface, center, inset=0.008, surface_z=0.90)
        np.testing.assert_allclose(raised, [-0.04, -0.238, 0.96], atol=1e-6)

    def test_handle_slide_keeps_the_wrist_height(self):
        center = np.array([-0.04, -0.28, 0.95])
        handle = np.array([
            [-0.04, -0.23, 0.95],
            [-0.04, -0.23, 0.96],
            [-0.04, -0.228, 0.95],
        ])
        quaternion = adaptive_grasp_orientation(handle, object_center=center, grasp_style="side")
        hand = np.array([0.40, -0.22, 0.12])

        slid = slide_grasp_toward_handle(hand, quaternion, 0.02)

        self.assertAlmostEqual(slid[2], hand[2])
        self.assertLess(slid[1], hand[1])
        self.assertAlmostEqual(float(np.linalg.norm(slid[:2] - hand[:2])), 0.02, places=4)

    def test_elongated_object_adaptive_yaw(self):
        # Line of points along x = y (45 deg)
        t = np.linspace(-0.05, 0.05, 20)
        pts = np.stack([t, t, np.zeros_like(t)], axis=1)
        q = adaptive_grasp_orientation(pts, grasp_style="top_down")
        R = quat_to_matrix(q)
        z_hand = R[:, 2]
        y_hand = R[:, 1]
        
        # z_hand is down
        np.testing.assert_allclose(z_hand, [0.0, 0.0, -1.0], atol=1e-6)
        # y_hand should be perpendicular to major axis [1/sqrt(2), 1/sqrt(2), 0]
        major = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        self.assertAlmostEqual(abs(float(np.dot(y_hand, major))), 0.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(y_hand)), 1.0, places=5)

    def test_approach_pose_top_down(self):
        target = np.array([0.1, 0.2, 0.5])
        pose_grasp = adaptive_approach_pose(target, TOP_DOWN_WXYZ, standoff=0.0, finger_reach=0.11)
        # For top down, hand +Z is world -Z, so hand position is target + [0, 0, 0.11]
        np.testing.assert_allclose(pose_grasp, [0.1, 0.2, 0.61], atol=1e-6)

        pose_hover = adaptive_approach_pose(target, TOP_DOWN_WXYZ, standoff=0.10, finger_reach=0.11)
        np.testing.assert_allclose(pose_hover, [0.1, 0.2, 0.71], atol=1e-6)

    def test_approach_pose_tilted(self):
        target = np.array([0.0, 0.0, 1.0])
        center = np.array([0.0, -0.05, 1.0]) # target is +Y from center
        q = adaptive_grasp_orientation(np.array([[0.0, 0.0, 1.0], [0.0, 0.01, 1.0], [0.0, -0.01, 1.0]]), object_center=center, grasp_style="tilted", tilt_angle_deg=45.0)
        
        grasp = adaptive_approach_pose(target, q, standoff=0.0, finger_reach=0.10)
        standoff = adaptive_approach_pose(target, q, standoff=0.10, finger_reach=0.10)
        
        R = quat_to_matrix(q)
        z_hand = R[:, 2]
        # standoff to grasp should be exactly along z_hand
        np.testing.assert_allclose(grasp - standoff, 0.10 * z_hand, atol=1e-6)

    def test_side_grasp_release_keeps_the_counter_clearance(self):
        support = np.array(
            [
                [-0.36, -0.26, 0.910],
                [-0.32, -0.26, 0.916],
                [-0.36, -0.22, 0.912],
                [-0.32, -0.22, 0.914],
            ]
        )
        hand = np.array([0.05, -0.28, 1.00])
        center = np.array([-0.04, -0.28, 0.95])
        preplace, place, moved, top = side_grasp_placement_world(
            support, hand, center, surface_z=0.90, clearance=0.30
        )

        self.assertAlmostEqual(top, 0.916)
        np.testing.assert_allclose(place, [-0.25, -0.24, 1.016])
        np.testing.assert_allclose(preplace, [-0.25, -0.24, 1.316])
        np.testing.assert_allclose(moved[:2], [-0.34, -0.24])
        self.assertAlmostEqual(moved[2] - center[2], place[2] - hand[2])

    def test_collision_gap_is_the_mug_bottom_above_the_plate_top(self):
        gap = collision_vertical_gap(
            np.array([-0.08, -0.32, 1.20]),
            np.array([0.02, -0.22, 1.29]),
            np.array([-0.44, -0.34, 0.900]),
            np.array([-0.24, -0.14, 0.916]),
        )

        self.assertAlmostEqual(gap, 1.20 - 0.916)

    def test_mask_gap_is_the_mug_bottom_above_the_plate(self):
        moving = np.array(
            [
                [0.0, 0.0, 1.20],
                [0.01, 0.0, 1.22],
                [0.0, 0.01, 1.24],
                [0.01, 0.01, 1.30],
            ]
        )
        support = np.array(
            [
                [0.0, 0.0, 0.90],
                [0.02, 0.0, 0.91],
                [0.0, 0.02, 0.90],
                [0.02, 0.02, 0.91],
            ]
        )

        gap = descent_between_points(moving, support)

        self.assertAlmostEqual(gap, float(np.percentile(moving[:, 2], 5)) - tracked_surface_top(support))
        self.assertGreater(gap, 0.25)

    def test_mask_gap_drops_pixels_shared_by_the_mug_and_plate(self):
        moving = np.zeros((20, 20), dtype=bool)
        support = np.zeros((20, 20), dtype=bool)
        moving[2:10, 2:10] = True
        support[6:16, 6:16] = True
        seen = []

        def fake_points(mask, _depth, _camera_params, _camera_path, count=64):
            seen.append(mask.copy())
            if mask[3, 3]:
                return np.tile(np.array([[0.0, 0.0, 1.20]]), (4, 1))
            return np.tile(np.array([[0.0, 0.0, 0.90]]), (4, 1))

        with patch("pseudo_vla.geometry.mask_depth_points", side_effect=fake_points):
            gap = descent_between_masks(moving, support, np.ones((20, 20)), {}, "/cam")

        self.assertAlmostEqual(gap, 0.30)
        self.assertTrue(seen[0][3, 3])
        self.assertFalse(seen[0][7, 7])
        self.assertTrue(seen[1][12, 12])
        self.assertFalse(seen[1][7, 7])


if __name__ == "__main__":
    unittest.main()
