import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pseudo_vla import perception
from pseudo_vla.view import _depth_image

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "track_and_servo.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("track_and_servo_under_test", _SCRIPT_PATH)
track_and_servo = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(track_and_servo)


class _FakeSam:
    def set_image(self, image):
        self.image = image

    def predict(self, box=None, point_coords=None, **kwargs):
        mask = np.zeros(self.image.shape[:2], dtype=bool)
        if point_coords is not None:
            x, y = (int(value) for value in point_coords[0])
            mask[max(0, y - 4):y + 5, max(0, x - 4):x + 5] = True
        else:
            x1, y1, x2, y2 = (int(value) for value in box)
            mask[y1:y2, x1:x2] = True
        return mask[None, ...], np.array([1.0]), None

    def reset_predictor(self):
        pass


class _FakeTracker:
    step = 4

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)


class StackingPerceptionTests(unittest.TestCase):
    def test_depth_feed_maps_near_far_and_invalid_values(self):
        image = _depth_image(np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32))

        self.assertEqual(image.shape, (2, 2, 3))
        self.assertGreater(int(image[0, 0, 2]), int(image[0, 0, 0]))
        self.assertGreater(int(image[1, 0, 0]), int(image[1, 0, 2]))
        np.testing.assert_array_equal(image[1, 1], [0, 0, 0])

    def test_grouped_segmentation_seeds_one_tracker_with_group_slices(self):
        instance = perception.Perception.__new__(perception.Perception)
        instance.sam = _FakeSam()
        instance.tracker = _FakeTracker()
        instance.frames = []
        instance.queries = None
        instance.query_groups = {}
        instance._window_start = 0
        instance.mask = None
        instance.query_xy = None
        instance.group_masks = {}
        instance._ensure_sam = lambda: None
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        prompts = {
            "support": (None, (4, 4, 28, 28)),
            "moving": (None, (36, 36, 60, 60)),
        }

        with patch.object(perception.torch.Tensor, "cuda", lambda tensor: tensor):
            points = instance.segment_groups(image, prompts, count_per_group=4)

        self.assertEqual(list(points), ["support", "moving"])
        self.assertEqual(instance.query_groups["support"], slice(0, 4))
        self.assertEqual(instance.query_groups["moving"], slice(4, 8))
        self.assertEqual(instance.queries.shape, (1, 8, 3))
        self.assertEqual(len(instance.tracker.calls), 1)
        self.assertTrue(np.all(instance.mask[4:28, 4:28]))
        self.assertTrue(np.all(instance.mask[36:60, 36:60]))

        tracks = np.arange(16, dtype=np.float32).reshape(8, 2)
        visible = np.ones(8, dtype=bool)
        moving_tracks, moving_visible = instance.track_group(tracks, visible, "moving")
        np.testing.assert_array_equal(moving_tracks, tracks[4:8])
        np.testing.assert_array_equal(moving_visible, visible[4:8])

        instance.retain_groups(image, ("moving",), count_per_group=4)
        self.assertEqual(instance.query_groups, {"moving": slice(0, 4)})
        self.assertEqual(instance.queries.shape, (1, 4, 3))
        self.assertEqual(len(instance.tracker.calls), 2)
        self.assertEqual(set(instance.group_masks), {"support", "moving"})
        self.assertFalse(np.any(instance.mask[4:28, 4:28]))
        self.assertTrue(np.all(instance.mask[36:60, 36:60]))

        instance.retain_groups(image, ("support",), count_per_group=4)
        self.assertEqual(instance.query_groups, {"support": slice(0, 4)})
        self.assertEqual(instance.queries.shape, (1, 4, 3))
        self.assertEqual(len(instance.tracker.calls), 3)
        self.assertTrue(np.all(instance.mask[4:28, 4:28]))
        self.assertFalse(np.any(instance.mask[36:60, 36:60]))

        instance.segment_groups(image, {"support": ((50.0, 10.0), None)}, count_per_group=4)
        self.assertEqual(instance.query_groups, {"support": slice(0, 4)})
        self.assertEqual(instance.queries.shape, (1, 4, 3))
        self.assertEqual(set(instance.group_masks), {"support"})
        self.assertTrue(np.any(instance.mask[6:15, 46:55]))
        self.assertFalse(np.any(instance.mask[4:28, 4:28]))
        self.assertFalse(np.any(instance.mask[36:60, 36:60]))

    def test_placement_targets_center_on_tracked_top_surface(self):
        support_points = np.array(
            [
                [0.40, 0.10, 0.20],
                [0.42, 0.10, 0.21],
                [0.40, 0.12, 0.22],
                [0.42, 0.12, 0.21],
            ],
            dtype=np.float64,
        )
        preplace, place, center, top = perception.stack_placement_world(
            support_points,
            np.array([0.05, 0.05, 0.05]),
            finger_reach=0.11,
            clearance=0.10,
        )

        np.testing.assert_allclose(center, [0.41, 0.11, 0.245])
        self.assertAlmostEqual(top, 0.22)
        np.testing.assert_allclose(place, [0.41, 0.11, 0.355])
        np.testing.assert_allclose(preplace, [0.41, 0.11, 0.455])

    def test_surface_top_rejects_high_depth_outlier(self):
        points = np.array(
            [
                [0.0, 0.0, 0.060],
                [0.01, 0.0, 0.061],
                [0.0, 0.01, 0.059],
                [0.01, 0.01, 0.062],
                [0.0, 0.0, 0.175],
            ],
            dtype=np.float64,
        )

        self.assertAlmostEqual(perception.tracked_surface_top(points), 0.062)

    def test_pickup_approach_goal_uses_selected_group_depth_tracks(self):
        instance = perception.Perception.__new__(perception.Perception)
        instance.query_groups = {"support": slice(0, 4), "moving": slice(4, 8)}
        tracks = np.arange(16, dtype=np.float32).reshape(8, 2)
        visible = np.ones(8, dtype=bool)
        points = np.array(
            [
                [0.20, -0.30, 0.05],
                [0.22, -0.30, 0.06],
                [0.20, -0.28, 0.07],
                [0.22, -0.28, 0.06],
            ],
            dtype=np.float64,
        )

        with (
            patch.object(perception, "unproject_tracks", return_value=points) as unproject,
            patch.object(
                perception,
                "prim_world_pose",
                return_value=(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
            ),
        ):
            goal = instance.approach_goal(
                tracks,
                visible,
                np.ones((8, 8)),
                {},
                "/camera",
                "/base",
                group="moving",
            )

        self.assertIsNotNone(goal)
        np.testing.assert_array_equal(unproject.call_args.args[0], tracks[4:8])
        np.testing.assert_allclose(goal.position, [0.21, -0.29, 0.17])
        self.assertAlmostEqual(goal.top_z, 0.07)

    def test_stack_cube_mask_prompt_uses_projected_scene_point(self):
        scene = SimpleNamespace(
            object_paths={"green_block": "/World/green"},
            camera_path="/World/EnvCamera",
        )
        image = np.zeros((480, 480, 3), dtype=np.uint8)
        messages = []
        with (
            patch.object(perception, "prim_world_pose", return_value=(np.array([0.0, -0.28, 0.035]), np.zeros(4))),
            patch.object(perception, "project_world_point", return_value=np.array([240.0, 320.0])) as project,
        ):
            point, box = track_and_servo._projected_cube_prompt(
                scene, image, {}, "green", messages.append
            )

        self.assertEqual(point, (240.0, 320.0))
        self.assertIsNone(box)
        np.testing.assert_allclose(project.call_args.args[0], [0.0, -0.28, 0.035])
        self.assertIn("seeding green SAM mask", messages[0])

    def test_completed_stack_reseeds_tracker_on_placed_green_cube(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        scene = SimpleNamespace(
            object_paths={"green_block": "/World/green"},
            camera_path="/World/EnvCamera",
            step=lambda: None,
            read_sensors=lambda: (image, np.ones((32, 32)), {}),
        )
        perception_instance = SimpleNamespace(
            mask=np.ones((32, 32), dtype=bool),
            query_xy=np.array([[16.0, 16.0]], dtype=np.float32),
            segment_groups=lambda frame, prompts: self.assertEqual(set(prompts), {"placed"})
        )
        say = unittest.mock.Mock()
        view = SimpleNamespace(update=lambda *args, **kwargs: None)
        with patch.object(
            track_and_servo,
            "_projected_cube_prompt",
            return_value=((16.0, 16.0), None),
        ) as prompt, patch.object(track_and_servo, "_update_camera_views"):
            track_and_servo._seed_placed_cube_tracking(scene, perception_instance, "green", say, view)

        prompt.assert_called_once()
        say.assert_any_call("dropped stale CoTracker points; tracking the placed green cube")

    def test_wrist_rgbd_goal_uses_segmented_depth_for_alignment(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        depth = np.ones((32, 32), dtype=np.float64)
        scene = SimpleNamespace(
            wrist_camera_path="/World/Arm/panda_hand/wrist_cam",
            base_path="/World/Arm/panda_link0",
            read_wrist_sensors=lambda: (image, depth, {"camera": "wrist"}),
        )
        perception_instance = SimpleNamespace(segment_mask=lambda _rgb, _point: np.ones((32, 32), dtype=bool))
        depth_points = np.array(
            [
                [0.20, -0.30, 0.050],
                [0.22, -0.30, 0.051],
                [0.20, -0.28, 0.052],
                [0.22, -0.28, 0.051],
            ],
            dtype=np.float64,
        )
        quaternion = np.array([0.0, 1.0, 0.0, 0.0])
        messages = []
        with (
            patch.object(perception, "project_world_point", return_value=np.array([16.0, 16.0])),
            patch.object(perception, "mask_depth_points", return_value=depth_points) as unproject,
            patch.object(
                perception,
                "prim_world_pose",
                return_value=(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
            ),
        ):
            goal = track_and_servo._wrist_rgbd_goal(
                scene,
                perception_instance,
                "green",
                np.array([0.21, -0.29, 0.035]),
                quaternion,
                np.array([0.05, 0.05, 0.05]),
                messages.append,
            )

        self.assertIsNotNone(goal)
        np.testing.assert_allclose(goal.center_world, [0.21, -0.29, 0.027])
        self.assertAlmostEqual(goal.top_z, 0.052)
        np.testing.assert_allclose(goal.position, [0.21, -0.29, 0.152])
        np.testing.assert_array_equal(goal.quaternion, quaternion)
        self.assertEqual(unproject.call_args.args[3], scene.wrist_camera_path)
        self.assertIn("wrist RGB-D aligned green", messages[0])


if __name__ == "__main__":
    unittest.main()
