import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pseudo_vla import controller as track_and_servo
from pseudo_vla import perception
from pseudo_vla import stacking_llm
from pseudo_vla.view import _depth_image

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

    def test_relative_placement_goals_place_left_of_target(self):
        from pseudo_vla.task_plan import Destination

        scene = SimpleNamespace(object_paths={"red_block": "/red"})
        with (
            patch(
                "pseudo_vla.perception.prim_world_pose",
                side_effect=[
                    (np.array([0.0, -0.3, 0.05]), np.array([1.0, 0.0, 0.0, 0.0])),
                    (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
                ],
            ),
            patch("pseudo_vla.perception.world_to_base", side_effect=lambda points, *_: points),
            patch("pseudo_vla.perception.quat_inverse", return_value=np.array([1.0, 0.0, 0.0, 0.0])),
            patch("pseudo_vla.perception.quat_multiply", return_value=np.array([1.0, 0.0, 0.0, 0.0])),
            patch("pseudo_vla.scene.object_dims", return_value=np.array([0.05, 0.05, 0.05])),
        ):
            preplace, place = track_and_servo._relative_placement_goals(
                scene, "blue", Destination("left_of", "red"), "/base"
            )

        np.testing.assert_allclose(place.center_world, [-0.07, -0.3, 0.05])
        self.assertAlmostEqual(place.position[2], 0.16)
        self.assertAlmostEqual(preplace.position[2], 0.26)

    def test_relative_placement_goals_place_right_of_target(self):
        from pseudo_vla.task_plan import Destination

        scene = SimpleNamespace(object_paths={"red_block": "/red"})
        with (
            patch(
                "pseudo_vla.perception.prim_world_pose",
                side_effect=[
                    (np.array([0.0, -0.3, 0.05]), np.array([1.0, 0.0, 0.0, 0.0])),
                    (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
                ],
            ),
            patch("pseudo_vla.perception.world_to_base", side_effect=lambda points, *_: points),
            patch("pseudo_vla.perception.quat_inverse", return_value=np.array([1.0, 0.0, 0.0, 0.0])),
            patch("pseudo_vla.perception.quat_multiply", return_value=np.array([1.0, 0.0, 0.0, 0.0])),
            patch("pseudo_vla.scene.object_dims", return_value=np.array([0.05, 0.05, 0.05])),
        ):
            _preplace, place = track_and_servo._relative_placement_goals(
                scene, "blue", Destination("right_of", "red"), "/base"
            )

        np.testing.assert_allclose(place.center_world, [0.07, -0.3, 0.05])

    def test_table_slot_clearance_detects_other_block(self):
        centers = {
            "/World/red_block": (np.array([-0.22, -0.28, 0.035]), None),
            "/World/green_block": (np.array([0.22, -0.28, 0.035]), None),
            "/World/blue_block": (np.array([0.0, -0.28, 0.035]), None),
        }
        scene = SimpleNamespace(
            object_paths={
                "red_block": "/World/red_block",
                "green_block": "/World/green_block",
                "blue_block": "/World/blue_block",
            }
        )
        with patch(
            "pseudo_vla.perception.prim_world_pose",
            side_effect=lambda path: centers[path],
        ), patch("pseudo_vla.scene.object_dims", return_value=np.array([0.05, 0.05, 0.05])):
            self.assertFalse(track_and_servo._table_slot_is_clear(scene, "blue"))
            centers["/World/green_block"] = (
                np.array([0.0, -0.28, 0.035]),
                None,
            )
            self.assertTrue(track_and_servo._table_slot_is_clear(scene, "blue"))

    def test_relative_placement_clearance_detects_block_in_destination(self):
        from pseudo_vla.task_plan import Destination

        centers = {
            "/red": (np.array([0.0, -0.3, 0.05]), None),
            "/green": (np.array([-0.07, -0.3, 0.05]), None),
            "/blue": (np.array([0.2, -0.3, 0.05]), None),
        }
        scene = SimpleNamespace(
            base_path="/base",
            object_paths={
                "red_block": "/red",
                "green_block": "/green",
                "blue_block": "/blue",
            },
        )
        destination = Destination("left_of", "red")
        with (
            patch.object(
                track_and_servo,
                "_relative_placement_goals",
                return_value=(
                    SimpleNamespace(center_world=np.zeros(3)),
                    SimpleNamespace(center_world=np.array([-0.07, -0.3, 0.05])),
                ),
            ),
            patch(
                "pseudo_vla.perception.prim_world_pose",
                side_effect=lambda path: centers[path],
            ),
            patch("pseudo_vla.scene.object_dims", return_value=np.array([0.05, 0.05, 0.05])),
        ):
            self.assertFalse(
                track_and_servo._relative_slot_is_clear(scene, "blue", destination)
            )
            centers["/green"] = (np.array([0.0, -0.5, 0.05]), None)
            self.assertTrue(
                track_and_servo._relative_slot_is_clear(scene, "blue", destination)
            )

    def test_approach_transit_follows_nearby_taller_blocks(self):
        centers = {
            "red_block": np.array([-0.22, -0.28, 0.035]),
            "green_block": np.array([0.0, -0.28, 0.035]),
            "blue_block": np.array([-0.22, -0.28, 0.085]),
        }
        scene = SimpleNamespace(object_paths={name: name for name in centers})

        def pose(path):
            return centers[path], None

        with (
            patch("pseudo_vla.perception.prim_world_pose", side_effect=pose),
            patch("pseudo_vla.scene.object_dims", return_value=np.array([0.05, 0.05, 0.05])),
        ):
            self.assertTrue(track_and_servo._needs_approach_transit(scene, "green"))
            self.assertTrue(track_and_servo._needs_approach_transit(scene, "red"))
            self.assertFalse(track_and_servo._needs_approach_transit(scene, "blue"))
            centers["blue_block"] = np.array([0.22, -0.28, 0.035])
            self.assertFalse(track_and_servo._needs_approach_transit(scene, "green"))

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

    def test_stack_transit_waypoint_preserves_xy_and_adds_clearance(self):
        hover_goal = perception.ApproachGoal(
            position=np.array([0.2, -0.3, 0.16]),
            quaternion=np.array([0.0, 1.0, 0.0, 0.0]),
            center_world=np.array([0.2, -0.3, 0.035]),
            top_z=0.06,
        )

        transit_goal = track_and_servo._stack_transit_goal(
            hover_goal,
            np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )

        np.testing.assert_allclose(transit_goal.position[:2], hover_goal.position[:2])
        self.assertAlmostEqual(transit_goal.position[2], 0.06 + 0.10 + 0.15)
        np.testing.assert_array_equal(transit_goal.quaternion, hover_goal.quaternion)

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

    def test_text_prompt_reuses_matching_active_tracker_group_if_grounding_fails(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        perception_instance = SimpleNamespace(
            ground=lambda frame, prompt: None,
            query_groups={"green": slice(0, 4)},
            query_xy=np.array(
                [[10.0, 12.0], [14.0, 12.0], [10.0, 16.0], [14.0, 16.0]],
                dtype=np.float32,
            ),
        )
        messages = []

        with patch(
            "pseudo_vla.stacking_llm.normalize_object_prompt",
            return_value="green block",
        ) as normalize:
            point, box = track_and_servo._prompt_to_sam(
                perception_instance, image, "please find the green cube", messages.append
            )

        normalize.assert_called_once_with("please find the green cube")
        self.assertEqual(point, (12.0, 14.0))
        self.assertIsNone(box)
        self.assertIn("reusing active CoTracker points for green", messages[-1])

    def test_numeric_point_and_box_prompts_bypass_llm(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        perception_instance = SimpleNamespace(ground=unittest.mock.Mock())
        with patch("pseudo_vla.stacking_llm.normalize_object_prompt") as normalize:
            point, point_box = track_and_servo._prompt_to_sam(
                perception_instance, image, "12 14", lambda _message: None
            )
            box_point, box = track_and_servo._prompt_to_sam(
                perception_instance, image, "1 2 20 24", lambda _message: None
            )

        self.assertEqual(point, (12.0, 14.0))
        self.assertIsNone(point_box)
        self.assertIsNone(box_point)
        self.assertEqual(box, (1.0, 2.0, 20.0, 24.0))
        normalize.assert_not_called()
        perception_instance.ground.assert_not_called()

    def test_llm_failure_prevents_object_grounding(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        perception_instance = SimpleNamespace(ground=unittest.mock.Mock())
        messages = []
        with patch(
            "pseudo_vla.stacking_llm.normalize_object_prompt",
            side_effect=stacking_llm.StackingModelError("connection refused"),
        ):
            point, box = track_and_servo._prompt_to_sam(
                perception_instance, image, "green block", messages.append
            )
        self.assertIsNone(point)
        self.assertIsNone(box)
        self.assertIn("could not interpret object prompt with Llama", messages[-1])
        perception_instance.ground.assert_not_called()

    def test_text_stack_instruction_dispatches_to_stack_workflow(self):
        scene, perception_instance, servo, commands = object(), object(), object(), object()
        messages = []
        from pseudo_vla.task_plan import Destination, PickPlaceAction, TaskPlan

        plan = TaskPlan(
            intent="pick_place",
            actions=(
                PickPlaceAction("blue", Destination("on", "red")),
            ),
        )
        with (
            patch(
                "pseudo_vla.task_llm.interpret_task_prompt",
                return_value=plan,
            ),
            patch.object(track_and_servo, "_task_scene_state", return_value={}),
            patch("pseudo_vla.task_llm.choose_task_event", side_effect=lambda event, *_args: event),
            patch.object(track_and_servo, "_run_stack_phase", return_value=True) as run_stack,
            patch.object(track_and_servo, "_move_to_prompt") as move,
        ):
            result = track_and_servo._dispatch_user_prompt(
                scene,
                perception_instance,
                servo,
                "stack blue on top of green then red on top of that",
                commands,
                600,
                messages.append,
                messages.append,
                None,
            )

        self.assertIsNone(result)
        self.assertEqual(run_stack.call_args.args[3], "red")
        self.assertEqual(run_stack.call_args.args[4], "blue")
        self.assertIn("validated plan: move blue on red", messages)
        self.assertIn("task plan complete", messages)
        move.assert_not_called()

    def test_relative_and_unstack_plans_execute_in_order(self):
        from pseudo_vla.task_plan import Destination, PickPlaceAction, TaskPlan

        scene, perception_instance, servo, commands = object(), object(), object(), object()
        messages = []
        plan = TaskPlan(
            intent="pick_place",
            actions=(
                PickPlaceAction("blue", Destination("table")),
                PickPlaceAction("green", Destination("left_of", "red")),
            ),
        )
        with (
            patch("pseudo_vla.task_llm.interpret_task_prompt", return_value=plan),
            patch.object(track_and_servo, "_task_scene_state", return_value={}),
            patch("pseudo_vla.task_llm.choose_task_event", side_effect=lambda event, *_args: event),
            patch.object(track_and_servo, "_table_slot_is_clear", return_value=True),
            patch.object(track_and_servo, "_relative_slot_is_clear", return_value=True),
            patch.object(track_and_servo, "_run_stack_phase", return_value=True) as run_phase,
        ):
            result = track_and_servo._dispatch_user_prompt(
                scene,
                perception_instance,
                servo,
                "unstack the tower and put green left of red",
                commands,
                600,
                messages.append,
                messages.append,
                None,
            )

        self.assertIsNone(result)
        destinations = [call.args[11] for call in run_phase.call_args_list]
        self.assertEqual(
            [(item.kind, item.target, call.args[4]) for item, call in zip(destinations, run_phase.call_args_list)],
            [("table", None, "blue"), ("left_of", "red", "green")],
        )
        self.assertIn("validated plan: move blue table then move green left_of red", messages)

    def test_occupied_relative_destination_does_not_move(self):
        from pseudo_vla.task_plan import Destination, PickPlaceAction, TaskPlan

        messages = []
        plan = TaskPlan(
            intent="pick_place",
            actions=(PickPlaceAction("blue", Destination("left_of", "red")),),
        )
        with (
            patch("pseudo_vla.task_llm.interpret_task_prompt", return_value=plan),
            patch.object(track_and_servo, "_task_scene_state", return_value={}),
            patch.object(track_and_servo, "_relative_slot_is_clear", return_value=False),
            patch.object(track_and_servo, "_run_stack_phase") as run_phase,
            patch("pseudo_vla.task_llm.choose_task_event") as choose,
        ):
            result = track_and_servo._dispatch_user_prompt(
                object(),
                object(),
                object(),
                "put blue to the left of red",
                object(),
                600,
                messages.append,
                messages.append,
                None,
            )

        self.assertIsNone(result)
        run_phase.assert_not_called()
        choose.assert_not_called()
        self.assertTrue(any("occupied" in message for message in messages))

    def test_text_object_instruction_uses_llm_query_for_grounding(self):
        from pseudo_vla.task_plan import TaskPlan

        parsed = TaskPlan(intent="find", query="small green cube")
        with (
            patch("pseudo_vla.task_llm.interpret_task_prompt", return_value=parsed),
            patch.object(track_and_servo, "_task_scene_state", return_value={}),
            patch("pseudo_vla.task_llm.choose_task_event", return_value="find"),
            patch.object(track_and_servo, "_find_object", return_value=True) as find,
        ):
            result = track_and_servo._dispatch_user_prompt(
                object(),
                object(),
                object(),
                "find the little green block",
                object(),
                600,
                lambda _message: None,
                lambda _message: None,
                None,
            )

        self.assertIsNone(result)
        self.assertEqual(find.call_args.args[3], "small green cube")

    def test_numeric_instruction_skips_llm_interpreter(self):
        with (
            patch("pseudo_vla.stacking_llm.interpret_user_prompt") as interpret,
            patch.object(track_and_servo, "_move_to_prompt", return_value=None) as move,
        ):
            result = track_and_servo._dispatch_user_prompt(
                object(),
                object(),
                object(),
                "12 14",
                object(),
                600,
                lambda _message: None,
                lambda _message: None,
                None,
            )

        self.assertIsNone(result)
        interpret.assert_not_called()
        self.assertNotIn("grounding_prompt", move.call_args.kwargs)

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
            segment_groups=lambda frame, prompts: self.assertEqual(set(prompts), {"green"})
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
