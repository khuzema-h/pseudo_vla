import unittest

import numpy as np

from pseudo_vla.affordance import match_seen_affordance, peak_pixel, region_points_from_heatmap
from pseudo_vla.geometry import project_tracks_into_view


class AffordancePixelTests(unittest.TestCase):
    def test_peak_maps_onto_the_original_image(self):
        heatmap = np.zeros((224, 224), dtype=np.float32)
        heatmap[112, 56] = 1.0
        x, y = peak_pixel(heatmap, 480, 256)
        self.assertAlmostEqual(x, (56 + 0.5) * 480 / 224)
        self.assertAlmostEqual(y, (112 + 0.5) * 256 / 224)

    def test_peak_uses_the_hottest_cell(self):
        heatmap = np.zeros((4, 8), dtype=np.float32)
        heatmap[1, 3] = 0.2
        heatmap[2, 7] = 0.9
        x, y = peak_pixel(heatmap, 8, 4)
        self.assertAlmostEqual(x, 7.5)
        self.assertAlmostEqual(y, 2.5)


class SeenAffordanceTests(unittest.TestCase):
    def test_hold_mug_matches_the_seen_hold_affordance(self):
        self.assertEqual(
            match_seen_affordance("hold mug and place it on the plate"),
            "hold",
        )

    def test_longer_affordance_wins_at_the_same_word(self):
        self.assertEqual(match_seen_affordance("cut with the knife"), "cut_with")
        self.assertEqual(match_seen_affordance("pick up the mug"), "pick_up")

    def test_unrelated_instruction_matches_nothing(self):
        self.assertIsNone(match_seen_affordance("move the fork to the left of the spoon"))

    def test_hot_heatmap_inside_the_mask_seeds_the_grasp_region(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[8:32, 8:32] = True
        heatmap = np.zeros((40, 40), dtype=np.float32)
        heatmap[8:16, 20:32] = 1.0
        points = region_points_from_heatmap(mask, heatmap, count=8, origin=(100, 50))
        self.assertIsNotNone(points)
        self.assertTrue(np.all(points[:, 0] >= 120))
        self.assertTrue(np.all(points[:, 0] < 132))
        self.assertTrue(np.all(points[:, 1] >= 58))
        self.assertTrue(np.all(points[:, 1] < 66))

    def test_warm_pixels_are_not_grasp_points(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[8:32, 8:32] = True
        heatmap = np.full((40, 40), 0.6, dtype=np.float32)
        heatmap[8:14, 24:32] = 1.0
        points = region_points_from_heatmap(mask, heatmap, count=8)
        self.assertIsNotNone(points)
        self.assertTrue(np.all(points[:, 0] >= 24))
        self.assertTrue(np.all(points[:, 0] < 32))
        self.assertTrue(np.all(points[:, 1] >= 8))
        self.assertTrue(np.all(points[:, 1] < 14))


class TrackTransferTests(unittest.TestCase):
    def test_only_points_inside_the_wrist_image_are_kept(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)

        def project(point):
            if point[0] == 1.0:
                return None
            if point[0] == 2.0:
                return np.array([120.0, 10.0])
            return np.array([12.0, 20.0])

        pixels = project_tracks_into_view(points, project, 100, 80)
        np.testing.assert_allclose(pixels, [[12.0, 20.0]])


if __name__ == "__main__":
    unittest.main()
