import unittest

from pseudo_vla.skills import REGISTRY, Catalog, SkillError, World, simulate, step


class SkillRegistryTests(unittest.TestCase):
    def test_world_reads_supported_by_from_scene_state(self):
        world = World.from_scene_state(
            {
                "objects": [{"id": "plate"}, {"id": "mug"}],
                "supported_by": {"mug": "plate"},
                "inside": {"spoon": "drawer"},
                "opened": ["drawer"],
                "held": "fork",
            }
        )
        self.assertEqual(world.held, "fork")
        self.assertEqual(world.on, (("mug", "plate"),))
        self.assertEqual(world.inside, (("spoon", "drawer"),))
        self.assertTrue(world.is_open("drawer"))
        self.assertEqual(
            Catalog.from_scene_state(
                {"objects": [{"id": "mug"}, "plate"], "locations": ["counter"]}
            ).objects,
            ("mug", "plate"),
        )

    def test_pick_requires_an_empty_gripper_and_a_clear_object(self):
        blocked = World(on=(("mug", "plate"),))
        self.assertEqual(REGISTRY["pick"].requires(blocked, {"object": "plate"}), "blocked")
        full = World(held="mug")
        self.assertEqual(REGISTRY["pick"].requires(full, {"object": "plate"}), "gripper_full")
        closed = World(inside=(("mug", "drawer"),))
        self.assertEqual(
            REGISTRY["pick"].requires(closed, {"object": "mug"}),
            "closed_container",
        )
        clear = World(on=(("mug", "plate"),), opened=("drawer",))
        self.assertIsNone(REGISTRY["pick"].requires(clear, {"object": "mug"}))

    def test_pick_apply_holds_the_object_and_clears_its_support(self):
        world = REGISTRY["pick"].apply(World(on=(("mug", "plate"),)), {"object": "mug"})
        self.assertEqual(world.held, "mug")
        self.assertEqual(world.on, ())

    def test_place_requires_the_held_object_and_a_legal_kind(self):
        held = World(held="mug")
        self.assertIsNone(
            REGISTRY["place"].requires(held, {"object": "mug", "kind": "on", "target": "plate"})
        )
        self.assertEqual(
            REGISTRY["place"].requires(World(), {"object": "mug", "kind": "on", "target": "plate"}),
            "not_held",
        )
        self.assertEqual(
            REGISTRY["place"].requires(
                held, {"object": "mug", "kind": "under", "target": "plate"}
            ),
            "bad_destination",
        )
        closed = World(held="mug")
        self.assertEqual(
            REGISTRY["place"].requires(
                closed, {"object": "mug", "kind": "inside", "target": "drawer"}
            ),
            "closed_container",
        )

    def test_place_on_sets_support_and_releases_the_gripper(self):
        world = REGISTRY["place"].apply(
            World(held="mug"),
            {"object": "mug", "kind": "on", "target": "plate"},
        )
        self.assertIsNone(world.held)
        self.assertEqual(world.on, (("mug", "plate"),))

    def test_pick_place_macro_matches_pick_then_place(self):
        start = World(on=(("plate", "counter"),))
        direct = simulate(start, [step("pick_place", object="mug", kind="on", target="plate")])
        expanded = simulate(
            start,
            [
                step("pick", object="mug"),
                step("place", object="mug", kind="on", target="plate"),
            ],
        )
        self.assertEqual(direct, expanded)
        self.assertEqual(direct.held, None)
        self.assertIn(("mug", "plate"), direct.on)
        self.assertIn(("plate", "counter"), direct.on)

    def test_simulate_stops_on_the_first_illegal_step(self):
        with self.assertRaises(SkillError):
            simulate(World(held="fork"), [step("pick", object="mug")])


if __name__ == "__main__":
    unittest.main()
