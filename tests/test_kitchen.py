import unittest
from unittest.mock import patch

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from pseudo_vla import cli, scene
from pseudo_vla.kitchen import (
    COUNTER_MAX,
    COUNTER_MIN,
    KITCHEN_OBJECTS,
    add_kitchen_room,
    place_kitchen_prop,
)
from pseudo_vla.scene import object_id, object_phrase, select_scene


class KitchenSceneTests(unittest.TestCase):
    def tearDown(self) -> None:
        select_scene("blocks")

    def test_block_phrases_stay_color_block_names(self):
        select_scene("blocks")
        phrases = [object_phrase(spec) for spec in scene.SCENE_OBJECTS]
        self.assertEqual(phrases, ["red block", "green block", "blue block"])

    def test_kitchen_catalog_lists_plates_mugs_and_utensils(self):
        select_scene("kitchen")
        self.assertEqual(
            [object_id(spec) for spec in scene.SCENE_OBJECTS],
            ["plate", "small_plate", "mug", "cup", "fork", "spoon", "knife"],
        )
        self.assertEqual(
            [object_phrase(spec) for spec in scene.SCENE_OBJECTS],
            ["dinner plate", "small plate", "mug", "cup", "fork", "spoon", "knife"],
        )
        self.assertEqual(scene.surface_location(), "counter")
        self.assertAlmostEqual(scene.TABLE_TOP_Z, 0.90)
        self.assertEqual(scene.ARM_POSITION[2], scene.TABLE_TOP_Z + 0.002)

    def test_kitchen_props_sit_on_the_counter_away_from_the_sink(self):
        for spec in KITCHEN_OBJECTS:
            x, y = spec["xy"]
            self.assertGreater(x, COUNTER_MIN[0] + 0.05)
            self.assertLess(x, 0.70)
            self.assertGreater(y, COUNTER_MIN[1] + 0.05)
            self.assertLess(y, 0.0)
        self.assertLess(COUNTER_MAX[2], 1.0)

    def test_authored_props_are_dynamic_and_the_counter_can_be_collided_with(self):
        stage = Usd.Stage.CreateInMemory()
        add_kitchen_room(stage)
        self.assertTrue(stage.GetPrimAtPath("/World/Kitchen/counter").IsValid())
        self.assertTrue(stage.GetPrimAtPath("/World/Kitchen/sink").HasAPI(UsdPhysics.CollisionAPI))
        for spec in KITCHEN_OBJECTS:
            if spec["kind"] == "asset":
                continue
            path = place_kitchen_prop(stage, spec, 0.90)
            prim = stage.GetPrimAtPath(path)
            self.assertTrue(prim.HasAPI(UsdPhysics.RigidBodyAPI))
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
            box = cache.ComputeLocalBound(prim).ComputeAlignedBox()
            self.assertGreater(float(box.GetMax()[2] - box.GetMin()[2]), 0.0)

        plate = UsdGeom.Cylinder(stage.GetPrimAtPath("/World/plate/body"))
        self.assertAlmostEqual(plate.GetRadiusAttr().Get(), 0.095)

    def test_select_scene_restores_the_block_table(self):
        select_scene("kitchen")
        select_scene("blocks")
        self.assertEqual(scene.SCENE_NAME, "blocks")
        self.assertEqual(scene.surface_location(), "table")
        self.assertAlmostEqual(scene.TABLE_TOP_Z, 0.01)
        np.testing.assert_allclose(scene.ENV_CAMERA_EYE, [0.0, 0.35, 0.65])

    def test_existing_asset_translate_is_updated(self):
        stage = Usd.Stage.CreateInMemory()
        prim = UsdGeom.Xform.Define(stage, "/World/mug").GetPrim()
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0))
        xform.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(0.01, 0.01, 0.01))

        scene._set_translate(prim, (-0.04, -0.28, 0.90))
        scene._set_translate(prim, (-0.04, -0.28, 0.95))

        translate = next(
            op.Get()
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
        )
        self.assertEqual([round(float(value), 3) for value in translate], [-0.04, -0.28, 0.95])

    def test_asset_yaw_overrides_the_referenced_rotate(self):
        stage = Usd.Stage.CreateInMemory()
        prim = UsdGeom.Xform.Define(stage, "/World/mug").GetPrim()
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0))
        xform.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(0.01, 0.01, 0.01))

        scene._set_yaw(prim, 90.0)

        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        rotate = next(op.Get() for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ)
        self.assertEqual([round(float(value), 3) for value in rotate], [0.0, 0.0, 90.0])
        self.assertEqual(sum(op.GetOpType() == UsdGeom.XformOp.TypeRotateZ for op in ops), 0)

    def test_cli_selects_the_kitchen_and_rejects_stack_prompt(self):
        with patch("sys.argv", ["track_and_servo.py", "--scene", "kitchen"]):
            args = cli.parse_args()
        self.assertEqual(args.scene, "kitchen")

        with patch("sys.argv", ["track_and_servo.py", "--scene", "kitchen", "--stack-prompt", "red green blue"]):
            with self.assertRaises(SystemExit) as exit_error:
                cli.main()
        self.assertEqual(exit_error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
