"""Kitchen counter scene: cabinets, plates, mugs, and utensils.

Isaac Sim 6.2 ships mug meshes and no kitchen, plate, or utensil assets.
The room, plates, and utensils are authored here. The mugs reference the
shipped meshes.
"""

from __future__ import annotations

import numpy as np

COUNTER_TOP_Z = 0.90
COUNTER_MIN = np.array([-1.15, -0.90, 0.08], dtype=np.float64)
COUNTER_MAX = np.array([1.15, 0.65, COUNTER_TOP_Z], dtype=np.float64)
ARM_POSITION = (0.0, -0.72, COUNTER_TOP_Z + 0.002)
ENV_CAMERA_EYE = np.array([0.05, 0.20, 1.55], dtype=np.float64)
ENV_CAMERA_TARGET = np.array([0.02, -0.32, 0.98], dtype=np.float64)
SURFACE_LOCATION = "counter"

# The mug is an Isaac Sim asset. The plate is built in this scene.
MUG_ASSET = "/Isaac/Props/Mugs/SM_Mug_A2.usd"

KITCHEN_OBJECTS = (
    {
        "id": "plate",
        "name": "plate",
        "phrase": "dinner plate",
        "kind": "plate",
        "xy": (-0.34, -0.24),
        "radius": 0.095,
        "height": 0.016,
        "color": (0.94, 0.94, 0.91),
        "mass": 0.35,
    },
    {
        "id": "mug",
        "name": "mug",
        "phrase": "mug",
        "kind": "asset",
        "xy": (-0.04, -0.28),
        "yaw": -90.0,
        "asset": MUG_ASSET,
        "mass": 0.30,
    },
)


def add_kitchen_room(stage) -> None:
    """Author the floor, walls, cabinets, counter, sink, and cooktop."""
    from pxr import Gf, UsdGeom, UsdLux

    floor = _color(0.46, 0.32, 0.20)
    wall = _color(0.93, 0.91, 0.86)
    cabinet = _color(0.23, 0.29, 0.27)
    door = _color(0.16, 0.20, 0.18)
    stone = _color(0.80, 0.78, 0.74)
    tile = _color(0.55, 0.64, 0.66)
    steel = _color(0.72, 0.75, 0.77)
    cooktop = _color(0.08, 0.08, 0.09)
    upper = _color(0.32, 0.36, 0.34)
    glass = _color(0.62, 0.76, 0.84)

    _add_box(stage, "/World/Kitchen/floor", (-2.4, -2.2, 0.0), (2.4, 2.2, 0.04), floor, collide=True)
    _add_box(stage, "/World/Kitchen/wall_back", (-2.4, 1.55, 0.04), (2.4, 1.70, 2.6), wall, collide=False)
    _add_box(stage, "/World/Kitchen/wall_left", (-2.4, -2.2, 0.04), (-2.25, 1.70, 2.6), wall, collide=False)
    _add_box(stage, "/World/Kitchen/wall_right", (2.25, -2.2, 0.04), (2.4, 1.70, 2.6), wall, collide=False)
    _add_box(stage, "/World/Kitchen/window", (-0.35, 1.52, 1.35), (0.45, 1.56, 1.95), glass, collide=False)

    _add_box(
        stage,
        "/World/Kitchen/base_cabinets",
        (-1.15, -0.82, 0.04),
        (1.15, 0.58, 0.86),
        cabinet,
        collide=False,
    )
    for index, x in enumerate((-0.85, -0.28, 0.28, 0.85)):
        _add_box(
            stage,
            f"/World/Kitchen/cabinet_door_{index}",
            (x - 0.24, -0.84, 0.12),
            (x + 0.24, -0.80, 0.78),
            door,
            collide=False,
        )
    _add_box(
        stage,
        "/World/Kitchen/counter",
        tuple(float(value) for value in COUNTER_MIN),
        tuple(float(value) for value in COUNTER_MAX),
        stone,
        collide=True,
    )
    _add_box(stage, "/World/Kitchen/backsplash", (-1.15, 0.58, 0.90), (1.15, 0.64, 1.35), tile, collide=False)
    _add_box(stage, "/World/Kitchen/sink", (0.72, 0.18, 0.90), (1.08, 0.52, 0.94), steel, collide=True)
    _add_box(stage, "/World/Kitchen/cooktop", (-1.05, 0.12, 0.90), (-0.62, 0.50, 0.93), cooktop, collide=True)
    _add_box(stage, "/World/Kitchen/upper_cabinets", (-1.15, 0.48, 1.40), (1.15, 0.66, 2.05), upper, collide=False)

    dome = UsdLux.DomeLight.Define(stage, "/World/Kitchen/dome")
    dome.CreateIntensityAttr(1000.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Kitchen/sun")
    sun.CreateIntensityAttr(3000.0)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-55.0, 0.0, 25.0))


def place_kitchen_prop(stage, spec: dict, surface_z: float) -> str:
    """Build one plate or utensil sitting on the counter."""
    kind = spec["kind"]
    path = f"/World/{spec['name']}"
    if kind == "plate":
        _plate(stage, path, spec)
    elif kind == "fork":
        _fork(stage, path)
    elif kind == "spoon":
        _spoon(stage, path)
    elif kind == "knife":
        _knife(stage, path)
    else:
        raise ValueError(f"Unsupported kitchen prop {kind!r}")
    _place_root(stage, path, spec, surface_z)
    return path


def _plate(stage, path: str, spec: dict) -> None:
    from pxr import UsdGeom

    UsdGeom.Xform.Define(stage, path)
    plate = UsdGeom.Cylinder.Define(stage, path + "/body")
    plate.CreateRadiusAttr(float(spec["radius"]))
    plate.CreateHeightAttr(float(spec["height"]))
    plate.CreateAxisAttr("Z")
    plate.CreateDisplayColorAttr([spec["color"]])
    _translate(plate, (0.0, 0.0, float(spec["height"]) / 2.0))
    rim = UsdGeom.Cylinder.Define(stage, path + "/rim")
    rim.CreateRadiusAttr(float(spec["radius"]) * 0.62)
    rim.CreateHeightAttr(float(spec["height"]) * 0.45)
    rim.CreateAxisAttr("Z")
    rim.CreateDisplayColorAttr([tuple(min(1.0, channel + 0.03) for channel in spec["color"])])
    _translate(rim, (0.0, 0.0, float(spec["height"]) * 0.72))


def _fork(stage, path: str) -> None:
    metal = (0.74, 0.76, 0.78)
    _xform(stage, path)
    _bar(stage, path + "/handle", (0.012, 0.09, 0.004), (0.0, -0.015, 0.002), metal)
    _bar(stage, path + "/head", (0.024, 0.028, 0.003), (0.0, 0.048, 0.0015), metal)
    for index, x in enumerate((-0.009, -0.003, 0.003, 0.009)):
        _bar(stage, f"{path}/tine_{index}", (0.003, 0.038, 0.002), (x, 0.078, 0.001), metal)


def _spoon(stage, path: str) -> None:
    from pxr import Gf, UsdGeom

    metal = (0.76, 0.77, 0.74)
    _xform(stage, path)
    _bar(stage, path + "/handle", (0.012, 0.09, 0.004), (0.0, -0.02, 0.002), metal)
    bowl = UsdGeom.Sphere.Define(stage, path + "/bowl")
    bowl.CreateRadiusAttr(0.02)
    bowl.CreateDisplayColorAttr([metal])
    xform = UsdGeom.Xformable(bowl.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.055, 0.006))
    xform.AddScaleOp().Set(Gf.Vec3d(0.85, 1.15, 0.35))


def _knife(stage, path: str) -> None:
    wood = (0.45, 0.28, 0.14)
    steel = (0.78, 0.80, 0.82)
    _xform(stage, path)
    _bar(stage, path + "/handle", (0.016, 0.08, 0.008), (0.0, -0.03, 0.004), wood)
    _bar(stage, path + "/blade", (0.016, 0.12, 0.002), (0.0, 0.07, 0.001), steel)


def _place_root(stage, path: str, spec: dict, surface_z: float) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    prim = stage.GetPrimAtPath(path)
    xform = UsdGeom.Xformable(prim)
    xy = spec["xy"]
    translate = xform.AddTranslateOp()
    translate.Set(Gf.Vec3d(float(xy[0]), float(xy[1]), float(surface_z) + 0.002))
    rotate = xform.AddRotateZOp()
    rotate.Set(float(spec.get("yaw", 0.0)))
    # Translate must stay outside the yaw. The other order spins the place point
    # around the origin and drops the knife across the mug approach.
    xform.SetXformOpOrder([translate, rotate])
    body = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.CreateRigidBodyEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(float(spec.get("mass", 0.2)))
    for descendant in Usd.PrimRange(prim):
        if descendant == prim or not descendant.IsA(UsdGeom.Gprim):
            continue
        UsdPhysics.CollisionAPI.Apply(descendant)


def _xform(stage, path: str) -> None:
    from pxr import UsdGeom

    UsdGeom.Xform.Define(stage, path)


def _bar(stage, path: str, scale: tuple[float, float, float], translate: tuple[float, float, float], color: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([color])
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xform.AddScaleOp().Set(Gf.Vec3d(*scale))


def _add_box(stage, path: str, minimum: tuple[float, float, float], maximum: tuple[float, float, float], color: tuple[float, float, float], *, collide: bool) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    minimum_array = np.asarray(minimum, dtype=np.float64)
    maximum_array = np.asarray(maximum, dtype=np.float64)
    center = (minimum_array + maximum_array) / 2.0
    dims = maximum_array - minimum_array
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([color])
    cube.CreateDoubleSidedAttr(True)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*(float(value) for value in center)))
    xform.AddScaleOp().Set(Gf.Vec3d(*(float(value) for value in dims)))
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        mesh = UsdPhysics.MeshCollisionAPI.Apply(cube.GetPrim())
        mesh.CreateApproximationAttr("boundingCube")


def _translate(geom, translation: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    UsdGeom.Xformable(geom.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*translation))


def _color(*channels: float) -> tuple[float, float, float]:
    return channels
