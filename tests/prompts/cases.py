"""Fifty instructions with the Tev1 answers and skill steps they should produce.

The runner in pseudo_vla.eval_plans feeds these answers back only when the planner
offered them. A case matches when the resulting steps, or the expected stop, agree.
"""

from __future__ import annotations

TWO = {"objects": ["mug", "plate"], "locations": ["counter"], "containers": []}
THREE = {"objects": ["mug", "plate", "bowl"], "locations": ["counter"], "containers": []}
ONE = {"objects": ["mug"], "locations": ["counter"], "containers": []}
BLOCKS = {"objects": ["blue", "red", "green"], "locations": ["table"], "containers": []}
DRAWER = {
    "objects": ["mug"],
    "locations": ["counter"],
    "containers": ["drawer"],
    "opened": [],
}
OPEN_DRAWER = {
    "objects": ["spoon"],
    "locations": ["counter"],
    "containers": ["drawer"],
    "opened": ["drawer"],
    "inside": {"spoon": "drawer"},
}


def _move(instruction, scene, script, steps):
    return {
        "instruction": instruction,
        "scene": scene,
        "answers": [("task_intent", "skill_sequence"), *script],
        "expected": {"intent": "skill_sequence", "steps": steps},
    }


def _find(instruction, scene, target):
    return {
        "instruction": instruction,
        "scene": scene,
        "answers": [("task_intent", "find"), ("find_target", target)],
        "expected": {"intent": "find", "query": target, "steps": []},
    }


def _stop(instruction, scene, answers, token):
    return {
        "instruction": instruction,
        "scene": scene,
        "answers": answers,
        "expected": {"error": token},
    }


def _pick_place(moving, kind, target, extra=()):
    steps = [
        {"skill": "pick", "object": moving},
        {"skill": "place", "object": moving, "kind": kind, "target": target},
    ]
    return steps


CASES = [
    _move(
        "hold the mug and place it on the plate",
        TWO,
        [("plan_more", True), ("skill_next", "pick_place"), ("moving_object", "mug"), ("destination_kind", "on"), ("plan_more", False)],
        _pick_place("mug", "on", "plate"),
    ),
    _move(
        "hold the plate and place it on the mug",
        TWO,
        [("plan_more", True), ("skill_next", "pick_place"), ("moving_object", "plate"), ("destination_kind", "on"), ("plan_more", False)],
        _pick_place("plate", "on", "mug"),
    ),
    _move(
        "place the mug to the left of the plate",
        TWO,
        [("plan_more", True), ("skill_next", "pick_place"), ("moving_object", "mug"), ("destination_kind", "left_of"), ("plan_more", False)],
        _pick_place("mug", "left_of", "plate"),
    ),
    _move(
        "place the mug to the right of the plate",
        TWO,
        [("plan_more", True), ("skill_next", "pick_place"), ("moving_object", "mug"), ("destination_kind", "right_of"), ("plan_more", False)],
        _pick_place("mug", "right_of", "plate"),
    ),
    _move(
        "put the mug on the counter",
        TWO,
        [("plan_more", True), ("skill_next", "pick_place"), ("moving_object", "mug"), ("destination_kind", "at"), ("plan_more", False)],
        _pick_place("mug", "at", "counter"),
    ),
    _move(
        "put the bowl on the plate",
        THREE,
        [("plan_more", True), ("skill_next", "pick_place"), ("moving_object", "bowl"), ("destination_kind", "on"), ("destination_target", "plate"), ("plan_more", False)],
        _pick_place("bowl", "on", "plate"),
    ),
    _move(
        "set the mug on the bowl",
        THREE,
        [("plan_more", True), ("skill_next", "pick_place"), ("moving_object", "mug"), ("destination_kind", "on"), ("destination_target", "bowl"), ("plan_more", False)],
        _pick_place("mug", "on", "bowl"),
    ),
    _move(
        "pick up the plate",
        {**TWO, "supported_by": {"mug": "plate"}},
        [("plan_more", True), ("skill_next", "pick"), ("moving_object", "plate"), ("plan_more", False)],
        [
            {"skill": "pick", "object": "mug"},
            {"skill": "place", "object": "mug", "kind": "at", "target": "counter"},
            {"skill": "pick", "object": "plate"},
        ],
    ),
    _move(
        "pick up the mug",
        ONE,
        [("plan_more", True), ("skill_next", "pick"), ("plan_more", False)],
        [{"skill": "pick", "object": "mug"}],
    ),
    _move(
        "set the mug down on the counter",
        ONE,
        [("plan_more", True), ("skill_next", "place"), ("plan_more", False)],
        _pick_place("mug", "at", "counter"),
    ),
    _find("where is the mug", TWO, "mug"),
    _find("find the plate", TWO, "plate"),
    _find("locate the mug", ONE, "mug"),
    _stop(
        "find the spoon",
        TWO,
        [("task_intent", "find"), ("find_target", "none")],
        "could not match",
    ),
    _move(
        "pick up the mug and then put it on the plate",
        TWO,
        [("plan_more", True), ("skill_next", "pick"), ("moving_object", "mug"), ("plan_more", True), ("skill_next", "place"), ("destination_kind", "on"), ("plan_more", False)],
        _pick_place("mug", "on", "plate"),
    ),
    _move(
        "pick up the mug and push the plate left",
        TWO,
        [("plan_more", True), ("skill_next", "pick"), ("moving_object", "mug"), ("plan_more", True), ("skill_next", "push"), ("push_direction", "left"), ("plan_more", False)],
        [
            {"skill": "pick", "object": "mug"},
            {"skill": "place", "object": "mug", "kind": "at", "target": "counter"},
            {"skill": "push", "object": "plate", "direction": "left"},
        ],
    ),
    _move(
        "put the blue block on the red block and the green block left of red",
        BLOCKS,
        [
            ("plan_more", True),
            ("skill_next", "pick_place"),
            ("moving_object", "blue"),
            ("destination_kind", "on"),
            ("destination_target", "red"),
            ("plan_more", True),
            ("skill_next", "pick_place"),
            ("moving_object", "green"),
            ("destination_kind", "left_of"),
            ("destination_target", "red"),
            ("plan_more", False),
        ],
        [
            *_pick_place("blue", "on", "red"),
            *_pick_place("green", "left_of", "red"),
        ],
    ),
    _move(
        "stack the blue block on the red block",
        BLOCKS,
        [("plan_more", True), ("skill_next", "pick_place"), ("moving_object", "blue"), ("destination_kind", "on"), ("destination_target", "red"), ("plan_more", False)],
        _pick_place("blue", "on", "red"),
    ),
    _move(
        "move the green block onto the table",
        BLOCKS,
        [("plan_more", True), ("skill_next", "pick_place"), ("moving_object", "green"), ("destination_kind", "at"), ("plan_more", False)],
        _pick_place("green", "at", "table"),
    ),
    _move(
        "pick up the blue block and then the green block",
        BLOCKS,
        [("plan_more", True), ("skill_next", "pick"), ("moving_object", "blue"), ("plan_more", True), ("skill_next", "pick"), ("moving_object", "green"), ("plan_more", False)],
        [
            {"skill": "pick", "object": "blue"},
            {"skill": "place", "object": "blue", "kind": "at", "target": "table"},
            {"skill": "pick", "object": "green"},
        ],
    ),
    _move(
        "open the drawer",
        DRAWER,
        [("plan_more", True), ("skill_next", "open"), ("plan_more", False)],
        [{"skill": "open", "container": "drawer"}],
    ),
    _move(
        "close the drawer",
        {**DRAWER, "opened": ["drawer"]},
        [("plan_more", True), ("skill_next", "close"), ("plan_more", False)],
        [{"skill": "close", "container": "drawer"}],
    ),
    _move(
        "push the mug left",
        ONE,
        [("plan_more", True), ("skill_next", "push"), ("push_direction", "left"), ("plan_more", False)],
        [{"skill": "push", "object": "mug", "direction": "left"}],
    ),
    _move(
        "push the plate to the right",
        TWO,
        [("plan_more", True), ("skill_next", "push"), ("moving_object", "plate"), ("push_direction", "right"), ("plan_more", False)],
        [{"skill": "push", "object": "plate", "direction": "right"}],
    ),
    _move(
        "press the button",
        {"objects": ["button"], "locations": ["counter"], "containers": []},
        [("plan_more", True), ("skill_next", "press"), ("plan_more", False)],
        [{"skill": "press", "object": "button"}],
    ),
    _move(
        "rotate the knob",
        {"objects": ["knob"], "locations": ["counter"], "containers": []},
        [("plan_more", True), ("skill_next", "rotate"), ("plan_more", False)],
        [{"skill": "rotate", "object": "knob", "grasp_family": "parallel"}],
    ),
    _move(
        "open the drawer and then close it",
        DRAWER,
        [("plan_more", True), ("skill_next", "open"), ("plan_more", True), ("skill_next", "close"), ("plan_more", False)],
        [{"skill": "open", "container": "drawer"}, {"skill": "close", "container": "drawer"}],
    ),
    _move(
        "push the mug forward and then back",
        ONE,
        [("plan_more", True), ("skill_next", "push"), ("push_direction", "forward"), ("plan_more", True), ("skill_next", "push"), ("push_direction", "back"), ("plan_more", False)],
        [
            {"skill": "push", "object": "mug", "direction": "forward"},
            {"skill": "push", "object": "mug", "direction": "back"},
        ],
    ),
    _move(
        "insert the peg into the drawer",
        {"objects": ["peg"], "locations": ["counter"], "containers": ["drawer"], "opened": ["drawer"], "held": "peg"},
        [("plan_more", True), ("skill_next", "insert"), ("plan_more", False)],
        [{"skill": "insert", "object": "peg", "receptacle": "drawer"}],
    ),
    _move(
        "take the spoon out of the drawer",
        OPEN_DRAWER,
        [("plan_more", True), ("skill_next", "extract"), ("plan_more", False)],
        [{"skill": "extract", "object": "spoon", "receptacle": "drawer"}],
    ),
    _move(
        "stack the blue block on the red block",
        {"objects": ["blue", "red"], "locations": ["table"], "containers": []},
        [("plan_more", True), ("skill_next", "stack"), ("stack_base", "red"), ("plan_more", False)],
        _pick_place("blue", "on", "red"),
    ),
    _move(
        "put the green block on the red block",
        BLOCKS,
        [("plan_more", True), ("skill_next", "stack"), ("stack_base", "red"), ("stack_next", "green"), ("plan_more", False)],
        _pick_place("green", "on", "red"),
    ),
    _move(
        "unstack the mug onto the counter",
        {**TWO, "supported_by": {"mug": "plate"}},
        [("plan_more", True), ("skill_next", "unstack"), ("destination_kind", "at"), ("plan_more", False)],
        _pick_place("mug", "at", "counter"),
    ),
    _move(
        "pour the mug into the bowl",
        {**THREE, "held": "mug"},
        [("plan_more", True), ("skill_next", "pour"), ("pour_target", "bowl"), ("plan_more", False)],
        [{"skill": "pour", "object": "mug", "target": "bowl", "transport_constraint": "upright"}],
    ),
    _move(
        "use the sponge on the plate",
        {"objects": ["sponge", "plate"], "locations": ["counter"], "containers": []},
        [("plan_more", True), ("skill_next", "use_tool"), ("tool_object", "sponge"), ("plan_more", False)],
        [
            {"skill": "pick", "object": "sponge"},
            {"skill": "use_tool", "tool": "sponge", "target": "plate", "grasp_family": "parallel"},
        ],
    ),
    _move(
        "hang the mug on the hook",
        {"objects": ["mug", "hook"], "locations": ["counter"], "containers": [], "held": "mug"},
        [("plan_more", True), ("skill_next", "hang"), ("transport_constraint", "upright"), ("plan_more", False)],
        [{"skill": "hang", "object": "mug", "target": "hook", "transport_constraint": "upright"}],
    ),
    _move(
        "wait",
        ONE,
        [("plan_more", True), ("skill_next", "wait"), ("plan_more", False)],
        [{"skill": "wait"}],
    ),
    _move(
        "go home",
        ONE,
        [("plan_more", True), ("skill_next", "go_home"), ("plan_more", False)],
        [{"skill": "go_home"}],
    ),
    _move(
        "do that two more times",
        ONE,
        [("plan_more", True), ("skill_next", "repeat"), ("repeat_count", "2"), ("plan_more", False)],
        [{"skill": "repeat", "count": "2"}],
    ),
    _move(
        "do the thing with the dishes",
        TWO,
        [("plan_more", True), ("skill_next", "ask_clarification"), ("needs_clarification", "ambiguous_skill")],
        [{"skill": "ask_clarification", "topic": "ambiguous_skill"}],
    ),
    _move(
        "move it",
        TWO,
        [("plan_more", True), ("skill_next", "ask_clarification"), ("needs_clarification", "missing_object")],
        [{"skill": "ask_clarification", "topic": "missing_object"}],
    ),
    _move(
        "put the mug somewhere",
        TWO,
        [("plan_more", True), ("skill_next", "ask_clarification"), ("needs_clarification", "missing_destination")],
        [{"skill": "ask_clarification", "topic": "missing_destination"}],
    ),
    _stop("fly the mug to the moon", TWO, [("task_intent", "unsupported")], "unsupported"),
    _stop(
        "fold the laundry",
        TWO,
        [("task_intent", "skill_sequence"), ("plan_more", True), ("skill_next", "unsupported")],
        "unsupported",
    ),
    _stop(
        "cut the bread",
        ONE,
        [("task_intent", "skill_sequence"), ("plan_more", True), ("skill_next", "unsupported")],
        "unsupported",
    ),
    _stop(
        "pick up the fork",
        TWO,
        [("task_intent", "skill_sequence"), ("plan_more", True), ("skill_next", "pick"), ("moving_object", "none")],
        "none",
    ),
    _stop(
        "put the mug underneath the plate",
        TWO,
        [("task_intent", "skill_sequence"), ("plan_more", True), ("skill_next", "pick_place"), ("moving_object", "mug"), ("destination_kind", "none")],
        "none",
    ),
    _stop(
        "pour the mug into nothing",
        {**THREE, "held": "mug"},
        [("task_intent", "skill_sequence"), ("plan_more", True), ("skill_next", "pour"), ("pour_target", "none")],
        "none",
    ),
    _stop(
        "pick up the plate",
        {**THREE, "supported_by": {"mug": "plate", "bowl": "plate"}},
        [
            ("task_intent", "skill_sequence"),
            ("plan_more", True),
            ("skill_next", "pick"),
            ("moving_object", "plate"),
            ("prerequisite_resolution", "unsupported"),
        ],
        "rejected",
    ),
    _stop(
        "push the mug diagonally",
        ONE,
        [("task_intent", "skill_sequence"), ("plan_more", True), ("skill_next", "push"), ("push_direction", "none")],
        "none",
    ),
]
