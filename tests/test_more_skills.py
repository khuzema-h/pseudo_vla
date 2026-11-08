import json
import unittest
from pathlib import Path
from unittest.mock import patch

from pseudo_vla import planner
from pseudo_vla.skills import REGISTRY, World, simulate, step


ROOT = Path(__file__).resolve().parents[1]
NEW_QUESTIONS = (
    "push_direction",
    "articulation_action",
    "pour_target",
    "insert_receptacle",
    "tool_object",
    "grasp_family",
    "transport_constraint",
    "repeat_count",
    "phase_event",
    "needs_clarification",
    "skill_next",
    "prerequisite_resolution",
)


def _answers(test, answers):
    def ask_choice(question_id, state, criteria):
        choice = answers[question_id].pop(0)
        test.assertIn(choice, criteria)
        test.assertLessEqual(len(criteria), 24)
        return choice

    def ask_boolean(question_id, state):
        return answers[question_id].pop(0)

    return ask_choice, ask_boolean


class MoreSkillTests(unittest.TestCase):
    def test_open_close_and_push_update_the_world(self):
        closed = World(containers=("drawer",))
        opened = simulate(closed, [step("open", container="drawer")])
        self.assertTrue(opened.is_open("drawer"))
        shut = simulate(opened, [step("close", container="drawer")])
        self.assertFalse(shut.is_open("drawer"))
        pushed = simulate(World(on=(("mug", "plate"),)), [step("push", object="mug", direction="left")])
        self.assertEqual(pushed.on, ())
        self.assertIsNone(REGISTRY["push"].requires(World(), {"object": "mug", "direction": "left"}))
        self.assertEqual(
            REGISTRY["push"].requires(World(held="fork"), {"object": "mug"}),
            "gripper_full",
        )

    def test_press_rotate_insert_and_extract(self):
        self.assertIsNone(REGISTRY["press"].requires(World(), {"object": "button"}))
        self.assertIsNone(
            REGISTRY["rotate"].requires(World(), {"object": "knob", "grasp_family": "parallel"})
        )
        held = World(held="peg", containers=("drawer",), opened=("drawer",))
        inserted = simulate(held, [step("insert", object="peg", receptacle="drawer")])
        self.assertEqual(inserted.inside, (("peg", "drawer"),))
        self.assertIsNone(inserted.held)
        extracted = simulate(inserted, [step("extract", object="peg", receptacle="drawer")])
        self.assertEqual(extracted.held, "peg")
        self.assertEqual(
            REGISTRY["insert"].requires(
                World(held="peg", containers=("drawer",)),
                {"object": "peg", "receptacle": "drawer"},
            ),
            "closed_container",
        )

    def test_stack_and_unstack_reuse_pick_and_place(self):
        stacked = simulate(World(), [step("stack", base="red", moving="blue")])
        self.assertEqual(stacked.on, (("blue", "red"),))
        self.assertEqual(
            simulate(World(on=(("blue", "red"),)), [step("unstack", object="blue", kind="at", target="table")]).on,
            (),
        )
        self.assertEqual(
            REGISTRY["unstack"].requires(World(), {"object": "blue", "kind": "at", "target": "table"}),
            "not_supported",
        )

    def test_pour_tool_hang_and_bookkeeping_skills(self):
        held = World(held="mug")
        self.assertIsNone(
            REGISTRY["pour"].requires(
                held, {"object": "mug", "target": "bowl", "transport_constraint": "upright"}
            )
        )
        self.assertEqual(simulate(held, [step("pour", object="mug", target="bowl")]).held, "mug")
        self.assertEqual(
            REGISTRY["use_tool"].requires(World(), {"tool": "sponge", "target": "plate"}),
            "not_held",
        )
        hung = simulate(held, [step("hang", object="mug", target="hook", transport_constraint="upright")])
        self.assertEqual(hung.on, (("mug", "hook"),))
        self.assertIsNone(simulate(World(), [step("wait")]).held)
        self.assertIsNone(REGISTRY["repeat"].requires(World(), {"count": "2"}))
        self.assertTrue(REGISTRY["ask_clarification"].ends_plan)

    def test_closed_container_is_opened_before_the_pick(self):
        ask_choice, ask_boolean = _answers(
            self,
            {"plan_more": [True, False], "skill_next": ["pick"]},
        )
        with (
            patch.object(planner, "ask_choice", side_effect=ask_choice),
            patch.object(planner, "ask_boolean", side_effect=ask_boolean),
        ):
            plan = planner.plan_steps(
                "take the mug out of the drawer",
                {
                    "objects": ["mug"],
                    "locations": ["counter"],
                    "containers": ["drawer"],
                    "inside": {"mug": "drawer"},
                },
            )
        self.assertEqual(
            plan.steps,
            (step("open", container="drawer"), step("pick", object="mug")),
        )

    def test_stack_uses_the_existing_stack_questions(self):
        seen = []

        def ask_choice(question_id, state, criteria):
            seen.append(question_id)
            return {"skill_next": "stack", "stack_base": "red"}[question_id]

        def ask_boolean(question_id, state):
            return [True, False].pop(0) if False else answers.pop(0)

        answers = [True, False]
        with (
            patch.object(planner, "ask_choice", side_effect=ask_choice),
            patch.object(planner, "ask_boolean", side_effect=lambda *_args: answers.pop(0)),
        ):
            plan = planner.plan_steps(
                "stack blue on red",
                {"objects": ["blue", "red"], "locations": ["table"]},
            )
        self.assertEqual(seen, ["skill_next", "stack_base"])
        self.assertEqual(
            plan.steps,
            (
                step("pick", object="blue"),
                step("place", object="blue", kind="on", target="red"),
            ),
        )

    def test_clarification_ends_the_plan(self):
        ask_choice, ask_boolean = _answers(
            self,
            {
                "plan_more": [True],
                "skill_next": ["ask_clarification"],
                "needs_clarification": ["ambiguous_skill"],
            },
        )
        with (
            patch.object(planner, "ask_choice", side_effect=ask_choice),
            patch.object(planner, "ask_boolean", side_effect=ask_boolean),
        ):
            plan = planner.plan_steps(
                "do the thing",
                {"objects": ["mug", "plate"], "locations": ["counter"]},
            )
        self.assertEqual(plan.steps, (step("ask_clarification", topic="ambiguous_skill"),))

    def test_new_question_ids_are_added_without_removing_old_ones(self):
        bank = json.loads((ROOT / "config" / "tev1_questions.json").read_text(encoding="utf-8"))
        for question_id in NEW_QUESTIONS:
            self.assertIn(question_id, bank["questions"])
        self.assertEqual(bank["questions"]["moving_object"]["type"], "choice")
        self.assertLessEqual(len(bank["questions"]), 64)


if __name__ == "__main__":
    unittest.main()
