import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from pseudo_vla import stacking_llm, task_llm, task_workflow
from pseudo_vla.task_plan import Destination, PickPlaceAction, TaskPlan


class GenericTaskTests(unittest.TestCase):
    def _mock_ollama(self, response):
        return patch.object(
            stacking_llm,
            "urlopen",
            return_value=io.BytesIO(json.dumps(response).encode("utf-8")),
        )

    def test_interpreter_parses_generic_left_of_action_and_sends_scene_state(self):
        response = {
            "message": {
                "content": json.dumps(
                    {
                        "intent": "pick_place",
                        "query": "",
                        "actions": [
                            {
                                "object": "blue",
                                "destination": {"kind": "left_of", "target": "red"},
                            }
                        ],
                    }
                )
            }
        }
        scene_state = {
            "objects": [{"id": "blue"}, {"id": "red"}],
            "locations": ["table"],
            "containers": [],
            "supported_by": {"blue": "red"},
        }
        with self._mock_ollama(response) as post:
            plan = task_llm.interpret_task_prompt(
                "place blue left of red",
                scene_state,
            )
        payload = json.loads(post.call_args.args[0].data)
        self.assertEqual(plan.actions[0].destination, Destination("left_of", "red"))
        self.assertIn('"supported_by": {"blue": "red"}', payload["messages"][1]["content"])

    def test_prompt_wording_does_not_replace_validated_actions(self):
        response = {
            "message": {
                "content": json.dumps(
                    {
                        "intent": "pick_place",
                        "query": "",
                        "actions": [
                            {
                                "object": "blue",
                                "destination": {"kind": "at", "target": "table"},
                            }
                        ],
                    }
                )
            }
        }
        with self._mock_ollama(response):
            plan = task_llm.interpret_task_prompt(
                "clear everything off the stack",
                {
                    "objects": ["blue", "green", "red"],
                    "locations": ["table"],
                    "containers": [],
                    "supported_by": {"blue": "green", "green": "red"},
                },
            )
        self.assertEqual(plan.actions, (PickPlaceAction("blue", Destination("at", "table")),))

    def test_unstack_with_explicit_side_keeps_model_destination(self):
        response = {
            "message": {
                "content": json.dumps(
                    {
                        "intent": "pick_place",
                        "query": "",
                        "actions": [
                            {
                                "object": "blue",
                                "destination": {"kind": "left_of", "target": "green"},
                            }
                        ],
                    }
                )
            }
        }
        with self._mock_ollama(response):
            plan = task_llm.interpret_task_prompt(
                "move blue to the left of green",
                {
                    "objects": ["blue", "green", "red"],
                    "locations": ["table"],
                    "containers": [],
                    "supported_by": {"blue": "red"},
                },
            )
        self.assertEqual(plan.actions, (PickPlaceAction("blue", Destination("left_of", "green")),))

    def test_d3_stop_prevents_task_action(self):
        plan = TaskPlan(
            intent="pick_place",
            actions=(PickPlaceAction("blue", Destination("on", "red")),),
        )
        events = []
        with patch.object(task_llm, "choose_task_event", side_effect=lambda event, *_args: "stop"):
            result = task_workflow.run_task_plan(
                plan,
                events.append,
                lambda _query: True,
                lambda _action, _authorize: self.fail("physical action must not run"),
            )
        self.assertFalse(result)
        self.assertTrue(any("stopped the task before find" in event for event in events))

    def test_workflow_authorizes_only_requested_phases_and_preserves_order(self):
        plan = TaskPlan(
            intent="pick_place",
            actions=(
                PickPlaceAction("blue", Destination("on", "red")),
                PickPlaceAction("green", Destination("left_of", "red")),
            ),
        )
        selected = []
        executed = []

        def choose(event, action, index, last_result, validated_plan):
            selected.append((event, action.object_id, index, len(validated_plan.actions)))
            return event

        def execute(action, authorize):
            for stage in ("approach", "grasp", "place"):
                self.assertTrue(authorize(stage))
                self.assertTrue(authorize(stage, completed=True))
            executed.append(action.object_id)
            return True

        with patch.object(task_llm, "choose_task_event", side_effect=choose):
            result = task_workflow.run_task_plan(plan, lambda _message: None, lambda _query: True, execute)
        self.assertTrue(result)
        self.assertEqual(executed, ["blue", "green"])
        self.assertEqual(
            [event for event, *_rest in selected],
            ["find", "plan", "approach", "grasp", "place", "find", "plan", "approach", "grasp", "place"],
        )


if __name__ == "__main__":
    unittest.main()
