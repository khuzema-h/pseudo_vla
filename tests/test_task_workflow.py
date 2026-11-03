import io
import json
import unittest
from unittest.mock import patch

from pseudo_vla import stacking_llm, task_llm, task_workflow, tev1_client
from pseudo_vla.task_plan import Destination, PickPlaceAction, TaskPlan


def _choice(question_id, choice):
    return {
        "answers": {
            question_id: {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 0.99},
            }
        }
    }


def _boolean(question_id, probability):
    return {"answers": {question_id: {"type": "noul", "noul": probability}}}


def _response_bytes(response):
    return io.BytesIO(json.dumps(response).encode("utf-8"))


def _pick_place_responses(moving, kind, target, more_probability=0.01, include_target=True):
    responses = [
        _choice("task_intent", "pick_place"),
        _boolean("plan_more", 0.99),
        _choice("moving_object", moving),
        _choice("destination_kind", kind),
    ]
    if include_target:
        responses.append(_choice("destination_target", target))
    responses.append(_boolean("plan_more", more_probability))
    return tuple(responses)


class GenericTaskTests(unittest.TestCase):
    def _mock_answers(self, *answers):
        return patch.object(
            tev1_client,
            "urlopen",
            side_effect=[_response_bytes(answer) for answer in answers],
        )

    def test_interpreter_parses_generic_left_of_action_and_sends_scene_state(self):
        scene_state = {
            "objects": [{"id": "blue"}, {"id": "red"}, {"id": "green"}],
            "locations": ["table"],
            "containers": [],
            "supported_by": {"blue": "red"},
        }
        with self._mock_answers(*_pick_place_responses("blue", "left_of", "red")) as post:
            plan = task_llm.interpret_task_prompt("place blue left of red", scene_state)
        payload = json.loads(post.call_args_list[0].args[0].data)
        self.assertEqual(plan.actions[0].destination, Destination("left_of", "red"))
        self.assertEqual(payload["model"], "tev1:4b")
        self.assertEqual(payload["state"]["instruction"], "place blue left of red")
        self.assertEqual(payload["state"]["scene"]["supported_by"], {"blue": "red"})

    def test_prompt_wording_does_not_replace_validated_actions(self):
        with self._mock_answers(
            *_pick_place_responses("blue", "at", "table", include_target=False)
        ):
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

    def test_unstack_with_explicit_side_keeps_selected_destination(self):
        with self._mock_answers(*_pick_place_responses("blue", "left_of", "green")):
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

    def test_tev1_stop_prevents_task_action(self):
        plan = TaskPlan(
            intent="pick_place",
            actions=(PickPlaceAction("blue", Destination("on", "red")),),
        )
        messages = []
        with patch.object(task_llm, "choose_task_event", side_effect=lambda event, *_args: "stop"):
            result = task_workflow.run_task_plan(
                plan,
                messages.append,
                lambda _query: True,
                lambda _action, _authorize: self.fail("physical action must not run"),
            )
        self.assertFalse(result)
        self.assertTrue(any("stopped the task before find" in message for message in messages))

    def test_workflow_authorizes_each_physical_stage_with_fresh_callback_state(self):
        plan = TaskPlan(
            intent="pick_place",
            actions=(
                PickPlaceAction("blue", Destination("on", "red")),
                PickPlaceAction("green", Destination("left_of", "red")),
            ),
        )
        selected = []
        executed = []
        callback_states = []
        stages = ("approach", "grasp", "lift", "place", "verify_placement")

        def choose(event, action, index, last_result, validated_plan):
            selected.append((event, action.object_id, index, len(validated_plan.actions)))
            if event in stages:
                callback_states.append(last_result)
            return event

        def execute(action, authorize):
            for stage in stages:
                self.assertTrue(authorize(stage))
                self.assertTrue(authorize(stage, completed=True))
            executed.append(action.object_id)
            return True

        with patch.object(task_llm, "choose_task_event", side_effect=choose):
            result = task_workflow.run_task_plan(
                plan,
                lambda _message: None,
                lambda _query: True,
                execute,
            )
        self.assertTrue(result)
        self.assertEqual(executed, ["blue", "green"])
        self.assertEqual(
            [event for event, *_rest in selected],
            [
                "find",
                "plan",
                "approach",
                "grasp",
                "lift",
                "place",
                "verify_placement",
                "find",
                "plan",
                "approach",
                "grasp",
                "lift",
                "place",
                "verify_placement",
            ],
        )
        self.assertEqual(callback_states[1]["event"], "approach")
        self.assertEqual(callback_states[5]["event"], "verify_placement")
        self.assertEqual(callback_states[4]["event"], "place")

    def test_find_event_runs_only_after_tev1_authorizes_it(self):
        plan = TaskPlan(intent="find", query="blue")
        found = []
        with patch.object(task_llm, "choose_task_event", return_value="find"):
            self.assertTrue(
                task_workflow.run_task_plan(
                    plan,
                    lambda _message: None,
                    lambda query: found.append(query) or True,
                    lambda *_args: self.fail("find must not run the motion action"),
                )
            )
        self.assertEqual(found, ["blue"])

    def test_controller_success_without_verified_placement_is_not_reported_as_success(self):
        plan = TaskPlan(
            intent="pick_place",
            actions=(PickPlaceAction("blue", Destination("on", "red")),),
        )
        messages = []
        with patch.object(task_llm, "choose_task_event", side_effect=lambda event, *_args: event):
            complete = task_workflow.run_task_plan(
                plan,
                messages.append,
                lambda _query: True,
                lambda _action, _authorize: True,
            )
        self.assertFalse(complete)
        self.assertIn(
            "controller reported success without completing placement verification",
            messages[-1],
        )

    def test_task_stage_request_includes_last_completed_callback(self):
        plan = TaskPlan(
            intent="pick_place",
            actions=(PickPlaceAction("blue", Destination("on", "red")),),
        )
        response = _choice("task_event", "grasp")
        with patch.object(tev1_client, "urlopen", return_value=_response_bytes(response)) as post:
            event = task_llm.choose_task_event(
                "grasp",
                plan.actions[0],
                0,
                {"event": "approach", "success": True, "object": "blue"},
                plan,
            )
        payload = json.loads(post.call_args.args[0].data)
        self.assertEqual(event, "grasp")
        self.assertEqual(payload["state"]["last_event_result"]["event"], "approach")
        self.assertEqual(payload["state"]["phase"], "authorize_grasp")
        self.assertEqual(
            set(payload["questions"]["task_event"]["criteria"]),
            {"grasp", "stop"},
        )


if __name__ == "__main__":
    unittest.main()
