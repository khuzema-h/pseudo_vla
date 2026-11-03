import io
import json
import unittest
from unittest.mock import patch

from pseudo_vla import cli, stacking_llm, stacking_workflow, tev1_client


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


class StackingLlmTests(unittest.TestCase):
    def _mock_answers(self, *answers):
        return patch.object(
            tev1_client,
            "urlopen",
            side_effect=[_response_bytes(answer) for answer in answers],
        )

    def test_stack_prompt_uses_adaptive_tev1_choices_and_preserves_order(self):
        responses = (
            _choice("stack_base", "green"),
            _choice("stack_next", "red"),
            _choice("stack_next", "blue"),
        )
        with self._mock_answers(*responses) as post, patch.dict(
            "os.environ", {"OLLAMA_HOST": "http://127.0.0.1:11434"}
        ):
            result = stacking_llm.parse_stacking_prompt("stack green, then red, then blue")

        self.assertEqual(result, ["green", "red", "blue"])
        requests = [call.args[0] for call in post.call_args_list]
        payloads = [json.loads(request.data) for request in requests]
        self.assertTrue(all(request.full_url == "http://127.0.0.1:11434/v1/systemone" for request in requests))
        self.assertTrue(all(payload["model"] == "tev1:4b" for payload in payloads))
        self.assertEqual(list(payloads[0]["questions"]), ["stack_base"])
        self.assertEqual(payloads[1]["state"]["placed_bottom_to_top"], ["green"])
        self.assertEqual(payloads[2]["state"]["placed_bottom_to_top"], ["green", "red"])

    def test_stack_prompt_rejects_missing_colors_before_calling_tev1(self):
        with patch.object(tev1_client, "urlopen") as post:
            with self.assertRaisesRegex(ValueError, "red, green, and blue"):
                stacking_llm.parse_stacking_prompt("stack green and red")
        post.assert_not_called()

    def test_find_prompt_uses_only_scene_object_choices(self):
        with self._mock_answers(
            _choice("task_intent", "find"),
            _choice("find_target", "green"),
        ) as post:
            result = stacking_llm.normalize_object_prompt("find the little green block")

        self.assertEqual(result, "green")
        payload = json.loads(post.call_args_list[0].args[0].data)
        self.assertEqual(payload["model"], "tev1:4b")
        self.assertEqual(payload["state"]["instruction"], "find the little green block")

    def test_find_prompt_rejects_none_match(self):
        with self._mock_answers(
            _choice("task_intent", "find"),
            _choice("find_target", "none"),
        ):
            with self.assertRaisesRegex(stacking_llm.StackingModelError, "could not match"):
                stacking_llm.normalize_object_prompt("find an unknown object")

    def test_interactive_task_interpreter_routes_one_supported_pair(self):
        with self._mock_answers(
            _choice("task_intent", "pick_place"),
            _boolean("plan_more", 0.99),
            _choice("moving_object", "blue"),
            _choice("destination_kind", "on"),
            _choice("destination_target", "red"),
            _boolean("plan_more", 0.01),
        ) as post:
            parsed = stacking_llm.interpret_user_prompt("stack blue on top of red")

        self.assertEqual(parsed.intent, "stack_blocks")
        self.assertEqual(
            parsed.stacking_relations,
            [stacking_llm.StackingRelation(support="red", moving="blue")],
        )
        final_payload = json.loads(post.call_args_list[-1].args[0].data)
        self.assertEqual(final_payload["state"]["actions_so_far"][0]["object"], "blue")

    def test_interactive_task_interpreter_preserves_requested_chain_order(self):
        with self._mock_answers(
            _choice("task_intent", "pick_place"),
            _boolean("plan_more", 0.99),
            _choice("moving_object", "red"),
            _choice("destination_kind", "on"),
            _choice("destination_target", "blue"),
            _boolean("plan_more", 0.99),
            _choice("moving_object", "green"),
            _choice("destination_kind", "on"),
            _choice("destination_target", "red"),
            _boolean("plan_more", 0.01),
        ):
            parsed = stacking_llm.interpret_user_prompt(
                "stack red on blue then put green on top of red"
            )

        self.assertEqual(
            parsed.stacking_relations,
            [
                stacking_llm.StackingRelation(support="blue", moving="red"),
                stacking_llm.StackingRelation(support="red", moving="green"),
            ],
        )

    def test_interactive_task_interpreter_rejects_disconnected_stack_relations(self):
        with self._mock_answers(
            _choice("task_intent", "pick_place"),
            _boolean("plan_more", 0.99),
            _choice("moving_object", "blue"),
            _choice("destination_kind", "on"),
            _choice("destination_target", "red"),
            _boolean("plan_more", 0.99),
            _choice("moving_object", "green"),
            _choice("destination_kind", "on"),
            _choice("destination_target", "red"),
            _boolean("plan_more", 0.01),
        ):
            with self.assertRaisesRegex(stacking_llm.StackingModelError, "unambiguous"):
                stacking_llm.interpret_user_prompt("stack blue and green on red")

    def test_stack_event_request_contains_current_callback_state(self):
        with self._mock_answers(_choice("stack_event", "stack_next")) as post:
            event = stacking_llm.choose_stack_event(
                ["green", "red", "blue"],
                ["green"],
                "green",
                {"event": "establish_base", "success": True},
            )

        payload = json.loads(post.call_args.args[0].data)
        self.assertEqual(event, "stack_next")
        self.assertEqual(payload["model"], "tev1:4b")
        self.assertEqual(payload["state"]["goal_order_bottom_to_top"], ["green", "red", "blue"])
        self.assertEqual(payload["state"]["placed_bottom_to_top"], ["green"])
        self.assertEqual(payload["state"]["current_support"], "green")
        self.assertEqual(payload["state"]["last_event_result"]["event"], "establish_base")
        self.assertEqual(
            set(payload["questions"]["stack_event"]["criteria"]),
            {"stack_next", "stop"},
        )

    def test_stack_event_rejects_unoffered_choice(self):
        with self._mock_answers(_choice("stack_event", "grasp")):
            with self.assertRaisesRegex(stacking_llm.StackingModelError, "invalid choice"):
                stacking_llm.choose_stack_event(["green", "red", "blue"], [], None, None)

    def test_stack_event_rejects_inconsistent_callback_state(self):
        with self.assertRaisesRegex(ValueError, "prefix"):
            stacking_llm.choose_stack_event(["green", "red", "blue"], ["red"], "red", None)

    def test_stack_stage_authorizes_only_the_requested_pair(self):
        with self._mock_answers(_choice("stack_event", "grasp")) as post:
            event = stacking_llm.choose_stack_action(
                "green",
                "red",
                "grasp",
                {"event": "approach", "success": True},
            )
        payload = json.loads(post.call_args.args[0].data)
        self.assertEqual(event, "grasp")
        self.assertEqual(payload["state"]["requested_pair"], {"support": "green", "moving": "red"})
        self.assertEqual(
            set(payload["questions"]["stack_event"]["criteria"]),
            {"grasp", "stop"},
        )

    def test_question_bank_is_bounded_and_supports_all_tev1_question_types(self):
        questions = tev1_client._question_bank()["questions"]
        self.assertLessEqual(len(questions), 64)
        self.assertEqual(
            {question["type"] for question in questions.values()},
            {"choice", "noul", "score"},
        )

    def test_score_question_returns_a_valid_ordered_rubric_value(self):
        response = {"answers": {"affordance_fit": {"type": "score", "score": 1.5}}}
        with patch.object(tev1_client, "urlopen", return_value=_response_bytes(response)):
            self.assertEqual(
                tev1_client.ask_score(
                    "affordance_fit",
                    {"instruction": "hold mug", "selected_affordance": "hold"},
                ),
                1.5,
            )

    def test_boolean_question_uses_probability_and_rejects_an_exact_tie(self):
        response = {"answers": {"plan_more": {"type": "noul", "noul": 0.8}}}
        with patch.object(tev1_client, "urlopen", return_value=_response_bytes(response)):
            self.assertTrue(tev1_client.ask_boolean("plan_more", {"instruction": "stack"}))

        tied = {"answers": {"plan_more": {"type": "noul", "noul": 0.5}}}
        with patch.object(tev1_client, "urlopen", return_value=_response_bytes(tied)):
            with self.assertRaisesRegex(tev1_client.Tev1Error, "undecided"):
                tev1_client.ask_boolean("plan_more", {"instruction": "stack"})

    def test_question_batch_over_64_is_rejected_before_network_call(self):
        with patch.object(tev1_client, "urlopen") as post:
            with self.assertRaisesRegex(ValueError, "one and 64"):
                tev1_client.ask_questions({}, ["task_intent"] * 65)
        post.assert_not_called()

    def test_decision_model_reports_endpoint_failure(self):
        with patch.object(tev1_client, "urlopen", side_effect=OSError("connection refused")):
            with self.assertRaisesRegex(stacking_llm.StackingModelError, "connection refused"):
                stacking_llm.choose_stack_event(["green", "red", "blue"], [], None, None)

    def test_requested_pair_workflow_executes_only_explicit_placement(self):
        relations = [stacking_llm.StackingRelation(support="red", moving="blue")]
        placements = []
        authorized = []

        def fake_stack_phase(scene, perception, servo, support, moving, *args):
            placements.append((support, moving))
            authorize = args[-1]
            for stage in stacking_llm.STACK_ACTION_EVENTS:
                self.assertTrue(authorize(stage))
                authorized.append(stage)
                self.assertTrue(authorize(stage, completed=True))
            return True

        with patch.object(
            stacking_llm,
            "choose_stack_action",
            side_effect=lambda support, moving, stage, _result: stage,
        ) as choose_action:
            completed = stacking_workflow.run_requested_stack_relations(
                object(),
                object(),
                object(),
                relations,
                object(),
                10,
                lambda _message: None,
                lambda _message: None,
                None,
                fake_stack_phase,
            )

        self.assertTrue(completed)
        self.assertEqual(placements, [("red", "blue")])
        self.assertEqual(authorized, list(stacking_llm.STACK_ACTION_EVENTS))
        self.assertEqual(choose_action.call_count, len(stacking_llm.STACK_ACTION_EVENTS))

    def test_prompted_workflow_dispatches_pairs_in_bottom_to_top_order(self):
        placements = []
        authorized = []

        def fake_stack_phase(scene, perception, servo, support, moving, *args):
            placements.append((support, moving))
            authorize = args[-1]
            for stage in stacking_llm.STACK_ACTION_EVENTS:
                self.assertTrue(authorize(stage))
                authorized.append(stage)
            return True

        def choose_event(goal_order, placed, support, last_result, **kwargs):
            if kwargs.get("expected_event") is not None:
                return kwargs["expected_event"]
            if not placed:
                return "establish_base"
            if len(placed) < len(goal_order):
                return "stack_next"
            return "finish"

        with (
            patch.object(stacking_llm, "choose_stack_event", side_effect=choose_event),
            patch(
                "pseudo_vla.controller._run_stack_phase",
                side_effect=fake_stack_phase,
            ),
        ):
            completed = stacking_workflow.run_prompted_stack_sequence(
                object(),
                object(),
                object(),
                ["green", "red", "blue"],
                object(),
                10,
                lambda _message: None,
                lambda _message: None,
                None,
                fake_stack_phase,
            )

        self.assertTrue(completed)
        self.assertEqual(placements, [("green", "red"), ("red", "blue")])
        self.assertEqual(authorized, list(stacking_llm.STACK_ACTION_EVENTS) * 2)

    def test_stop_decision_prevents_stack_motion(self):
        moved = []
        with patch.object(stacking_llm, "choose_stack_event", return_value="stop"):
            complete = stacking_workflow.run_prompted_stack_sequence(
                object(),
                object(),
                object(),
                ["green", "red", "blue"],
                object(),
                10,
                lambda _message: None,
                lambda _message: None,
                None,
                lambda *_args: moved.append(True),
            )
        self.assertFalse(complete)
        self.assertEqual(moved, [])

    def test_stack_prompt_parse_failure_exits_before_simulator_launch(self):
        with (
            patch("sys.argv", ["track_and_servo.py", "--stack-prompt", "stack red and blue"]),
            patch.object(cli, "launch") as launch,
            self.assertRaises(SystemExit) as error,
        ):
            cli.main()
        self.assertEqual(error.exception.code, 2)
        launch.assert_not_called()

    def test_stack_prompt_and_target_are_mutually_exclusive(self):
        with patch(
            "sys.argv",
            ["track_and_servo.py", "--stack-prompt", "stack red green blue", "--target", "green"],
        ):
            with self.assertRaises(SystemExit):
                cli.parse_args()

    def test_instruction_and_legacy_target_are_mutually_exclusive(self):
        with patch(
            "sys.argv",
            ["track_and_servo.py", "--instruction", "move blue", "--target", "green"],
        ):
            with self.assertRaises(SystemExit):
                cli.parse_args()


if __name__ == "__main__":
    unittest.main()
