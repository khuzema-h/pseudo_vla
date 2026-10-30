import contextlib
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pseudo_vla import cli
from pseudo_vla import controller as track_and_servo
from pseudo_vla import stacking_llm
from pseudo_vla import stacking_workflow

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


class StackingLlmTests(unittest.TestCase):
    def _mock_ollama(self, response):
        return patch.object(
            stacking_llm,
            "urlopen",
            return_value=io.BytesIO(json.dumps(response).encode("utf-8")),
        )

    def test_prompt_parser_posts_structured_chat_and_preserves_order(self):
        response = {
            "message": {
                "content": json.dumps({"sequence": ["green", "red block", "BLUE cube"]})
            }
        }
        with self._mock_ollama(response) as post, patch.dict(
            "os.environ", {"OLLAMA_HOST": "http://127.0.0.1:11434"}
        ):
            result = stacking_llm.parse_stacking_prompt("green then red then blue")

        request = post.call_args.args[0]
        payload = json.loads(request.data)
        config = json.loads((_CONFIG_DIR / "llama_stacking.json").read_text(encoding="utf-8"))
        self.assertEqual(result, ["green", "red", "blue"])
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(payload["model"], config["model"])
        self.assertEqual(payload["messages"][0]["content"], config["system_prompt"])
        self.assertEqual(payload["format"], config["response_schema"])
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["format"]["properties"]["sequence"]["items"]["enum"], ["red", "green", "blue"])

    def test_object_prompt_uses_llama_structured_query(self):
        response = {
            "message": {
                "content": json.dumps(
                    {
                        "intent": "find_object",
                        "query": "small green cube",
                        "stacking_relations": [],
                    }
                )
            }
        }
        with self._mock_ollama(response) as post, patch.dict(
            "os.environ", {"OLLAMA_HOST": "localhost:11434"}
        ):
            result = stacking_llm.normalize_object_prompt("find the little green block")

        payload = json.loads(post.call_args.args[0].data)
        config = json.loads((_CONFIG_DIR / "llama_task.json").read_text(encoding="utf-8"))
        self.assertEqual(result, "small green cube")
        self.assertEqual(post.call_args.args[0].full_url, "http://localhost:11434/api/chat")
        self.assertEqual(payload["model"], config["model"])
        self.assertEqual(payload["messages"][0]["content"], config["system_prompt"])
        self.assertEqual(payload["messages"][1]["content"], "find the little green block")
        self.assertEqual(payload["format"], config["response_schema"])
        self.assertFalse(payload["stream"])

    def test_object_prompt_rejects_empty_or_invalid_llm_query(self):
        responses = (
            {
                "message": {
                    "content": json.dumps(
                        {"intent": "find_object", "query": " ", "stacking_relations": []}
                    )
                }
            },
            {"message": {"content": json.dumps({"response": "green block"})}},
            {"message": {"content": "not json"}},
        )
        for response in responses:
            with self.subTest(response=response), self._mock_ollama(response):
                with self.assertRaises(stacking_llm.StackingModelError):
                    stacking_llm.normalize_object_prompt("find a green block")

    def test_interactive_task_interpreter_parses_only_requested_pair(self):
        response = {
            "message": {
                "content": json.dumps(
                    {
                        "intent": "stack_blocks",
                        "query": "",
                        "stacking_relations": [
                            {"upper": "blue", "lower": "red"},
                        ],
                    }
                )
            }
        }
        with self._mock_ollama(response) as post, patch.dict(
            "os.environ", {"OLLAMA_HOST": "localhost:11434"}
        ):
            parsed = stacking_llm.interpret_user_prompt(
                "stack blue on top of red"
            )

        payload = json.loads(post.call_args.args[0].data)
        config = json.loads((_CONFIG_DIR / "llama_task.json").read_text(encoding="utf-8"))
        self.assertEqual(parsed.intent, "stack_blocks")
        self.assertEqual(
            parsed.stacking_relations,
            [stacking_llm.StackingRelation(support="red", moving="blue")],
        )
        self.assertIsNone(parsed.query)
        self.assertEqual(payload["messages"][1]["content"], "stack blue on top of red")
        self.assertEqual(payload["format"], config["response_schema"])

    def test_interactive_task_interpreter_resolves_top_then_that_to_base_red_green(self):
        response = {
            "message": {
                "content": json.dumps(
                    {
                        "intent": "stack_blocks",
                        "query": "",
                        "stacking_relations": [
                            {"upper": "red", "lower": "blue"},
                            {"upper": "green", "lower": "red"},
                        ],
                    }
                )
            }
        }
        with self._mock_ollama(response):
            parsed = stacking_llm.interpret_user_prompt(
                "stack red on top of blue then put green on top of that"
            )

        self.assertEqual(
            parsed.stacking_relations,
            [
                stacking_llm.StackingRelation(support="blue", moving="red"),
                stacking_llm.StackingRelation(support="red", moving="green"),
            ],
        )

    def test_interactive_task_interpreter_returns_object_query(self):
        response = {
            "message": {
                "content": json.dumps(
                    {
                        "intent": "find_object",
                        "query": "small green cube",
                        "stacking_relations": [],
                    }
                )
            }
        }
        with self._mock_ollama(response):
            parsed = stacking_llm.interpret_user_prompt("find the little green block")

        self.assertEqual(parsed.intent, "find_object")
        self.assertEqual(parsed.query, "small green cube")
        self.assertIsNone(parsed.stacking_relations)

    def test_interactive_task_interpreter_rejects_invalid_stacking_relations(self):
        invalid_relations = (
            [],
            [
                {"upper": "red", "lower": "blue"},
                {"upper": "blue", "lower": "red"},
            ],
            [
                {"upper": "red", "lower": "blue"},
                {"upper": "green", "lower": "blue"},
            ],
        )
        for relations in invalid_relations:
            response = {
                "message": {
                    "content": json.dumps(
                        {
                            "intent": "stack_blocks",
                            "query": "",
                            "stacking_relations": relations,
                        }
                    )
                }
            }
            with self.subTest(relations=relations), self._mock_ollama(response):
                with self.assertRaises(stacking_llm.StackingModelError):
                    stacking_llm.interpret_user_prompt("stack blue on green")

    def test_prompt_parser_rejects_missing_duplicate_and_unknown_colors(self):
        cases = (
            {"sequence": ["green", "red"]},
            {"sequence": ["green", "green", "blue"]},
            {"sequence": ["green", "purple", "blue"]},
        )
        for parsed in cases:
            with self.subTest(parsed=parsed), self._mock_ollama(
                {"message": {"content": json.dumps(parsed)}}
            ):
                with self.assertRaises(ValueError):
                    stacking_llm.parse_stacking_prompt("stack red, green, and blue")

    def test_prompt_parser_rejects_bad_json(self):
        with self._mock_ollama({"message": {"content": "not json"}}):
            with self.assertRaises(stacking_llm.StackingModelError):
                stacking_llm.parse_stacking_prompt("stack green red blue")

    def test_prompt_parser_rejects_missing_colors_before_calling_llama(self):
        with patch.object(stacking_llm, "urlopen") as post:
            with self.assertRaisesRegex(ValueError, "must specify red, green, and blue"):
                stacking_llm.parse_stacking_prompt("stack the purple cube")
        post.assert_not_called()

    def test_decision_model_receives_current_state_and_returns_typed_event(self):
        response = {
            "answers": {
                "next_event": {
                    "type": "choice",
                    "choice": "stack_next",
                    "probabilities": {"stack_next": 0.99, "stop": 0.01},
                }
            }
        }
        goal_order = ["green", "red", "blue"]
        placed = ["green"]
        with self._mock_ollama(response) as post, patch.dict(
            "os.environ", {"OLLAMA_HOST": "localhost:11434"}
        ):
            event = stacking_llm.choose_stack_event(
                goal_order,
                placed,
                "green",
                {"event": "establish_base", "success": True},
            )

        request = post.call_args.args[0]
        payload = json.loads(request.data)
        config = json.loads((_CONFIG_DIR / "d3_stacking.json").read_text(encoding="utf-8"))
        self.assertEqual(event, "stack_next")
        self.assertEqual(request.full_url, "http://localhost:11434/v1/systemone")
        self.assertEqual(payload["model"], config["model"])
        self.assertEqual(payload["state"]["goal_order_bottom_to_top"], goal_order)
        self.assertEqual(payload["state"]["placed_bottom_to_top"], placed)
        self.assertEqual(payload["state"]["current_support"], "green")
        self.assertEqual(payload["state"]["task"], config["task"])
        self.assertEqual(
            payload["questions"][config["question_id"]]["instructions"],
            config["instructions"],
        )
        self.assertEqual(payload["questions"]["next_event"]["criteria"], {
            "stack_next": "Place the next required cube (red) on the current support (green).",
            "stop": "Stop the stacking workflow at the current safe boundary.",
        })

    def test_decision_model_rejects_unoffered_event(self):
        with self._mock_ollama(
            {"answers": {"next_event": {"type": "choice", "choice": "grasp"}}}
        ):
            with self.assertRaises(stacking_llm.StackingModelError):
                stacking_llm.choose_stack_event(["green", "red", "blue"], [], None, None)

    def test_decision_model_rejects_non_prefix_task_state(self):
        with self.assertRaises(ValueError):
            stacking_llm.choose_stack_event(
                ["green", "red", "blue"], ["red"], "red", None
            )

    def test_decision_model_receives_specific_pair_stage(self):
        response = {
            "answers": {
                "next_event": {
                    "type": "choice",
                    "choice": "grasp",
                    "probabilities": {"grasp": 0.99, "stop": 0.01},
                }
            }
        }
        with self._mock_ollama(response) as post:
            event = stacking_llm.choose_stack_event(
                ["green", "red", "blue"],
                ["green"],
                "green",
                {"event": "approach", "success": True},
                expected_event="grasp",
                current_pair={"support": "green", "moving": "red"},
            )

        payload = json.loads(post.call_args.args[0].data)
        self.assertEqual(event, "grasp")
        self.assertEqual(payload["state"]["phase"], "authorize_grasp")
        self.assertEqual(
            payload["state"]["current_pair"],
            {"support": "green", "moving": "red"},
        )
        self.assertEqual(set(payload["questions"]["next_event"]["criteria"]), {"grasp", "stop"})

    def test_decision_model_authorizes_requested_pair_without_full_stack(self):
        response = {
            "answers": {
                "next_event": {
                    "type": "choice",
                    "choice": "approach",
                }
            }
        }
        with self._mock_ollama(response) as post:
            event = stacking_llm.choose_stack_action(
                "red",
                "blue",
                "approach",
                {"event": "placement_complete", "success": True},
            )

        payload = json.loads(post.call_args.args[0].data)
        self.assertEqual(event, "approach")
        self.assertEqual(payload["state"]["task"], "requested_pair_stacking")
        self.assertEqual(
            payload["state"]["requested_pair"],
            {"support": "red", "moving": "blue"},
        )
        self.assertEqual(
            set(payload["questions"]["next_event"]["criteria"]),
            {"approach", "stop"},
        )

    def test_requested_pair_workflow_executes_only_explicit_placement(self):
        relations = [
            stacking_llm.StackingRelation(support="red", moving="blue")
        ]
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

    def test_decision_model_reports_endpoint_failure(self):
        with patch.object(
            stacking_llm,
            "urlopen",
            side_effect=OSError("connection refused"),
        ):
            with self.assertRaisesRegex(stacking_llm.StackingModelError, "connection refused"):
                stacking_llm.choose_stack_event(["green", "red", "blue"], [], None, None)

    def test_prompted_workflow_dispatches_pairs_in_bottom_to_top_order(self):
        placements = []
        authorized = []
        say = lambda message: None
        def fake_stack_phase(scene, perception, servo, support, moving, *args):
            placements.append((support, moving))
            authorize = args[-1]
            for stage in stacking_llm.STACK_ACTION_EVENTS:
                self.assertTrue(authorize(stage))
                authorized.append(stage)
            return True

        def choose_event(goal_order, placed, support, last_result, **kwargs):
            if kwargs.get("expected_event") is not None:
                last_result["expected_event"] = kwargs["expected_event"]
                return kwargs["expected_event"]
            if not placed:
                return "establish_base"
            if len(placed) < len(goal_order):
                return "stack_next"
            return "finish"

        with (
            patch.object(stacking_llm, "choose_stack_event", side_effect=choose_event),
            patch.object(
                track_and_servo,
                "_run_stack_phase",
                side_effect=fake_stack_phase,
            ),
        ):
            completed = track_and_servo._run_prompted_stack_sequence(
                object(),
                object(),
                object(),
                ["green", "red", "blue"],
                object(),
                10,
                say,
                say,
                None,
            )

        self.assertTrue(completed)
        self.assertEqual(placements, [("green", "red"), ("red", "blue")])
        self.assertEqual(authorized, list(stacking_llm.STACK_ACTION_EVENTS) * 2)

    def test_d3_stop_prevents_base_or_cube_motion(self):
        say = lambda message: None
        with (
            patch.object(stacking_llm, "choose_stack_event", return_value="stop"),
            patch.object(track_and_servo, "_run_stack_phase") as stack_phase,
        ):
            completed = track_and_servo._run_prompted_stack_sequence(
                object(),
                object(),
                object(),
                ["green", "red", "blue"],
                object(),
                10,
                say,
                say,
                None,
            )

        self.assertFalse(completed)
        stack_phase.assert_not_called()

    def test_d3_stop_at_approach_boundary_prevents_pair_motion(self):
        events = []
        say = lambda message: None

        def choose_event(goal_order, placed, support, last_result, **kwargs):
            if kwargs.get("expected_event") is not None:
                return "stop"
            return "establish_base" if not placed else "stack_next"

        def fake_stack_phase(scene, perception, servo, support, moving, *args):
            authorize = args[-1]
            events.append("approach")
            return authorize("approach")

        with (
            patch.object(stacking_llm, "choose_stack_event", side_effect=choose_event),
            patch.object(track_and_servo, "_run_stack_phase", side_effect=fake_stack_phase),
        ):
            completed = track_and_servo._run_prompted_stack_sequence(
                object(),
                object(),
                object(),
                ["green", "red", "blue"],
                object(),
                10,
                say,
                say,
                None,
            )

        self.assertFalse(completed)
        self.assertEqual(events, ["approach"])

    def test_prompt_parse_failure_exits_before_simulator_launch(self):
        with (
            patch.object(
                cli,
                "parse_args",
                return_value=SimpleNamespace(stack_prompt="bad request", headless=False),
            ),
            patch.object(cli, "launch") as launch,
            patch.object(
                stacking_llm,
                "parse_stacking_prompt",
                side_effect=ValueError("invalid stack"),
            ),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            with self.assertRaises(SystemExit) as exit_error:
                cli.main()

        self.assertEqual(exit_error.exception.code, 2)
        self.assertIn("could not parse --stack-prompt", stderr.getvalue())
        launch.assert_not_called()

    def test_stack_prompt_and_target_are_mutually_exclusive(self):
        with patch(
            "sys.argv",
            [
                "track_and_servo.py",
                "--target",
                "red block",
                "--stack-prompt",
                "green red blue",
            ],
        ):
            with self.assertRaises(SystemExit) as exit_error:
                cli.parse_args()

        self.assertEqual(exit_error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
