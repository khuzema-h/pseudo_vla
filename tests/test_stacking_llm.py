import contextlib
import importlib.util
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pseudo_vla import stacking_llm

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "track_and_servo.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("track_and_servo_llm_tests", _SCRIPT_PATH)
track_and_servo = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(track_and_servo)


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
        self.assertEqual(result, ["green", "red", "blue"])
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(payload["model"], "llama3.2:3b")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["format"]["properties"]["sequence"]["items"]["enum"], ["red", "green", "blue"])

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
        self.assertEqual(event, "stack_next")
        self.assertEqual(request.full_url, "http://localhost:11434/v1/systemone")
        self.assertEqual(payload["model"], "d3:4b")
        self.assertEqual(payload["state"]["goal_order_bottom_to_top"], goal_order)
        self.assertEqual(payload["state"]["placed_bottom_to_top"], placed)
        self.assertEqual(payload["state"]["current_support"], "green")
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
                track_and_servo,
                "parse_args",
                return_value=SimpleNamespace(stack_prompt="bad request", headless=False),
            ),
            patch.object(track_and_servo, "launch") as launch,
            patch.object(
                stacking_llm,
                "parse_stacking_prompt",
                side_effect=ValueError("invalid stack"),
            ),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            with self.assertRaises(SystemExit) as exit_error:
                track_and_servo.main()

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
                track_and_servo.parse_args()

        self.assertEqual(exit_error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
