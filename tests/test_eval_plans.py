import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from pseudo_vla.eval_plans import evaluate, format_baseline

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "tests" / "prompts" / "cases.py"
BASELINE_PATH = ROOT / "tests" / "prompts" / "baseline.txt"


def _cases():
    spec = importlib.util.spec_from_file_location("prompt_cases", CASES_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CASES


class PromptSuiteTests(unittest.TestCase):
    def test_suite_matches_the_recorded_baseline(self):
        cases = _cases()
        self.assertGreaterEqual(len(cases), 50)
        with tempfile.TemporaryDirectory() as directory:
            log_path = str(Path(directory) / "tev1.jsonl")
            report = evaluate(cases, log_path=log_path)
            self.assertEqual(format_baseline(report), BASELINE_PATH.read_text(encoding="utf-8"))
            self.assertEqual(report["mismatches"], [])
            record = json.loads(Path(log_path).read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(
            set(record),
            {"question", "state", "options", "answer", "latency_ms"},
        )
        self.assertIn(record["answer"]["choice"], record["options"])


if __name__ == "__main__":
    unittest.main()
