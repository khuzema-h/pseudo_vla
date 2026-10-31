import os
import tempfile
import unittest
from pathlib import Path

from pseudo_vla.runlog import append_run_log, start_run_log, stop_run_log


class RunLogTests(unittest.TestCase):
    def test_each_run_saves_terminal_output_and_step_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            try:
                first = start_run_log(directory)
                print("visible line", flush=True)
                os.write(1, b"fd-stdout\n")
                os.write(2, b"fd-stderr\n")
                append_run_log("step detail")
                stop_run_log()
                second = start_run_log(directory)
                print("second run", flush=True)
                stop_run_log()
            finally:
                stop_run_log()

            first_text = first.read_text(encoding="utf-8")
            second_text = second.read_text(encoding="utf-8")
            self.assertNotEqual(first, second)
            self.assertIn("visible line", first_text)
            self.assertIn("fd-stdout", first_text)
            self.assertIn("fd-stderr", first_text)
            self.assertIn("step detail", first_text)
            self.assertNotIn("second run", first_text)
            self.assertIn("second run", second_text)
            self.assertTrue((directory / "latest.log").samefile(second))


if __name__ == "__main__":
    unittest.main()
