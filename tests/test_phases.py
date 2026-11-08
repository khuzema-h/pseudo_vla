import unittest
from unittest.mock import patch

from pseudo_vla import phases
from pseudo_vla.tev1_client import Tev1Error


class PhaseEventTests(unittest.TestCase):
    def test_success_offers_only_the_expected_phase_or_stop(self):
        seen = {}

        def ask_choice(question_id, state, criteria):
            seen["question"] = question_id
            seen["criteria"] = criteria
            return "grasp"

        with patch.object(phases, "ask_choice", side_effect=ask_choice):
            choice = phases.choose_phase_event("pick", 1, {"ok": True, "data": {}}, {})
        self.assertEqual(choice, "grasp")
        self.assertEqual(seen["question"], "phase_event")
        self.assertEqual(set(seen["criteria"]), {"grasp", "stop"})

    def test_each_failure_offers_only_its_recovery_until_the_cap(self):
        for failure, recoveries in phases.RECOVERY.items():
            offered = {}

            def ask_choice(question_id, state, criteria, offered=offered):
                offered["criteria"] = set(criteria)
                return "stop"

            with patch.object(phases, "ask_choice", side_effect=ask_choice):
                phases.choose_phase_event(
                    "pick",
                    1,
                    {"ok": False, "failure": failure, "data": {}},
                    {failure: 0},
                )
            self.assertEqual(offered["criteria"], set(recoveries) | {"stop"})

            def unexpected(*_args, **_kwargs):
                raise AssertionError("exhausted retries must stop without asking")

            with patch.object(phases, "ask_choice", side_effect=unexpected):
                decision = phases.choose_phase_event(
                    "pick",
                    1,
                    {"ok": False, "failure": failure, "data": {}},
                    {failure: phases.MAX_RETRIES[failure]},
                )
            self.assertEqual(decision, "stop")

    def test_unknown_failure_is_rejected(self):
        with self.assertRaises(Tev1Error):
            phases.choose_phase_event("pick", 0, {"ok": False, "failure": "explode"}, {})


if __name__ == "__main__":
    unittest.main()
