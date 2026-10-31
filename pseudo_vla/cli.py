"""Command-line parsing and Isaac Sim application lifecycle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pseudo_vla.controller import run
from pseudo_vla.scene import launch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpret and execute pick-and-place instructions in the Isaac Sim scene."
    )
    command_group = parser.add_mutually_exclusive_group()
    command_group.add_argument(
        "--instruction",
        type=str,
        help="Natural-language pick-and-place instruction to execute after startup",
    )
    command_group.add_argument(
        "--target",
        type=str,
        help="Deprecated alias for --instruction",
    )
    command_group.add_argument(
        "--stack-prompt",
        type=str,
        help="Natural-language bottom-to-top cube order, parsed by local Llama before Isaac Sim starts",
    )
    parser.add_argument(
        "--scene",
        choices=("blocks", "kitchen"),
        default="blocks",
        help="blocks is the simple-room cube table; kitchen is the counter with plates, mugs, and utensils",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Steps to spend on one move before returning to the prompt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "scene", "blocks") == "kitchen" and args.stack_prompt is not None:
        print("the kitchen scene does not support --stack-prompt", file=sys.stderr, flush=True)
        raise SystemExit(2)
    if args.stack_prompt is not None:
        from pseudo_vla.stacking_llm import StackingModelError, parse_stacking_prompt

        try:
            args.stack_order = parse_stacking_prompt(args.stack_prompt)
        except (StackingModelError, ValueError) as error:
            print(f"could not parse --stack-prompt: {error}", file=sys.stderr, flush=True)
            raise SystemExit(2) from error
    from pseudo_vla.runlog import start_run_log, stop_run_log

    log_path = start_run_log(Path(__file__).resolve().parents[1] / "outputs" / "logs")
    print(f"saving terminal log to {log_path}", flush=True)
    app = launch(args.headless)
    try:
        run(app, args)
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        app.close()
        stop_run_log()
