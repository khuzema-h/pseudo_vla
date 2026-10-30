"""Command-line parsing and Isaac Sim application lifecycle."""

from __future__ import annotations

import argparse
import sys

from pseudo_vla.controller import run
from pseudo_vla.scene import launch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find, stack, or servo to colored blocks with the environment camera."
    )
    command_group = parser.add_mutually_exclusive_group()
    command_group.add_argument(
        "--target",
        type=str,
        help="First text prompt for SAM, before the prompt",
    )
    command_group.add_argument(
        "--stack-prompt",
        type=str,
        help="Natural-language bottom-to-top cube order, parsed by local Llama before Isaac Sim starts",
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
    if args.stack_prompt is not None:
        from pseudo_vla.stacking_llm import StackingModelError, parse_stacking_prompt

        try:
            args.stack_order = parse_stacking_prompt(args.stack_prompt)
        except (StackingModelError, ValueError) as error:
            print(f"could not parse --stack-prompt: {error}", file=sys.stderr, flush=True)
            raise SystemExit(2) from error
    app = launch(args.headless)
    try:
        run(app, args)
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        app.close()
