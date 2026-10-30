# Natural-Language, Model-Orchestrated Block Stacking

## Goal

Allow a user to request a three-block stack in natural language, for example:

> Stack the cubes in the order green, red, and blue.

The execution roles are:

1. **Llama parses the user's prompt** into a validated block order.
2. **d3 orchestrates high-level task events** using the parsed goal and current
   task state.
3. **The existing controller executes the selected events** using the current
   perception, grasping, placement, and cuRobo motion functions. The controller
   remains responsible for physical safety; d3 does not directly command joints
   or bypass collision checks.

The order is **bottom-to-top**: the first named cube is the base on the table,
the second is stacked on it, and the third is stacked on the second. A prompted
sequence runs through all placements without an inter-placement user pause.

## Build status

- Implemented in `pseudo_vla/stacking_llm.py` using the Python standard library.
- Local Ollama was verified as version `0.40.11` with `llama3.2:3b` and `d3:4b`.
  Llama uses `POST /api/chat` with structured JSON output. d3 uses
  `POST /v1/systemone` with `state` and typed `questions`; the model returns a
  typed choice with probabilities.
- `scripts/track_and_servo.py` accepts `--stack-prompt`. Llama parses and
  validates the full order before Isaac Sim launches. d3 selects base, pair,
  and finish transitions and authorizes each pair's approach, grasp, lift,
  place, and placement-completion event at safe controller boundaries.
- Prompted stacks use bottom-to-top order and do not pause between placements.
  The existing bare `stack` command and `--target` behavior remain separate.
- Mock-based suite: 24 tests pass with `conda run -n ps_vla python -m unittest
  discover -s tests -v`. A live local-model check completed the
  green→red→blue decision sequence.
- Isaac Sim motion/contact validation remains outstanding; local model/API
  checks do not establish that physical stack execution succeeds.

## Existing repository structure

- `scripts/track_and_servo.py` is the Isaac Sim entry point. Its current CLI
  provides `--target`, `--headless`, and `--max-steps`.
- The existing `stack` command uses `_run_stack_phase(...)` and
  `_stack_tracked_cube(...)` to place blue on red, pause, and then place green on
  blue. There is no `pick_and_place(color)` function or generic sequence loop.
- The scene contains exactly red, green, and blue blocks.
- Existing SAM 2, OWL-ViT, CoTracker, RGB-D, wrist-camera alignment, and cuRobo
  functionality should be reused. Do not modify their internals for this
  feature.
- Current Python tests use `unittest` in `tests/test_stacking.py`.
- Run Python commands and tests using the user's `ps_vla` Conda environment.
- No Ollama client or `11434` integration exists in the project. The reported
  local models are `llama3.2:3b` and `d3:4b`; confirm they are actually installed
  and available before relying on them.

## Phase 0: Verify model/runtime interfaces (completed)

The installed local interfaces were verified before adding the clients:

1. Ollama reports version `0.40.11`; both models are installed locally.
2. `llama3.2:3b` accepts non-streaming structured responses at `/api/chat`.
3. `d3:4b` advertises decision capability and answers typed choices at
   `/v1/systemone` when sent `state` plus a `questions` object. The initial
   probe's guessed schema was rejected; the supported typed-choice schema was
   then confirmed with a successful local request.
4. The Python standard-library HTTP client uses a 120-second timeout, reports
   HTTP/transport/JSON errors explicitly, and does not silently substitute a
   decision or continue the robot workflow after an invalid response.

**Deliverable:** Verified local model interfaces and a concrete, testable d3
event contract.

## Phase 1: Parse the user goal with Llama

Add a small prompt-parsing module, for example
`pseudo_vla/stacking_llm.py`, with a function such as:

```python
def parse_stacking_prompt(user_prompt: str) -> list[str]:
    """Return a bottom-to-top order of the three supported block colors."""
```

Requirements:

- Request structured JSON from the verified local Llama interface.
- Accept exactly the supported colors `red`, `green`, and `blue`, exactly once
  each. Normalize case and common natural-language variations.
- Reject missing, repeated, or unknown colors with an explicit error. Do not
  start Isaac Sim if the prompt cannot be parsed.
- Keep parsing independent of Isaac Sim imports so it is straightforward to
  test in the `ps_vla` environment.
- Unit tests should mock the model response. Include valid wording variations,
  malformed JSON, unknown/repeated colors, and requests that omit one or more
  blocks.
- Test the real local model separately after confirming availability; keep model
  calls out of the required unit-test path.

**Deliverable:** Deterministic validation around the model's parsed
bottom-to-top order, with mocked tests.

## Phase 2: Define d3's event-orchestration contract

d3 is an **active high-level orchestrator**, not a log-only validator and not a
joint-space motion planner.

Provide d3 with:

- The parsed bottom-to-top goal.
- Current task state, including which block is the base, which blocks have been
  placed, the current block/support pair, and the last event result.
- A documented set of high-level events, such as selecting the next
  approach/grasp/lift/place/verification operation.

The exact event vocabulary and schema must be based on the verified d3 API. Keep
model output structured and limited to the orchestration contract. The robot
controller dispatches each event through existing movement/perception routines;
raw model output must never be treated as executable code or direct joint
commands.

Requirements:

- d3 decides the task-event progression from the requested goal and current
  state. The parsed order is the requested objective; d3 is responsible for
  orchestrating how the task advances toward it.
- The controller performs input/schema checks needed to safely dispatch events
  and enforces physical safety, collision handling, and safe stop behavior. A
  malformed, unavailable, or unsafe event must not be executed.
- On model timeout, endpoint error, or invalid output, stop the sequence safely
  and report the failure. Do not silently invent a d3 decision, fall back to a
  different model, or report success.
- Mock model responses for tests, including valid event progression, malformed
  output, invalid event, and endpoint failure.

**Deliverable:** A verified d3 client/decision contract and tests for event
selection and failure handling.

## Phase 3: Integrate prompt and event orchestration with the controller

Modify `scripts/track_and_servo.py` only at the CLI/state-machine integration
layer, reusing current perception and motion behavior.

1. Add a dedicated option such as `--stack-prompt` so natural-language stacking
   does not change the meaning of the existing `--target` option.
2. Parse and validate the prompt before launching Isaac Sim. A parsing failure
   exits with a clear error and does not create or move the robot.
3. For an order such as `green, red, blue`, leave green as the base and use the
   existing pairwise stacking operation to place red on green, then blue on red.
4. Replace the prompted workflow's fixed pause/continue transition with d3's
   event decisions so the requested stack runs without inter-placement user
   pauses.
5. Refactor the orchestration boundary only as needed to let d3 select discrete
   task events while the existing pairwise motion/perception operations carry
   them out. Do not duplicate or rewrite the grasp, tracking, RGB-D alignment,
   collision, or cuRobo implementations.
6. Preserve the existing bare `stack` command as the backward-compatible
   red→blue→green workflow, including its confirmation pause, unless the user
   explicitly requests changing that behavior.
7. Keep interactive `--target`, `home`, `quit`, and existing single-object
   prompt workflows working.

**Deliverable:** Prompt-driven stacking that uses Llama for the requested order
and d3 for high-level event orchestration.

## Phase 4: Validate

Run unit checks inside `ps_vla`:

```bash
conda run -n ps_vla python -m unittest discover -s tests -v
conda run -n ps_vla python -m py_compile pseudo_vla/stacking_llm.py scripts/track_and_servo.py
git diff --check
```

Add tests that establish:

- Prompt parsing returns the exact bottom-to-top order and rejects invalid goals
  before simulator launch.
- Every d3 event is handled by the correct orchestration path and reflected in
  the next state request.
- The order `green, red, blue` leaves green at the base, places red on green,
  then blue on red, with no confirmation pause in the prompted workflow.
- d3 failures and malformed decisions stop safely.
- Existing bare `stack` and single-object prompt behavior remain available.

Automated tests cannot establish simulator motion or model-server availability.

Example launch after local model interfaces have been verified:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n ps_vla python scripts/track_and_servo.py \
  --stack-prompt "stack green, then red, then blue"
```

## Out of scope

- Isaac Sim motion/contact validation remains a required manual runtime check.

- Replacing or modifying OWL-ViT, SAM 2, CoTracker, RGB-D, wrist-camera, or cuRobo
  internals.
- Having d3 produce trajectories, joint targets, or executable code.
- Adding a new Python dependency unless the verified client implementation
  requires it and the dependency is explicitly justified.
- Changing the existing bare `stack` behavior as part of prompt integration.
