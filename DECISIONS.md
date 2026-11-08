# Decisions

Choices that the skill-registry brief left open, recorded so later changes do not have to rediscover them.

## Legacy pick-and-place stays on its old questions

`task_intent` still offers `pick_place` beside the new `skill_sequence` option. A `pick_place` answer keeps today's question order: `plan_more`, `moving_object`, `destination_kind`, and `destination_target` only when more than one target exists. The same moves are also stored as `pick` then `place` steps. `skill_sequence` is the generic loop and is the only path that asks `skill_next`.

## One repair is automatic

After a step fails `requires`, the planner applies a repair without Tev1 when exactly one repair can run in the current world. It asks `prerequisite_resolution` only when two or more repairs are legal. A repair is one parked object, one cleared blocker, one opened container, or one pick that a later skill needs. The cap is three repairs for a single requested step.

Parking and clearing use destination kind `at` and the first location in the scene catalog. There is no second Tev1 call to choose that surface.

## This arm has one parallel gripper

`grasp_family` is a real slot on `rotate` and `use_tool`, and its only option is `parallel`, so the planner stores it without asking. `pour` likewise has a single `upright` transport constraint. `hang` offers `upright` and `free`, so that question is asked.

Container identity is copied onto the world from the scene catalog. `held`, `on`, `inside`, and `opened` stay the facts that skills change. A listed container that is not in `opened` is closed, which lets `insert` tell a shut drawer from a peg.

`ask_clarification` ends planning immediately. `repeat` records a count of one, two, or three and does not copy earlier steps; the executor reads that count.

## Stored plans contain primitive skills

`pick_place` is planned as one choice, then stored as `pick` followed by `place`. The executor never has to know which steps were macros.

## Tev1 call logs are opt-in

Every Tev1 call is appended as JSONL when `TEV1_CALL_LOG` points at a file. Each line has the question key, state, options, answer, and latency. The prompt suite turns that log on. Ordinary runs stay quiet so tests do not write a log as a side effect. The checked-in baseline is 50/50 exact step matches and 248 Tev1 calls, 4.96 calls per plan.
