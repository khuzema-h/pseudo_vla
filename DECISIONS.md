# Decisions

Choices that the skill-registry brief left open, recorded so later changes do not have to rediscover them.

## Legacy pick-and-place stays on its old questions

`task_intent` still offers `pick_place` beside the new `skill_sequence` option. A `pick_place` answer keeps today's question order: `plan_more`, `moving_object`, `destination_kind`, and `destination_target` only when more than one target exists. The same moves are also stored as `pick` then `place` steps. `skill_sequence` is the generic loop and is the only path that asks `skill_next`.

## One repair is automatic

After a step fails `requires`, the planner applies a repair without Tev1 when exactly one repair can run in the current world. It asks `prerequisite_resolution` only when two or more repairs are legal. A repair is one parked object, one cleared blocker, one opened container, or one pick that a later skill needs. The cap is three repairs for a single requested step.

Parking and clearing use destination kind `at` and the first location in the scene catalog. There is no second Tev1 call to choose that surface.

## Stored plans contain primitive skills

`pick_place` is planned as one choice, then stored as `pick` followed by `place`. The executor never has to know which steps were macros.
