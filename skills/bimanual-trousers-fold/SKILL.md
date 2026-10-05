---
name: bimanual-trousers-fold
description: "Fold trousers with two arms: first fold the trousers sideways into one long strip with both arms, then fold the strip in half with one arm. Use when the user wants to fold trousers/pants and two arms cooperate."
metadata:
  alias: BimanualTrousersFold
  dual_task: MixedTrousersFold
  arms: bimanual
  cloth_type: Trousers
  num_steps: 2
  config: outputs/bimanual/config.yaml
  checkpoint: outputs/bimanual/checkpoints/best.pth
---

# Bimanual Trousers Folding

## When to use

The instruction is about trousers / pants and two arms cooperate.

## Decomposed step instructions

One user instruction is decomposed into the following two instructions,
which the model executes in order:

1. `Fold the Trousers in half, left to right, so they become one long strip.` (two arms)
2. `Fold the long strip in half by bringing the waistband down to the hem.` (one arm)

Step 1 uses two arms, step 2 uses one arm; both are served by the same bimanual model.

## Input / output

- Input: the initial observation image (RGB, depth if available) and one user instruction.
- Output: pick / place pixel points for each step; 2 points = two arms, 1 point = one arm.
- Pixel points are converted to robot coordinates by the execution backend.
- After each step a new image is captured and used as input for the next step.
