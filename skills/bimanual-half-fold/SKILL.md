---
name: bimanual-half-fold
description: "Fold a cloth in half with two arms: first fold it along one edge with both arms, then fold it again along the perpendicular direction with one arm. Use when the user wants a cloth folded in half and two arms are available."
metadata:
  alias: BimanualHalfFold
  dual_task: MixedSquareHalfFold
  arms: bimanual
  cloth_type: Square
  num_steps: 2
  config: outputs/bimanual/config.yaml
  checkpoint: outputs/bimanual/checkpoints/best.pth
---

# Bimanual Half Folding

## When to use

The instruction is about folding a cloth in half (fold in half, halve the cloth)
and two arms are available.

## Decomposed step instructions

One user instruction is decomposed into the following two instructions,
which the model executes in order:

1. `Fold the cloth in half from the upper side to the lower side.` (two arms)
2. `Fold the cloth in half again from the left side to the right side.` (one arm)

## Input / output

- Input: the initial observation image (RGB, depth if available) and one user instruction.
- Output: pick / place pixel points for each step; 2 points = two arms, 1 point = one arm.
- Pixel points are converted to robot coordinates by the execution backend.
- After each step a new image is captured and used as input for the next step.
