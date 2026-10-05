---
name: bimanual-corner-fold
description: "Fold the four corners of a square cloth to the center with two arms, two corners at a time, in two steps. Use when the user wants the corners of a square cloth / napkin folded and two arms are available."
metadata:
  alias: BimanualCornerFold
  dual_task: MixedSquareCornerFold
  arms: bimanual
  cloth_type: Square
  num_steps: 2
  config: outputs/bimanual/config.yaml
  checkpoint: outputs/bimanual/checkpoints/best.pth
---

# Bimanual Corner Folding

## When to use

The instruction is about folding the corners of a square cloth / napkin and
two arms are available (two corners folded at the same time).

## Decomposed step instructions

One user instruction is decomposed into the following two instructions,
which the model executes in order:

1. `Fold the upper left corner and the upper right corner of the cloth towards the center.`
2. `Fold the lower left corner and the lower right corner of the cloth towards the center.`

The two steps use complementary corner pairs so that all four corners are folded once.

## Input / output

- Input: the initial observation image (RGB, depth if available) and one user instruction.
- Output: pick / place pixel points for each step; 2 points per step = two arms.
- Pixel points are converted to robot coordinates by the execution backend.
- After each step a new image is captured and used as input for the next step.
