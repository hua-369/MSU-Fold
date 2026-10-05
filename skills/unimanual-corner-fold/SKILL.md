---
name: unimanual-corner-fold
description: "Fold the four corners of a square cloth to the center one by one with a single arm. Use when the user wants the corners of a square cloth / napkin folded to the center and only one arm is available."
metadata:
  alias: CornerFold
  skill_class: CornerFold
  arms: unimanual
  cloth_type: Square
  num_steps: 4
  config: outputs/unimanual/config.yaml
  checkpoint: outputs/unimanual/checkpoints/best.pth
---

# Unimanual Corner Folding

## When to use

The instruction is about folding the corners of a square cloth / napkin to the
center and only one arm is available.

## Decomposed step instructions

One user instruction is decomposed into four instructions, one corner per step,
which the model executes in order:

1. `Fold the upper left corner of the fabric towards the center.`
2. `Fold the upper right corner of the fabric towards the center.`
3. `Fold the lower right corner of the fabric towards the center.`
4. `Fold the lower left corner of the fabric towards the center.`

## Input / output

- Input: the initial observation image (RGB, depth if available) and one user instruction.
- Output: pick / place pixel points for each step; 1 point per step = one arm.
- Pixel points are converted to robot coordinates by the execution backend.
- After each step a new image is captured and used as input for the next step.
