---
name: unimanual-straight-fold
description: "Fold a rectangular cloth in half along one edge with a single arm. Use when the user asks for a straight fold, folding a rectangular cloth in half, or creasing the cloth along an edge."
metadata:
  alias: StraightFold
  skill_class: StraightFold
  arms: unimanual
  cloth_type: Rectangular
  num_steps: 3
  config: outputs/unimanual/config.yaml
  checkpoint: outputs/unimanual/checkpoints/best.pth
---

# Unimanual Straight Folding

## When to use

The instruction is about folding a rectangular cloth in half along one edge and
only one arm is available.

## Decomposed step instructions

One user instruction is decomposed into the following three instructions,
which the model executes in order:

1. `Crease the cloth in half from left to right.`
2. `Crease the cloth in half from right to left.`
3. `Fold the cloth in half from the upper side to the lower side.`

## Input / output

- Input: the initial observation image (RGB, depth if available) and one user instruction.
- Output: pick / place pixel points for each step; 1 point per step = one arm.
- Pixel points are converted to robot coordinates by the execution backend.
- After each step a new image is captured and used as input for the next step.
