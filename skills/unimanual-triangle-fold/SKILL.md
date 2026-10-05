---
name: unimanual-triangle-fold
description: "Fold a square cloth along its diagonal into a triangle with a single arm. Use when the user asks for a diagonal fold, triangle fold, or folding the cloth into a triangle."
metadata:
  alias: TriangleFold
  skill_class: TriangleFold
  arms: unimanual
  cloth_type: Square
  num_steps: 2
  config: outputs/unimanual/config.yaml
  checkpoint: outputs/unimanual/checkpoints/best.pth
---

# Unimanual Triangle Folding

## When to use

The instruction asks for a diagonal fold / triangle fold and only one arm is available.

## Decomposed step instructions

One user instruction is decomposed into the following two instructions,
which the model executes in order:

1. `Fold the upper left corner of the fabric to its diagonal corner.`
2. `Fold the upper right corner of the fabric to its diagonal corner.`

## Input / output

- Input: the initial observation image (RGB, depth if available) and one user instruction.
- Output: pick / place pixel points for each step; 1 point per step = one arm.
- Pixel points are converted to robot coordinates by the execution backend.
- After each step a new image is captured and used as input for the next step.
