---
name: unimanual-trousers-fold
description: "Fold trousers with a single arm: fold them sideways into one long strip, then fold the strip in half from the waistband down to the hem. Use when the user wants to fold trousers/pants and only one arm is available."
metadata:
  alias: TrousersFold
  skill_class: TrousersFold
  arms: unimanual
  cloth_type: Trousers
  num_steps: 3
  config: outputs/unimanual/config.yaml
  checkpoint: outputs/unimanual/checkpoints/best.pth
---

# Unimanual Trousers Folding

## When to use

The instruction is about trousers / pants and only one arm is available.

## Decomposed step instructions

One user instruction is decomposed into the following three instructions,
which the model executes in order:

1. `Fold the Trousers in half, left to right.`
2. `Fold the Trousers in half, right to left.`
3. `Fold the long strip in half by bringing the waistband down to the hem.`

## Input / output

- Input: the initial observation image (RGB, depth if available) and one user instruction.
- Output: pick / place pixel points for each step; 1 point per step = one arm.
- Pixel points are converted to robot coordinates by the execution backend.
- After each step a new image is captured and used as input for the next step.
