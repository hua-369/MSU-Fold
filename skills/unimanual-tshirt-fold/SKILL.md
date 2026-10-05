---
name: unimanual-tshirt-fold
description: "Fold a T-shirt with a single arm: fold the left and right sleeves inward one after another, then bring the hem up to the neckline. Use when the user wants to fold a T-shirt/top/shirt and only one arm is available."
metadata:
  alias: TshirtFold
  skill_class: TshirtFold
  arms: unimanual
  cloth_type: Tshirt
  num_steps: 4
  config: outputs/unimanual/config.yaml
  checkpoint: outputs/unimanual/checkpoints/best.pth
---

# Unimanual T-shirt Folding

## When to use

The instruction is about a T-shirt / top / shirt and only one arm is available.

## Decomposed step instructions

One user instruction is decomposed into the following four instructions,
which the model executes in order:

1. `Fold the left sleeve towards the inside.`
2. `Fold the right sleeve towards the inside.`
3. `Bring the bottom of the T-shirt up towards the neckline.`
4. `Roll the bottom of the T-shirt up towards the top.`

## Input / output

- Input: the initial observation image (RGB, depth if available) and one user instruction.
- Output: pick / place pixel points for each step; 1 point per step = one arm.
- Pixel points are converted to robot coordinates by the execution backend
  (depth back-projection in simulation, hand-eye calibration on the real robot).
- After each step a new image is captured and used as input for the next step.
