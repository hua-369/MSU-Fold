---
name: bimanual-tshirt-fold
description: "Fold a T-shirt with two arms: fold both sleeves inward at the same time, then bring the hem up to the neckline. Use when the user wants to fold a T-shirt/top/shirt and two arms are available (both arms, two arms, dual-arm)."
metadata:
  alias: BimanualTshirtFold
  dual_task: MixedTshirtFold
  arms: bimanual
  cloth_type: Tshirt
  num_steps: 2
  config: outputs/bimanual/config.yaml
  checkpoint: outputs/bimanual/checkpoints/best.pth
---

# Bimanual T-shirt Folding

## When to use

The instruction is about a T-shirt / top / shirt and requires two arms
(both arms, two arms, dual-arm).

## Decomposed step instructions

One user instruction is decomposed into the following two instructions,
which the model executes in order:

1. `Fold both sleeves of the T-shirt towards the body.` (two arms, one sleeve each)
2. `Bring the bottom of the T-shirt up towards the neckline.` (two arms on the hem)

## Input / output

- Input: the initial observation image (RGB, depth if available) and one user instruction.
- Output: pick / place pixel points for each step; 2 points per step = two arms acting together.
- Pixel points are converted to robot coordinates by the execution backend
  (depth back-projection in simulation, hand-eye calibration on the real robot).
- After each step a new image is captured and used as input for the next step.
