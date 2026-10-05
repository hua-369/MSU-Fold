# Data generation (expert demos -> training pkl)

This directory generates the **sequential** folding datasets used for training.
Everything runs inside SoftGym (PyFlex): an analytical expert policy (Demonstrator)
folds randomly initialized cloth step by step, and we record RGB / depth /
pick-place keypoints / language instruction before and after each step, then pack
them into the pkl files the trainer reads.

Unimanual and bimanual data do **not** come from two separate experts: every step of
a bimanual task (`Mixed*`) is labelled `dual` (two arms pick and place at the same
time) or `single`, so a single episode can mix both. This is where the
"shared heatmap head + read k peaks per step" design gets its supervision.

```
data_generation/
├── generate_configs.py          random initial cloth states -> configs/<Cloth>.pkl
├── generate_demo_fold.py        unimanual demos -> raw_data/<Task>/<i>/
├── bimanual/
│   ├── demonstrator.py          Mixed* task definitions (keypoints + instruction templates)
│   ├── generate_demo_fold.py    bimanual demos -> raw_data_mixed/<Task>/<i>/
│   └── create_dataset.py        pack -> datasets/dual_data_sequential/All_<N>.pkl
├── Policy/demonstrator.py       unimanual task definitions
├── softgym/                     SoftGym environments (ClothEnv / flex_utils)
├── utils/visual.py              pixel projection and action rendering
└── gen_datasets.sh              runs all of the above end to end
```

## 1. Cloth3D meshes

The `Tshirt` / `Trousers` scenes need CLOTH3D meshes. Download them into
`data_generation/cloth3d/` and point `CLOTH3D_PATH` there:

```
cloth3d/
├── Tshirt/0028.obj ...
├── Trousers/...
└── keypoints/<Cloth>_keypoints.pkl
```

## 2. Generate everything

```bash
conda activate MSU-Fold
bash data_generation/gen_datasets.sh                  # 1000 demos per task
N_DEMOS=100 bash data_generation/gen_datasets.sh      # 100 demos (much faster)
```

Output:

| File | Content |
| --- | --- |
| `datasets/single_data_sequential/All_<N>.pkl` | unimanual sequential dataset (5 tasks) |
| `datasets/dual_data_sequential/All_<N>.pkl` | bimanual sequential dataset (4 Mixed tasks) |
| `datasets/softgym_cache/<Cloth>.pkl` | initial cloth states for closed-loop evaluation |

## 3. Step by step (debugging)

```bash
cd data_generation
python generate_configs.py --cloth_type Square --num_configs 10
python generate_demo_fold.py --task CornerFold --cloth_type Square \
    --randomize_pose --num_demonstrations 2
python bimanual/generate_demo_fold.py --task MixedTshirtFold \
    --randomize_pose --num_demonstrations 2 --record_video
```

## 4. How the data drives inference

- The **language instruction** of each step comes from `Policy/demonstrator.py`
  (unimanual) and `bimanual/demonstrator.py` (bimanual; `dual_instruction` randomly
  prepends `Using both arms, ...` or similar). Skills decompose a user instruction
  into exactly these step instructions.
- The **number of pick/place keypoints** of a step decides how many peaks are read
  from the heatmap at inference (k=1 -> one arm, k=2 -> two arms), see
  `skills/bimanual_skill.py::_n_arms`.
- Training targets are Gaussian mixtures centered at **all** annotated keypoints of
  the step (`strategy: gmm`), so bimanual steps are supervised with a bimodal target.

## 5. Credits

`softgym/`, `Policy/demonstrator.py` and `utils/visual.py` are taken from
[dengyh16code/language_deformable](https://github.com/dengyh16code/language_deformable)
and [SoftGym](https://github.com/Xingyu-Lin/softgym) under their original licenses.
`bimanual/` is added by this project.
