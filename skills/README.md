# Folding skills

Each skill is a directory with a `SKILL.md` (YAML frontmatter + Markdown body),
following the [Agent Skills](https://agentskills.io/specification) spec.

## Skill list

| Skill | Arms | Scene | Steps | Default checkpoint |
| --- | --- | --- | --- | --- |
| `unimanual-tshirt-fold` | single | Tshirt | 4 | `outputs/unimanual/checkpoints/best.pth` |
| `unimanual-trousers-fold` | single | Trousers | 3 | " |
| `unimanual-corner-fold` | single | Square | 4 | " |
| `unimanual-triangle-fold` | single | Square | 2 | " |
| `unimanual-straight-fold` | single | Rectangular | 3 | " |
| `bimanual-tshirt-fold` | dual | Tshirt | 2 | `outputs/bimanual/checkpoints/best.pth` |
| `bimanual-trousers-fold` | dual + single | Trousers | 2 | " |
| `bimanual-corner-fold` | dual | Square | 2 | " |
| `bimanual-half-fold` | dual + single | Square | 2 | " |

## SKILL.md metadata

```yaml
metadata:
  alias: BimanualTshirtFold      # legacy CamelCase name
  dual_task: MixedTshirtFold     # bimanual task key (unimanual uses skill_class)
  arms: bimanual                 # unimanual / bimanual
  cloth_type: Tshirt             # scene type
  num_steps: 2                   # number of decomposed step instructions
  config: outputs/bimanual/config.yaml
  checkpoint: outputs/bimanual/checkpoints/best.pth
```

`skills/__init__.py` scans every `SKILL.md`, binds it to a Python skill class and
exports `SKILLS` / `SKILL_DESCRIPTIONS` / `SKILL_SPECS`.

## Backends

A skill only emits **pixels**; `skills/backends.py` converts them to robot
coordinates, so the same weights run in simulation and on hardware.

```python
class ExecutionBackend:
    def observe(self)                        # capture one observation
    def to_world(pixel, observation)         # pixel -> robot coordinates
    def move(picks_world, places_world)      # 1 point -> single, 2 points -> dual
```

- `SoftgymBackend`: simulation, depth back-projection.
- `RealRobotBackend`: inject `camera.capture()`, `calibration.pixel_to_world()` and
  `arms.pick_and_place_single|dual()`.

## Usage

```bash
python inference_skill_softgym.py --list-skills
python inference_skill_softgym.py --instruction "Fold the T-shirt."
python inference_skill_softgym.py --skill bimanual-tshirt-fold --num-evals 1
python inference_skill_softgym.py --backend real --robot-factory myrobot:build_backend \
    --instruction "Fold the T-shirt with both arms."
```

Routing traces are logged to `outputs/skill_eval/llm_router.jsonl`.
