#!/usr/bin/env python3
"""准备仿真的初始布料状态，供 skill 推理按路径选用。

对每种布料配置生成若干组"随机角度 + 随机大小 + 随机摆位"的初始状态：

    outputs/skill_eval/initial_states/<cloth_type>/case_XX/
        rgb.png      初始观测图像（推理时看到的那一帧）
        depth.npy    深度图
        mask.png     前景 mask
        meta.json    cloth_type / 随机参数 / 缓存索引
        scene.pkl    softgym 的 config+state，用于精确复现该初始状态

用法:
    python prepare_skill_states.py
    python prepare_skill_states.py --cloth-types Tshirt Trousers --num-cases 3
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from msufold.data.utils import get_mask_from_depth  # noqa: E402

CLOTH_TYPES = ["Square", "Rectangular", "Tshirt", "Trousers"]


def sample_angle(cloth_type, rng):
    """与推理时的随机化范围保持一致。"""
    if cloth_type == "Rectangular":
        return float(rng.uniform(-80, 80))
    if cloth_type in ("Tshirt", "Trousers"):
        return float(rng.uniform(-40, 40))
    return float(rng.uniform(0, 40))


def main():
    parser = argparse.ArgumentParser(description="Prepare randomized initial cloth states")
    parser.add_argument("--cloth-types", nargs="+", default=CLOTH_TYPES)
    parser.add_argument("--num-cases", type=int, default=3, help="每种布料生成几组")
    parser.add_argument("--size-range", type=float, nargs=2, default=[0.85, 1.15])
    parser.add_argument("--offset-range", type=float, default=0.04)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dataset-root", type=str, default=str(REPO_ROOT / "datasets"))
    parser.add_argument("--output-dir", type=str,
                        default=str(REPO_ROOT / "outputs" / "skill_eval" / "initial_states"))
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    from msufold.env._pyflex_bootstrap import ensure_pyflex
    from msufold.env.softgym_cloth_env import SoftgymClothEnv, rotate_particles
    from skills.backends import scale_cloth, translate_cloth

    pyflex = ensure_pyflex()

    rng = np.random.default_rng(args.seed)
    env = SoftgymClothEnv(render_dim=args.image_size)

    for cloth_type in args.cloth_types:
        cloth3d = cloth_type not in ("Square", "Rectangular")
        cache_path = os.path.join(args.dataset_root, "softgym_cache", cloth_type + ".pkl")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        configs, states = cache["configs"], cache["states"]

        for case_idx in range(args.num_cases):
            index = int(rng.integers(len(configs)))
            angle = sample_angle(cloth_type, rng)
            size_scale = float(rng.uniform(*args.size_range))
            offset = rng.uniform(-args.offset_range, args.offset_range, size=2)

            env.reset(
                config=configs[index],
                state=states[index],
                cloth3d=cloth3d,
                pick_speed=0.005,
                move_speed=0.005,
                place_speed=0.005,
                lift_height=0.1,
            )
            rotate_particles([0, angle, 0])
            scale_cloth(size_scale)
            translate_cloth(offset)
            for _ in range(300):
                pyflex.step()
                if np.all(np.abs(pyflex.get_velocities()) < 0.2):
                    break

            rgb, depth = env.render_image()
            mask = get_mask_from_depth(depth)

            case_dir = Path(args.output_dir) / cloth_type / f"case_{case_idx:02d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(case_dir / "rgb.png")
            Image.fromarray((mask > 0).astype(np.uint8) * 255).save(case_dir / "mask.png")
            np.save(case_dir / "depth.npy", depth.astype(np.float32))
            with open(case_dir / "scene.pkl", "wb") as f:
                pickle.dump({"config": configs[index], "state": states[index],
                             "cloth3d": cloth3d}, f)
            with open(case_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "cloth_type": cloth_type,
                        "cloth3d": cloth3d,
                        "cache_index": index,
                        "angle_deg": angle,
                        "size_scale": size_scale,
                        "offset": np.round(offset, 5).tolist(),
                        "image_size": args.image_size,
                    },
                    f,
                    indent=2,
                )
            print(f"saved: {case_dir} (angle={angle:.2f}, size={size_scale:.3f}, "
                  f"offset={np.round(offset, 4).tolist()})")

    env.close()
    print("all initial states prepared under:", args.output_dir)


if __name__ == "__main__":
    main()
