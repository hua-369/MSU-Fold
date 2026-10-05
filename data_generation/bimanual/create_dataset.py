"""把 bimanual 生成的演示(raw_data_mixed)打包成与单臂数据集对齐的 pkl。

输出结构与 scripts/create_unimanual_sequential_dataset.py 完全一致:
    {"episodes": [ {depth, pick, place, instruction, success, primitive, rgbs}, ... ]}
区别只在于: 双臂步骤的 pick/place 是 (2,2)（左臂/右臂各一个像素点），
单臂步骤仍是 (1,2)，primitive 为 "dual" / "single"。

用法:
    python bimanual/create_dataset.py --tasks All --n_demos 150 --use_rgb \
        --root raw_data_mixed \
        --save_path_root ../datasets/dual_data_sequential
"""

import argparse
import os
import pickle
import random

import imageio
import numpy as np

Done = np.array([0, 0])


def create_dataset(root, tasks, save_path, use_rgb, n_demos):
    episodes = []
    total_num = 0
    seen_num = 0

    if "All" in tasks:
        tasks = sorted(os.listdir(root))
        # 只保留真正的任务目录（里面有数字编号的轨迹目录），跳过 preview 等辅助目录
        tasks = [
            t
            for t in tasks
            if os.path.isdir(os.path.join(root, t))
            and any(d.isdigit() for d in os.listdir(os.path.join(root, t)))
        ]
        print("Load All Tasks: ", tasks)

    trajs = [
        os.path.join(root, task, traj)
        for task in tasks
        for traj in sorted(os.listdir(os.path.join(root, task)))
    ]
    random.shuffle(trajs)

    each_task_num = {task: 0 for task in tasks}
    for traj in trajs:
        task = traj.split(os.path.sep)[-2]
        # 跳过不完整的轨迹（生成中途被中断）
        if not os.path.isfile(os.path.join(traj, "info.pkl")):
            print("Skip incomplete trajectory:", traj)
            continue
        if each_task_num[task] >= n_demos:
            continue

        depths, picks, places, instructions, success, primitives, rgbs = (
            [], [], [], [], [], [], []
        )
        with open(os.path.join(traj, "info.pkl"), "rb") as f:
            data = pickle.load(f)
        pick_pixels = data["pick"]
        place_pixels = data["place"]
        langs = data["instruction"]
        prims = data.get("primitive", ["dual"] * len(langs))
        unseens = data.get("unseen_flags", [0] * len(langs))

        num_actions = len(pick_pixels)
        total_num += num_actions
        each_task_num[task] += 1

        depth_path = os.path.join(traj, "depth")
        rgb_path = os.path.join(traj, "rgb")

        for i in range(num_actions):
            if unseens[i]:
                continue
            seen_num += 1
            picks.append(np.asarray(pick_pixels[i], dtype=float).reshape(-1, 2))
            places.append(np.asarray(place_pixels[i], dtype=float).reshape(-1, 2))
            instructions.append(langs[i])
            primitives.append(prims[i])
            success.append(0)
            depths.append(imageio.imread(os.path.join(depth_path, str(i) + ".png")))
            if use_rgb:
                rgbs.append(imageio.imread(os.path.join(rgb_path, str(i) + ".png")))

        if depths:
            assert len(depths) == len(picks) == len(places) == len(instructions) == len(success)
            episodes.append({
                "depth": depths,
                "pick": picks,
                "place": places,
                "instruction": instructions,
                "success": success,
                "primitive": primitives,
            })
            if use_rgb:
                episodes[-1]["rgbs"] = rgbs

    print("build {} seen tasks from {} tasks".format(seen_num, total_num))
    print(each_task_num)

    with open(save_path, "wb+") as f:
        pickle.dump({"episodes": episodes}, f)
    print("Done! saved to", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dual/mixed dataset")
    parser.add_argument("--tasks", type=str, default="All", help="choose single task / all tasks")
    parser.add_argument("--use_rgb", action="store_true", help="include rgb images")
    parser.add_argument("--root", type=str, default="raw_data_mixed")
    parser.add_argument("--save_path_root", type=str, default="dual_data_sequential")
    parser.add_argument("--n_demos", type=int, default=100, help="num of demos per task")
    args = parser.parse_args()

    os.makedirs(args.save_path_root, exist_ok=True)
    if args.tasks == "All":
        save_path = os.path.join(args.save_path_root, f"All_{args.n_demos}.pkl")
        tasks = ["All"]
    else:
        save_path = os.path.join(args.save_path_root, args.tasks + ".pkl")
        tasks = [args.tasks]

    create_dataset(args.root, tasks, save_path, args.use_rgb, args.n_demos)
