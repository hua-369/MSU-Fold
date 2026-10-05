"""Generate folding demonstrations with **mixed single/dual-arm multi-step** sequences.

每条轨迹是一条有序的多步折叠序列 (bimanual/demonstrator.py 定义),
序列中每一步可能是:
    type="dual"   -> ClothEnv.pick_and_place_dual   (两臂同时抓、同时放)
    type="single" -> ClothEnv.pick_and_place_single (单臂抓取放置)

保存结构 (与单臂 raw_data 完全对齐):
    <save_root>/<task>/<idx>/rgb/0.png ...      # 每步动作前后的观测 (224x224)
    <save_root>/<task>/<idx>/depth/0.png ...    # uint8 深度 (x255)
    <save_root>/<task>/<idx>/viz/0.png ...      # pick->place 箭头可视化 (双臂步骤画两条)
    <save_root>/<task>/<idx>/info.pkl           # pick/place/指令/type/gamma/unseen_flags
    <save_root>/<task>/<idx>/video.mp4          # 演示视频 (--record_video, 每步显示对应指令)

用法 (data_generation 根目录, MSU-Fold 环境):
    python bimanual/generate_demo_fold.py --task MixedSquareHalfFold \
        --num_demonstrations 1 --record_video --seed 0
"""

import argparse
import os
import pickle
import random

import cv2
import imageio
import numpy as np
import pyflex
from tqdm import tqdm

from bimanual.demonstrator import MixedDemonstrator
from softgym.envs.cloth_env import ClothEnv
from softgym.envs.flex_utils import move_to_pos, rotate_particles
from utils.visual import action_viz, get_pixel_coord_from_world


def randomize_state(cloth3d):
    """与单臂 generate_demo_fold.py 保持一致的初始位姿随机化。"""
    max_wait_step = 300
    stable_vel_threshold = 0.2
    if cloth3d:
        random_angle = np.random.uniform(-40, 40)
    else:
        random_angle = np.random.uniform(0, 40)
    random_pos_move = np.random.uniform(low=-0.02, high=0.02, size=(3,))
    rotate_particles([0, random_angle, 0])
    random_pos_move[1] = 0
    move_to_pos(random_pos_move)
    for _ in range(max_wait_step):
        pyflex.step()
        curr_vel = pyflex.get_velocities()
        if np.all(np.abs(curr_vel) < stable_vel_threshold):
            break
    return random_angle


def draw_instruction(img, text):
    """在画面上方叠加当前步骤的指令 (只支持 ASCII)。"""
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.42, 1
    max_chars = 46
    lines = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    y = 18
    cv2.rectangle(img, (0, 0), (img.shape[1], 20 * len(lines) + 8), (0, 0, 0), -1)
    for line in lines:
        cv2.putText(img, line, (6, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += 20
    return img


def save_video(frames, frame_actions, path, fps=30):
    """frames: 观测帧 + 运动帧; frame_actions: [(start, end, instruction), ...]"""
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", quality=8, macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    for f_idx, frame in enumerate(frames):
        img = frame
        for start, end, lang in frame_actions:
            if start <= f_idx < end:
                img = draw_instruction(img, lang)
                break
        writer.append_data(np.ascontiguousarray(img))
    writer.close()


def generate_demos(args):
    task_name = args.task
    policy = MixedDemonstrator[task_name](randomize_pose=args.randomize_pose)
    cloth3d = policy.cloth_type not in ("Square", "Rectangular")

    env = ClothEnv(
        gui=args.gui,
        cloth3d=cloth3d,
        dump_visualizations=args.record_video,  # 记录运动帧用于生成视频
        pick_speed=policy.pick_speed,
        move_speed=policy.move_speed,
        place_speed=policy.place_speed,
        lift_height=policy.lift_height,
    )

    cached_path = os.path.join("configs", policy.cloth_type + ".pkl")
    with open(cached_path, "rb") as f:
        config_data = pickle.load(f)
    configs = config_data["configs"]
    states = config_data["states"]
    print("load {} configs from {}".format(len(configs), cached_path))

    task_root = os.path.join(args.save_root, task_name)
    os.makedirs(task_root, exist_ok=True)
    dirs = [d for d in os.listdir(task_root) if d.isdigit()]
    max_index = max([int(d) for d in dirs]) + 1 if dirs else 0

    for i in tqdm(range(args.num_demonstrations)):
        rand_idx = np.random.randint(len(configs))
        config, state = configs[rand_idx], states[rand_idx]
        env.reset(config=config, state=state)

        if cloth3d:
            keypoints_index = config_data["keypoints"][rand_idx]
        else:
            keypoints_index = env.get_square_keypoints_idx()

        if args.randomize_pose:
            randomize_state(cloth3d)

        traj_dir = os.path.join(task_root, str(max_index + i))
        rgb_folder = os.path.join(traj_dir, "rgb")
        depth_folder = os.path.join(traj_dir, "depth")
        viz_folder = os.path.join(traj_dir, "viz")
        for folder in (rgb_folder, depth_folder, viz_folder):
            os.makedirs(folder, exist_ok=True)

        frames = []          # 观测帧 + 运动帧
        obs_frames = []      # 每步动作前后的观测帧 (数据集观测 + 可视化用)
        frame_actions = []   # (start_frame, end_frame, instruction)
        # 注意: dump_visualizations=False 时 ClothEnv 没有 frames 属性
        recorded = len(getattr(env, "frames", []))

        # 初始观测
        action_index = 0
        rgb, depth = env.render_image()
        imageio.imwrite(os.path.join(rgb_folder, str(action_index) + ".png"), rgb)
        imageio.imwrite(
            os.path.join(depth_folder, str(action_index) + ".png"),
            (depth * 255).astype(np.uint8),
        )
        frames.append(rgb)
        obs_frames.append(rgb)

        pick_pixels = []
        place_pixels = []
        instructions = []
        step_types = []
        unseen_flags = []

        actions, flags = policy.get_action_instruction()
        for act, flag in zip(actions, flags):
            keypoints_pos = env.get_keypoints(keypoints_index)

            if act["type"] == "dual":
                pick_idxs, place_idxs = list(act["pick"]), list(act["place"])
            else:
                pick_idxs, place_idxs = [act["pick"]], [act["place"]]

            # 世界坐标 pick/place (place 沿 pick->place 方向按 gamma 插值)
            pick_worlds, place_worlds = [], []
            picks, places = [], []
            for pick_idx, place_idx in zip(pick_idxs, place_idxs):
                pick_pos = keypoints_pos[pick_idx].copy()
                place_pos = keypoints_pos[place_idx].copy()
                place_pos = pick_pos + act["gamma"] * (place_pos - pick_pos)
                pick_worlds.append(pick_pos)
                place_worlds.append(place_pos)
                picks.append(
                    get_pixel_coord_from_world(pick_pos, depth.shape, env.camera_params)
                )
                places.append(
                    get_pixel_coord_from_world(place_pos, depth.shape, env.camera_params)
                )

            start_frame = len(frames)
            if act["type"] == "dual":
                # 双臂: 两个抓手同时抓取、同时搬运、同时放置
                env.pick_and_place_dual(
                    pick_worlds[0].copy(), place_worlds[0].copy(),
                    pick_worlds[1].copy(), place_worlds[1].copy(),
                )
            else:
                # 单臂: 只用 picker1
                env.pick_and_place_single(
                    pick_worlds[0].copy(), place_worlds[0].copy()
                )

            frames.extend(getattr(env, "frames", [])[recorded:])
            recorded = len(getattr(env, "frames", []))
            action_index += 1
            rgb, depth = env.render_image()
            imageio.imwrite(os.path.join(rgb_folder, str(action_index) + ".png"), rgb)
            imageio.imwrite(
                os.path.join(depth_folder, str(action_index) + ".png"),
                (depth * 255).astype(np.uint8),
            )
            frames.append(rgb)
            obs_frames.append(rgb)
            frame_actions.append((start_frame, len(frames), act["instruction"]))

            pick_pixels.append(np.array(picks))
            place_pixels.append(np.array(places))
            instructions.append(act["instruction"])
            step_types.append(act["type"])
            unseen_flags.append(flag)

        with open(os.path.join(traj_dir, "info.pkl"), "wb+") as f:
            data = {
                "pick": pick_pixels,
                "place": place_pixels,
                "instruction": instructions,
                "primitive": step_types,
                "unseen_flags": unseen_flags,
                "cloth_type": policy.cloth_type,
                "task": task_name,
            }
            pickle.dump(data, f)

        # 动作可视化: 在动作前的观测上画出 pick->place 箭头 (双臂步骤画两条)
        num_actions = len(pick_pixels)
        for act in range(num_actions + 1):
            if act < num_actions:
                img = obs_frames[act].copy()
                for k in range(len(pick_pixels[act])):
                    img = action_viz(img, pick_pixels[act][k], place_pixels[act][k])
            else:
                img = obs_frames[act]
            imageio.imwrite(os.path.join(viz_folder, str(act) + ".png"), img)

        if args.record_video:
            save_video(frames, frame_actions, os.path.join(traj_dir, "video.mp4"))

    print("Done! saved under", task_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mixed single/dual-arm demonstrations")
    parser.add_argument("--task", type=str, default="MixedSquareHalfFold",
                        choices=list(MixedDemonstrator.keys()))
    parser.add_argument("--gui", action="store_true", help="run with gui")
    parser.add_argument("--randomize_pose", action="store_true",
                        help="randomize initial cloth pose")
    parser.add_argument("--num_demonstrations", type=int, default=100)
    parser.add_argument("--record_video", action="store_true",
                        help="save an mp4 video for each demonstration")
    parser.add_argument("--save_root", type=str, default="raw_data_mixed")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    generate_demos(args)
