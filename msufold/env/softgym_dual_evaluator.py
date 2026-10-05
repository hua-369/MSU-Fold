"""双臂混合任务的 softgym 评估器。

评估内容与训练数据集 datasets/dual_data_sequential/All_150.pkl 一致：
4 个 Mixed 任务、每条 2 步（dual/dual 或 dual/single）、指令风格相同。
按用户要求**不评 ut**（没有 held-out 的双臂任务），只评 si（seen instruction）
与 usi（unseen instruction）。

指标计算与 SoftgymSingleEvaluator 完全一致（每步与 oracle 轨迹比粒子误差、
mask IoU、iou_success 阈值），输出格式相同：
    MixedSquareHalfFold si/usi, average_success, error/iou/iou_success_*

动作执行区别：
  * 双臂步骤（pick/place 各 2 个关键点）用 env.pick_and_place_dual；
  * 单臂步骤（1 个关键点）用 env.pick_and_place_single；
  * 模型预测：双臂步骤从 pick/place 热力图取前 2 个峰（贪心 NMS），
    单臂步骤仍取 argmax。
"""

import os
import pickle
from typing import Optional

import cv2
import imageio
import matplotlib

matplotlib.use("Agg")  # 评估阶段无显示设备，用无头后端保存热力图
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
from tqdm import trange

from msufold.data.single_dataset import get_mask_from_depth
from msufold.env._pyflex_bootstrap import ensure_pyflex
from msufold.metrics.utils import iou
from msufold.models.utils import nearest_to_mask

from .softgym_cloth_env import SoftgymClothEnv, rotate_particles
from .softgym_dual_demonstrators import DualDemonstrator, dual_task_to_cloth_type
from .softgym_evaluator import SoftgymEvaluator

pyflex = ensure_pyflex()


def match_places_to_picks(pick_peaks, place_peaks):
    """把放置点按就近原则与抓取点配对（双臂不应交叉）。

    抓取/放置两组峰是独立取的，峰之间的左右配对关系是随机的；
    这里用贪心最近邻把每个抓取点分配给离它最近的放置点，
    避免左臂抓的布被右臂放到对侧导致布料拧住。
    """
    places = [np.asarray(p, dtype=float) for p in place_peaks]
    matched = []
    for p in pick_peaks:
        dists = [np.linalg.norm(np.asarray(p, dtype=float) - q) for q in places]
        j = int(np.argmin(dists))
        matched.append(places.pop(j))
    return matched


def sample_topk_peaks(heatmap, k, min_distance=12, mask=None):
    """从热力图取前 k 个峰（贪心 NMS）。

    返回 [[u, v], ...]，与 models.sample_from_heatmap 的输出约定完全一致
    （unravel 得到 (row, col) 后交换为 [col, row]，并把点吸附到布料 mask 上）。
    """
    if hasattr(heatmap, "detach"):
        heatmap = heatmap.detach().cpu().numpy()
    hm = heatmap[0] if heatmap.ndim == 3 else heatmap
    work = hm.astype(float).copy()
    peaks = []
    for _ in range(k):
        v, u = np.unravel_index(np.argmax(work), work.shape)
        if mask is not None:
            r, c = nearest_to_mask(int(v), int(u), mask)
        else:
            r, c = int(v), int(u)
        peaks.append([int(c), int(r)])
        y0 = max(0, v - min_distance)
        y1 = min(work.shape[0], v + min_distance + 1)
        x0 = max(0, u - min_distance)
        x1 = min(work.shape[1], u + min_distance + 1)
        work[y0:y1, x0:x1] = -1.0
    return np.array(peaks)


class SoftgymDualEvaluator(SoftgymEvaluator):
    def __init__(self, cfg, model, processor, env: Optional[SoftgymClothEnv] = None):
        if env is not None:
            # 复用单臂评估器已创建的环境（一个进程只能 init 一次 pyflex）
            self.model = model
            self.processor = processor
            self.cache = cfg.softgym_cache
            self.visualize_predictions = cfg.visualize_predictions
            self.env = env
            self.K = self.env.intrinsic_from_fov(
                height=cfg.model.image_size, width=cfg.model.image_size
            )
            self.error_threshold = self.env.particle_radius * 2
            self.iou_thresholds = [50, 80, 90]
            self.success = {}
            self.additional_metrics = {}
        else:
            super().__init__(cfg, model, processor)

    def reset(
        self,
        config,
        state,
        task=None,
        random_angle=None,
        max_wait_step=300,
        stable_vel_threshold=0.2,
    ):
        self.demonstrator = DualDemonstrator[task]()
        self.env.reset(
            config=config,
            state=state,
            cloth3d=self.cloth3d,
            pick_speed=self.demonstrator.pick_speed,
            move_speed=self.demonstrator.move_speed,
            place_speed=self.demonstrator.place_speed,
            lift_height=self.demonstrator.lift_height,
        )
        self.task = task if task is not None else ""
        if random_angle:
            rotate_particles([0, random_angle, 0])
            for _ in range(max_wait_step):
                pyflex.step()
                curr_vel = pyflex.get_velocities()
                if np.all(np.abs(curr_vel) < stable_vel_threshold):
                    break

    def evaluate(
        self,
        cloth_type: Optional[str] = None,
        cloth3d: Optional[bool] = None,
        num_evals: Optional[int] = None,
        task: Optional[str] = None,
        **kwargs,
    ):
        if cloth_type is None:
            assert task is not None
            cloth_type = dual_task_to_cloth_type[task]
            cloth3d = cloth_type not in ("Square", "Rectangular")
        else:
            assert cloth3d is not None

        super().evaluate(cloth_type=cloth_type, cloth3d=cloth3d)

        if task not in self.success:
            self.success[task] = {}
            self.additional_metrics = {
                k: {task: {}}
                for k in ["error", "iou"]
                + [f"iou_success_{thresh}" for thresh in self.iou_thresholds]
            }

        assert num_evals is not None
        video_dir = None
        self.heatmap_dir = None
        if self.visualize_predictions:
            video_dir = os.path.join("eval", "softgym", task, "videos")
            os.makedirs(video_dir, exist_ok=True)
            # 热力图可视化：每步一张（RGB + pick 热力图 + place 热力图）
            self.heatmap_dir = os.path.join("eval", "softgym", task, "heatmaps")
            os.makedirs(self.heatmap_dir, exist_ok=True)

        for i in trange(num_evals, desc=f"Evaluating {task}"):
            rand_idx = np.random.randint(len(self.cached_configs))
            config = self.cached_configs[rand_idx]
            state = self.cached_states[rand_idx]

            random_angle = (
                np.random.uniform(-40, 40) if cloth3d else np.random.uniform(0, 40)
            )

            self.reset(config=config, state=state, task=task, random_angle=random_angle)

            if self.cloth3d:
                keypoints_index = self.cached_keypoints[rand_idx]
            else:
                keypoints_index = self.env.get_square_keypoints_idx()

            # 双臂任务不评 ut：示范器返回 (seen, unseen_instruction, unseen_task)，忽略第三个
            eval_seen, eval_unseen_instr, _ = self.demonstrator.get_eval_instruction()
            eval_datas = [eval_seen, eval_unseen_instr]
            eval_name_list = ["si", "usi"]

            for eval_index in range(2):
                eval_data = eval_datas[eval_index]
                eval_name = eval_name_list[eval_index]

                if eval_name not in self.success[task]:
                    self.success[task][eval_name] = []
                    for k in self.additional_metrics.keys():
                        self.additional_metrics[k][task][eval_name] = []

                self.reset(
                    config=config, state=state, task=task, random_angle=random_angle
                )
                oracle_results, oracle_masks = self.execute_oracle(
                    pick_steps=eval_data["pick"],
                    place_steps=eval_data["place"],
                    gammas=eval_data["gammas"],
                    keypoints_index=keypoints_index,
                    video_path=(
                        os.path.join(video_dir, f"{eval_name}_trial{i}_oracle.mp4")
                        if video_dir
                        else None
                    ),
                )
                self.reset(
                    config=config, state=state, task=task, random_angle=random_angle
                )
                self.execute_model(
                    pick_steps=eval_data["pick"],
                    place_steps=eval_data["place"],
                    gammas=eval_data["gammas"],
                    instructions=eval_data["instructions"],
                    unseen_flags=eval_data["flags"],
                    keypoints_index=keypoints_index,
                    eval_name=eval_name,
                    trial_index=i,
                    oracle_results=oracle_results,
                    oracle_masks=oracle_masks,
                    video_path=(
                        os.path.join(video_dir, f"{eval_name}_trial{i}_model.mp4")
                        if video_dir
                        else None
                    ),
                )

    def _execute_step(self, picks, places):
        if len(picks) == 2:
            self.env.pick_and_place_dual(picks[0], places[0], picks[1], places[1])
        else:
            self.env.pick_and_place_single(picks[0], places[0])

    @staticmethod
    def _draw_label(img, text, picks=None, places=None):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 20), (0, 0, 0), -1)
        cv2.putText(
            img, text[:60], (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        if picks is not None:
            for k, p in enumerate(picks):
                # 点约定 [u, v] -> cv2 坐标 (x=u, y=v)
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)
        if places is not None and picks is not None:
            for k, (p, q) in enumerate(zip(picks, places)):
                color = (0, 0, 255) if k == 0 else (0, 150, 255)
                cv2.arrowedLine(
                    img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 2
                )
        return img

    def _save_heatmaps(self, path, rgb, raw_output, picks, places, instruction, n_arms):
        """保存一步的三联图：RGB(带抓取/放置点) + pick 热力图 + place 热力图。"""

        def _to_numpy(heatmap):
            array = heatmap.detach().cpu().numpy()
            return array[0] if array.ndim == 3 else array

        pick_hm = _to_numpy(raw_output["pick_heatmap"])
        place_hm = _to_numpy(raw_output["place_heatmap"])
        # 双臂步的放置点是 list（就近配对后），统一成 (N,2)
        picks = None if picks is None else np.asarray(picks).reshape(-1, 2)
        places = None if places is None else np.asarray(places).reshape(-1, 2)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        panels = [
            (axes[0], None, f"RGB | {n_arms} arm(s)"),
            (axes[1], pick_hm, "Pick heatmap"),
            (axes[2], place_hm, "Place heatmap"),
        ]
        for ax, hm, title in panels:
            ax.imshow(rgb)
            if hm is not None:
                ax.imshow(hm, cmap="hot", alpha=0.6, interpolation="bilinear")
            if picks is not None:
                ax.scatter(picks[:, 0], picks[:, 1], c="cyan", s=45, marker="x")
            if places is not None:
                ax.scatter(places[:, 0], places[:, 1], c="lime", s=45, marker="+")
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        fig.suptitle(instruction[:90], fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _save_video(self, path, marks, frames_offset=0):
        """marks: [{"start","end","text","picks","places"}]，帧区间为绝对下标。"""
        frames = getattr(self.env, "frames", [])[frames_offset:]
        if not frames:
            return
        writer = imageio.get_writer(
            path, fps=30, codec="libx264", quality=8, macro_block_size=1,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        for idx, frame in enumerate(frames):
            f_idx = frames_offset + idx
            img = frame
            for mark in marks:
                if mark["start"] <= f_idx < mark["end"]:
                    img = self._draw_label(
                        frame, mark["text"], mark.get("picks"), mark.get("places")
                    )
                    break
            writer.append_data(np.ascontiguousarray(img))
        writer.close()

    def execute_oracle(
        self, pick_steps, place_steps, gammas, keypoints_index, video_path=None
    ):
        oracle_results = []
        oracle_masks = []
        frames_start = len(getattr(self.env, "frames", []))
        marks = []
        for step_i, (pick_list, place_list, gamma) in enumerate(
            zip(pick_steps, place_steps, gammas)
        ):
            step_start = len(getattr(self.env, "frames", []))
            keypoints_pos = self.env.get_keypoints(keypoints_index)
            picks, places = [], []
            for pick_idx, place_idx in zip(pick_list, place_list):
                pick_pos = keypoints_pos[pick_idx]
                place_pos = keypoints_pos[place_idx]
                place_pos = pick_pos + gamma * (place_pos - pick_pos)
                picks.append(pick_pos.copy())
                places.append(place_pos.copy())
            self._execute_step(picks, places)
            marks.append(
                {
                    "start": step_start,
                    "end": len(getattr(self.env, "frames", [])),
                    "text": f"oracle step {step_i + 1}",
                }
            )
            rgb, depth = self.env.render_image()
            mask = get_mask_from_depth(depth)
            oracle_results.append(pyflex.get_positions().reshape(-1, 4)[:, :3])
            oracle_masks.append(mask)

        if video_path is not None:
            self._save_video(video_path, marks, frames_offset=frames_start)
        return oracle_results, oracle_masks

    def execute_model(
        self,
        pick_steps,
        place_steps,
        gammas,
        instructions,
        unseen_flags,
        keypoints_index,
        eval_name,
        trial_index,
        oracle_results,
        oracle_masks,
        video_path=None,
    ):
        rgb, depth = self.env.render_image()
        mask = get_mask_from_depth(depth)
        context = []
        frames_start = len(getattr(self.env, "frames", []))
        marks = []

        assert (
            len(pick_steps)
            == len(place_steps)
            == len(gammas)
            == len(instructions)
            == len(unseen_flags)
            == len(oracle_results)
            == len(oracle_masks)
        )

        for action_index, (
            pick_list,
            place_list,
            gamma,
            instruction,
            unseen_flag,
        ) in enumerate(zip(pick_steps, place_steps, gammas, instructions, unseen_flags)):
            sample = self.processor(
                depth=depth,
                instruction=instruction,
                rgb=rgb,
                mask=mask,
                context=context,
                matrix_world_to_camera=self.env.camera_matrix,
                K=self.K,
            )
            for k, val in sample.items():
                if isinstance(val, torch.Tensor):
                    sample[k] = val.unsqueeze(0).to(self.model.device)

            n_arms = len(pick_list)
            step_start = len(getattr(self.env, "frames", []))
            draw_picks, draw_places = None, None
            if unseen_flag == 1:  # oracle 执行该步
                keypoints_pos = self.env.get_keypoints(keypoints_index)
                picks, places = [], []
                for pick_idx, place_idx in zip(pick_list, place_list):
                    pick_pos = keypoints_pos[pick_idx]
                    place_pos = keypoints_pos[place_idx]
                    place_pos = pick_pos + gamma * (place_pos - pick_pos)
                    picks.append(pick_pos.copy())
                    places.append(place_pos.copy())
            else:  # 模型执行该步
                action, raw_output = self.model.get_action(
                    sample, return_raw_output=True
                )
                if n_arms == 1:
                    draw_picks, draw_places = np.array([action.pick[0]]), np.array(
                        [action.place[0]]
                    )
                    picks = [self.env.get_world_coord_from_pixel(action.pick[0], depth)]
                    places = [
                        self.env.get_world_coord_from_pixel(action.place[0], depth)
                    ]
                else:
                    mask_np = sample["mask"].squeeze().detach().cpu().numpy()
                    pick_peaks = sample_topk_peaks(
                        raw_output["pick_heatmap"], n_arms, mask=mask_np
                    )
                    place_peaks = sample_topk_peaks(
                        raw_output["place_heatmap"], n_arms, mask=mask_np
                    )
                    # 就近配对, 防止左右臂的抓取/放置交叉
                    place_peaks = match_places_to_picks(pick_peaks, place_peaks)
                    draw_picks, draw_places = pick_peaks, place_peaks
                    picks = [
                        self.env.get_world_coord_from_pixel(np.array(pk), depth)
                        for pk in pick_peaks
                    ]
                    places = [
                        self.env.get_world_coord_from_pixel(np.array(pl), depth)
                        for pl in place_peaks
                    ]

            # 训练/评估期间的热力图可视化（visualize_predictions=true 时开启）
            if self.heatmap_dir is not None and unseen_flag == 0:
                self._save_heatmaps(
                    path=os.path.join(
                        self.heatmap_dir,
                        f"{eval_name}_trial{trial_index}_step{action_index + 1}.png",
                    ),
                    rgb=rgb,
                    raw_output=raw_output,
                    picks=draw_picks,
                    places=draw_places,
                    instruction=instruction,
                    n_arms=n_arms,
                )

            self._execute_step(picks, places)

            context.append(
                {"rgb": rgb.copy(), "depth": depth.copy(), "mask": mask.copy()}
            )
            rgb, depth = self.env.render_image()
            mask = get_mask_from_depth(depth)
            particle_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]

            error = np.linalg.norm(
                oracle_results[action_index] - particle_pos, axis=1
            ).mean()
            success = error < self.error_threshold
            iou_value = iou(mask, oracle_masks[action_index])

            self.success[self.task][eval_name].append(success)
            self.additional_metrics["error"][self.task][eval_name].append(error)
            self.additional_metrics["iou"][self.task][eval_name].append(iou_value)
            for thresh in self.iou_thresholds:
                self.additional_metrics[f"iou_success_{thresh}"][self.task][
                    eval_name
                ].append((iou_value > thresh) * 100)

            marks.append(
                {
                    "start": step_start,
                    "end": len(getattr(self.env, "frames", [])),
                    "text": f"model step {action_index + 1} | {instruction}",
                    "picks": draw_picks,
                    "places": draw_places,
                }
            )

        if video_path is not None:
            self._save_video(video_path, marks, frames_offset=frames_start)
