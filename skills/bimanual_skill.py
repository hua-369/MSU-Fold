"""双臂折叠 skill：单/双臂混合步的闭环执行。"""

from typing import List

import numpy as np
import torch

from msufold.data.utils import get_mask_from_depth
from msufold.env.softgym_dual_demonstrators import DualDemonstrator
from msufold.models.utils import nearest_to_mask

from .base_skill import BaseSkill, Observation


def sample_topk_peaks(heatmap, k, min_distance=12, mask=None):
    """从热力图取前 k 个峰（贪心 NMS），返回 [[u, v], ...]。"""
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


def match_places_to_picks(pick_peaks, place_peaks):
    """把放置点按就近原则分配给抓取点（双臂不应交叉）。"""
    places = [np.asarray(p, dtype=float) for p in place_peaks]
    matched = []
    for p in pick_peaks:
        dists = [np.linalg.norm(np.asarray(p, dtype=float) - q) for q in places]
        matched.append(places.pop(int(np.argmin(dists))))
    return matched


class BimanualSkill(BaseSkill):
    """双臂（含单臂混合步）折叠 skill 基类，子类只需指定 dual_task。

    分步计划来自 DualDemonstrator：每一步的 pick/place 是关键点索引列表，
    长度 2 表示双臂同时抓取，长度 1 表示单臂，因此一个 skill 内可混用。
    """

    arms = "bimanual"
    dual_task = None
    cloth_type = "Square"
    cloth3d = False
    demonstrator_cls = None

    def __init__(self):
        assert self.dual_task is not None, "双臂 skill 必须指定 dual_task"
        self.demonstrator = DualDemonstrator[self.dual_task]()

    @property
    def num_steps(self) -> int:
        return len(self.get_plan()["instructions"])

    def get_plan(self, random_angle: float = 0.0):
        """取双臂任务分步计划（si 版本）。

        DualDemonstrator 返回 (seen, unseen_instruction, unseen_task)，
        闭环执行只用 seen 那套指令。
        """
        plan, _unseen_instruction, _unseen_task = self.demonstrator.get_eval_instruction()
        return plan

    def _n_arms(self, plan, step_idx) -> int:
        """该步是双臂还是单臂：看这一步抓取了几个关键点。"""
        return len(plan["pick"][step_idx])

    def _predict_and_execute_step(
        self,
        backend,
        model,
        processor,
        device,
        observation: Observation,
        context: List[dict],
        instruction: str,
        n_arms: int = 2,
    ) -> Observation:
        """单步闭环：观测 + 指令 -> 像素级关键点 -> 后端执行 -> 重新采集观测。

        双臂步从热力图取前 2 个峰，单臂步取峰值；输出始终是**像素点**，
        由后端换算成机械臂坐标。
        """
        sample = processor(
            depth=observation.depth,
            instruction=instruction,
            rgb=observation.rgb,
            mask=observation.mask,
            context=context,
            **backend.processor_kwargs(),
        )
        for k, val in sample.items():
            if isinstance(val, torch.Tensor):
                sample[k] = val.unsqueeze(0).to(device)

        action, raw_output = model.get_action(sample, return_raw_output=True)

        if n_arms == 1:
            assert len(action.pick) >= 1 and len(action.place) >= 1
            picks_pixel = [action.pick[0]]
            places_pixel = [action.place[0]]
        else:
            mask_np = sample["mask"].squeeze().detach().cpu().numpy()
            pick_peaks = sample_topk_peaks(raw_output["pick_heatmap"], n_arms, mask=mask_np)
            place_peaks = sample_topk_peaks(raw_output["place_heatmap"], n_arms, mask=mask_np)
            # 就近配对，避免左右臂的抓取/放置交叉把布料拧住
            picks_pixel = list(pick_peaks)
            places_pixel = list(match_places_to_picks(pick_peaks, place_peaks))

        backend.execute(picks_pixel, places_pixel, observation)

        context.append(
            {
                "rgb": observation.rgb.copy(),
                "depth": observation.depth.copy(),
                "mask": observation.mask.copy(),
            }
        )

        return backend.observe()


class BimanualTshirtFoldSkill(BimanualSkill):
    """T恤：① 双臂同时折两只袖子 ② 双臂把下摆折到领口（2 步）。"""

    name = "BimanualTshirtFold"
    dual_task = "MixedTshirtFold"
    cloth_type = "Tshirt"
    cloth3d = True


class BimanualTrousersFoldSkill(BimanualSkill):
    """裤子：① 双臂左右对折成一条长条 ② 单臂把长条对折（2 步）。"""

    name = "BimanualTrousersFold"
    dual_task = "MixedTrousersFold"
    cloth_type = "Trousers"
    cloth3d = True


class BimanualCornerFoldSkill(BimanualSkill):
    """方巾：双臂分两步把 4 个角依次折向中心，两步角组合互补（2 步）。"""

    name = "BimanualCornerFold"
    dual_task = "MixedSquareCornerFold"
    cloth_type = "Square"
    cloth3d = False


class BimanualHalfFoldSkill(BimanualSkill):
    """长方巾：① 双臂沿一条边对折 ② 单臂沿垂直方向再对折（2 步）。"""

    name = "BimanualHalfFold"
    dual_task = "MixedSquareHalfFold"
    cloth_type = "Square"
    cloth3d = False


BIMANUAL_SKILL_CLASSES = {
    cls.dual_task: cls
    for cls in (
        BimanualTshirtFoldSkill,
        BimanualTrousersFoldSkill,
        BimanualCornerFoldSkill,
        BimanualHalfFoldSkill,
    )
}

BIMANUAL_SKILL_NAMES = {
    cls.__name__: cls for cls in BIMANUAL_SKILL_CLASSES.values()
}
