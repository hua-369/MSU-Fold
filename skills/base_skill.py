"""Skill 基类：一个 skill = 一个完整的多次折叠流程。

skill 的输入只有"初始观测图像 + 一句总指令"，输出是每步的 pick/place 像素点，
像素点 -> 机械臂坐标由执行后端（仿真反投影 / 真机手眼标定）完成，
因此同一份 skill 既能跑仿真也能上真机。

执行流程（闭环）：
  初始观测 -> step1 指令 -> 模型推理 -> 后端执行 -> 重新采集观测
  -> step2 指令 -> ... -> 直到该 skill 的所有步骤完成。
每步的指令来自 msufold/env/softgym_demonstrators.py（单臂）或
softgym_dual_demonstrators.py（双臂）里对应示范器的语言模板。
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from msufold.data.utils import get_mask_from_depth


@dataclass
class Observation:
    """环境中的一帧观测。"""

    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray


class BaseSkill:
    """所有折叠 skill 的基类。

    子类需要指定:
      name:             skill 名称（LLM 路由时使用）
      cloth_type:       softgym 缓存场景类型（Square / Rectangular / Tshirt / Trousers）
      cloth3d:          是否为 Cloth3D 网格场景
      demonstrator_cls: msufold/env/softgym_demonstrators.py 中对应的 Demonstrator
    """

    name: str = "BaseSkill"
    cloth_type: str = "Square"
    cloth3d: bool = False
    demonstrator_cls = None
    #: 该技能用几条手臂执行（子类 BimanualSkill 为 "bimanual"）
    arms: str = "unimanual"

    def __init__(self):
        assert self.name and self.cloth_type and self.demonstrator_cls is not None
        self.demonstrator = self.demonstrator_cls()

    @property
    def num_steps(self) -> int:
        return len(self.demonstrator.primitives)

    def get_plan(self, random_angle: float = 0.0):
        """从 Demonstrator 中取整个任务的分步指令序列。

        只有 StraightFold.get_action_instruction 需要 random_angle 参数，
        其余 Demonstrator 不需要，这里用 TypeError 统一处理。
        """
        try:
            plan = self.demonstrator.get_action_instruction(random_angle)
        except TypeError:
            plan = self.demonstrator.get_action_instruction()
        (
            pick_idxs,
            place_idxs,
            gammas,
            instructions,
            primitives,
            unseen_flags,
        ) = plan
        assert len(instructions) == self.num_steps
        return {
            "pick": pick_idxs,
            "place": place_idxs,
            "gammas": gammas,
            "instructions": instructions,
            "primitives": primitives,
            "flags": unseen_flags,
        }

    def prepare_env(self, backend, random_angle: float = 0.0, **scene_kwargs):
        """复位场景：仿真加载初始状态，真机回到拍照位。

        机械臂速度/抬升高度等只有仿真用得到，真机后端会忽略这些参数。
        """
        backend.reset(
            cloth3d=self.cloth3d,
            speeds={
                "pick_speed": self.demonstrator.pick_speed,
                "move_speed": self.demonstrator.move_speed,
                "place_speed": self.demonstrator.place_speed,
                "lift_height": self.demonstrator.lift_height,
            },
            random_angle=random_angle,
            **scene_kwargs,
        )

    def _n_arms(self, plan, step_idx) -> int:
        """第 step_idx 步用几条手臂执行（单臂技能恒为 1，双臂技能按步骤的抓取点数决定）。"""
        return 1

    def _predict_and_execute_step(
        self,
        backend,
        model,
        processor,
        device,
        observation: Observation,
        context: List[dict],
        instruction: str,
        n_arms: int = 1,
    ) -> Observation:
        """单步闭环：观测 + 本步指令 -> 像素级 pick/place -> 后端执行 -> 新观测。

        这里只输出**像素点**，换算到机械臂坐标由后端（仿真反投影 / 真机手眼标定）负责。
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

        action, _raw_output = model.get_action(sample, return_raw_output=True)
        assert len(action.pick) >= 1 and len(action.place) >= 1
        backend.execute([action.pick[0]], [action.place[0]], observation)

        # 把执行前的观测存入历史上下文（与训练时 sequential context 一致）
        context.append(
            {
                "rgb": observation.rgb.copy(),
                "depth": observation.depth.copy(),
                "mask": observation.mask.copy(),
            }
        )

        # 执行后重新采集观测（真机=重新拍照），作为下一步的输入
        return backend.observe()

    def execute(
        self,
        backend,
        model,
        processor,
        device,
        observation: Optional[Observation] = None,
        context: Optional[List[dict]] = None,
        random_angle: float = 0.0,
        log_dir: Optional[str] = None,
        trial_name: str = "trial",
        verbose: bool = True,
    ):
        """完整执行该 skill 的所有折叠步骤（闭环：每步执行后的状态作为下步输入）。

        输入只有初始观测图像 + 一句总指令；若 observation 为 None 则由后端采集。
        返回 (最终观测, 上下文列表, 实际使用的分步指令)。
        """
        if context is None:
            context = []

        if observation is None:
            observation = backend.observe()

        plan = self.get_plan(random_angle)
        instructions: List[str] = plan["instructions"]

        if log_dir is not None:
            import os

            os.makedirs(log_dir, exist_ok=True)
            Image.fromarray(observation.rgb).save(
                os.path.join(log_dir, f"{trial_name}_step0_initial.png")
            )

        for step_idx, instruction in enumerate(instructions):
            if verbose:
                print(
                    f"[{self.name}] step {step_idx + 1}/{len(instructions)}: {instruction}"
                )
            observation = self._predict_and_execute_step(
                backend=backend,
                model=model,
                processor=processor,
                device=device,
                observation=observation,
                context=context,
                instruction=instruction,
                n_arms=self._n_arms(plan, step_idx),
            )
            if log_dir is not None:
                Image.fromarray(observation.rgb).save(
                    os.path.join(log_dir, f"{trial_name}_step{step_idx + 1}_final.png")
                )

        if verbose:
            print(f"[{self.name}] all {len(instructions)} folding steps finished")
        return observation, context, instructions
