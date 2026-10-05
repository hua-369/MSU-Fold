"""执行后端：把 skill 与"仿真 / 真机"解耦。

skill 只做三件事，且只依赖本文件定义的接口：

1. 从后端取一帧观测（rgb / depth / mask）——真机就是相机拍到的图像；
2. 用模型把「观测 + 本步指令」变成 pick/place 的**像素点**；
3. 把像素点交回后端执行：1 个点 = 单臂，2 个点 = 双臂。

「像素点 -> 机械臂末端世界坐标」的换算由后端负责，skill 不关心：
  * SoftgymBackend：用仿真相机内参与深度图反投影；
  * RealRobotBackend：用相机 RGB-D 与手眼标定结果换算，再下发机械臂。

因此同一份 SKILL.md / 同一套权重既能跑仿真，也能直接上真机。
"""

from typing import List, Sequence

import numpy as np

from .base_skill import Observation


class ExecutionBackend:
    """执行后端抽象基类。"""

    name = "base"

    # ---- 观测 -----------------------------------------------------------
    def observe(self) -> Observation:
        """采集一帧观测（真机=相机取图，仿真=渲染当前状态）。"""
        raise NotImplementedError

    def processor_kwargs(self) -> dict:
        """送进 Processor 的相机参数（内参 K / 世界->相机矩阵），没有就返回空。"""
        return {}

    # ---- 执行 -----------------------------------------------------------
    def to_world(self, pixel: Sequence[float], observation: Observation) -> np.ndarray:
        """像素点 + 深度 -> 机械臂世界坐标（真机走手眼标定）。"""
        raise NotImplementedError

    def move(self, picks_world: List[np.ndarray], places_world: List[np.ndarray]):
        """下发动作：2 个点走双臂，1 个点走单臂。"""
        raise NotImplementedError

    def execute(
        self,
        picks_pixel: Sequence[Sequence[float]],
        places_pixel: Sequence[Sequence[float]],
        observation: Observation,
    ):
        """skill 调用的入口：像素点 -> 世界坐标 -> 机械臂执行。"""
        picks = [self.to_world(np.asarray(p, dtype=float), observation) for p in picks_pixel]
        places = [self.to_world(np.asarray(p, dtype=float), observation) for p in places_pixel]
        self.move(picks, places)

    # ---- 场景复位 / 释放 -------------------------------------------------
    def reset(self, **kwargs):
        """复位场景：仿真加载初始状态，真机回到拍照位（不需要则忽略）。"""
        return None

    def close(self):
        pass


def scale_cloth(scale, pyflex=None):
    """绕布料中心缩放水平面上的尺寸（仿真里的"随机大小"）。"""
    if pyflex is None:
        from msufold.env._pyflex_bootstrap import ensure_pyflex

        pyflex = ensure_pyflex()
    pos = pyflex.get_positions().reshape(-1, 4)
    center = np.mean(pos[:, :3], axis=0)
    pos[:, 0] = center[0] + (pos[:, 0] - center[0]) * scale
    pos[:, 2] = center[2] + (pos[:, 2] - center[2]) * scale
    pyflex.set_positions(pos)


def translate_cloth(offset, pyflex=None):
    """把布料整体平移一小段（仿真里的"随机摆位"）。"""
    if pyflex is None:
        from msufold.env._pyflex_bootstrap import ensure_pyflex

        pyflex = ensure_pyflex()
    pos = pyflex.get_positions().reshape(-1, 4)
    pos[:, 0] += offset[0]
    pos[:, 2] += offset[1] if len(offset) > 1 else offset[0]
    pyflex.set_positions(pos)


class SoftgymBackend(ExecutionBackend):
    """仿真后端：包装 SoftgymClothEnv，仅在 SoftGym 评估时使用。"""

    name = "softgym"

    def __init__(self, env, K):
        self.env = env
        self.K = K

    def observe(self) -> Observation:
        from msufold.data.utils import get_mask_from_depth

        rgb, depth = self.env.render_image()
        return Observation(rgb=rgb, depth=depth, mask=get_mask_from_depth(depth))

    def processor_kwargs(self) -> dict:
        return {"matrix_world_to_camera": self.env.camera_matrix, "K": self.K}

    def to_world(self, pixel, observation) -> np.ndarray:
        return self.env.get_world_coord_from_pixel(np.asarray(pixel, dtype=float), observation.depth)

    def move(self, picks_world, places_world):
        if len(picks_world) == 2:
            self.env.pick_and_place_dual(
                picks_world[0].copy(), places_world[0].copy(),
                picks_world[1].copy(), places_world[1].copy(),
            )
        else:
            self.env.pick_and_place_single(picks_world[0].copy(), places_world[0].copy())

    def reset(
        self,
        config=None,
        state=None,
        speeds=None,
        cloth3d=False,
        random_angle=0.0,
        size_scale=1.0,
        translate=None,
    ):
        """复位仿真场景（真机后端不会用到 config/state）。

        random_angle：绕竖直轴随机旋转布料；size_scale：随机缩放布料大小；
        translate：随机平移布料，避免每次都是同一个初始摆位。
        """
        from msufold.env._pyflex_bootstrap import ensure_pyflex
        from msufold.env.softgym_cloth_env import rotate_particles

        pyflex = ensure_pyflex()
        speeds = speeds or {}
        self.env.reset(config=config, state=state, cloth3d=cloth3d, **speeds)

        if random_angle:
            rotate_particles([0, random_angle, 0])
        if size_scale != 1.0:
            scale_cloth(size_scale, pyflex)
        if translate is not None:
            translate_cloth(np.asarray(translate, dtype=float), pyflex)

        if random_angle or size_scale != 1.0 or translate is not None:
            for _ in range(300):
                pyflex.step()
                if np.all(np.abs(pyflex.get_velocities()) < 0.2):
                    break

    def close(self):
        self.env.close()


class RealRobotBackend(ExecutionBackend):
    """真机后端：像素点 -> 手眼标定 -> 机械臂执行。

    camera / calibration / arms 三件套由用户用自己机器人的 SDK 实现后注入：

        camera.capture()                      -> (rgb, depth)      # 相机取图
        calibration.pixel_to_world(pixel, depth) -> np.ndarray(3,)  # 手眼标定 + 深度
        calibration.camera_params()            -> {"K":..., "matrix_world_to_camera":...}
        arms.pick_and_place_single(pick, place)
        arms.pick_and_place_dual(pick_l, place_l, pick_r, place_r)
        arms.home()                                                 # 回到拍照位

    只要实现这三个对象，SKILL.md 与 skill 代码无需任何改动即可上真机。
    """

    name = "real"

    def __init__(self, camera=None, calibration=None, arms=None):
        self.camera = camera
        self.calibration = calibration
        self.arms = arms
        for key, value in (("camera", camera), ("calibration", calibration), ("arms", arms)):
            if value is None:
                print(f"[RealRobotBackend] 警告: 未提供 {key}，真机执行会在用到时报错")

    def observe(self) -> Observation:
        from msufold.data.utils import get_mask_from_depth

        rgb, depth = self.camera.capture()
        return Observation(rgb=rgb, depth=depth, mask=get_mask_from_depth(depth))

    def processor_kwargs(self) -> dict:
        if self.calibration is None:
            return {}
        return self.calibration.camera_params()

    def to_world(self, pixel, observation) -> np.ndarray:
        return np.asarray(
            self.calibration.pixel_to_world(np.asarray(pixel, dtype=float), observation.depth),
            dtype=float,
        )

    def move(self, picks_world, places_world):
        if len(picks_world) == 2:
            self.arms.pick_and_place_dual(
                picks_world[0], places_world[0], picks_world[1], places_world[1]
            )
        else:
            self.arms.pick_and_place_single(picks_world[0], places_world[0])

    def reset(self, **kwargs):
        """真机不需要加载仿真状态，回到拍照位即可。"""
        if self.arms is not None:
            self.arms.home()


def build_backend(name: str, **kwargs):
    """按名字构造后端。真机后端需要传入 camera/calibration/arms。"""
    if name == "softgym":
        return SoftgymBackend(env=kwargs["env"], K=kwargs["K"])
    if name == "real":
        return RealRobotBackend(
            camera=kwargs.get("camera"),
            calibration=kwargs.get("calibration"),
            arms=kwargs.get("arms"),
        )
    raise ValueError(f"未知后端: {name}（可选 softgym / real）")
