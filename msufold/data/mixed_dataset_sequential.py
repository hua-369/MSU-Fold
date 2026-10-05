"""双臂（以及可选混合）的顺序数据集。

加载格式与 single_data_sequential 完全一致的 episodes pkl
（如 datasets/dual_data_sequential/All_150.pkl，纯双臂数据），
通过 `dataset@train_dataset=bimanual_sequential` 选用；
`use_dual_data=true` / `mixed_sequential` 时也可用它做混合训练
（此时 dataset_path 指向单臂 pkl，dual_dataset_path 指向双臂 pkl）。

与 SingleDatasetSequential 的唯一区别：
  * 提供 collate_fn，把 batch 内点数不一致的 pick/place padding 到相同形状
    （单臂 1 个点、双臂 2 个点；用 -1 占位，
    processor / 空间增强 / loss / metrics 都会跳过这些无效点）。
"""

import pickle

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from msufold.env.softgym_utils import get_matrix_world_to_camera

from . import BaseDataset
from .utils import DENG_CAMERA_PARAMS, get_mask_from_depth

# 需要按点数 padding 的关键字
_PAD_KEYS = ("pick", "place")
_PAD_VALUE = -1.0


class MixedDatasetSequential(BaseDataset):
    def __init__(self, cfg, *args, **kwargs):
        self.max_context_length = cfg.max_context_length

        super().__init__(cfg, *args, **kwargs, max_context_length=self.max_context_length)

        with open(self.dataset_path, "rb") as f:
            data = pickle.load(f)
        self.episodes = list(data["episodes"])

        # 可选：再混入一份双臂 episodes（仅混合训练时使用，纯双臂训练不设置）
        dual_path = cfg.get("dual_dataset_path", None)
        if dual_path is not None and dual_path != self.dataset_path:
            with open(dual_path, "rb") as f:
                dual_data = pickle.load(f)
            self.episodes += list(dual_data["episodes"])

        self.img_size = None

        self.num_events = 0
        self.event_data = []
        for num_episode, episode in enumerate(self.episodes):
            for num_event, depth in enumerate(episode["depth"]):
                if self.img_size is None:
                    self.img_size = depth.shape[0]

                self.event_data.append({
                    "episode": num_episode,
                    "index": num_event,
                    "context": list(range(num_event)),
                })
                assert num_event - 1 <= self.max_context_length, (
                    f"The context of the dataset exceeds {self.max_context_length} "
                    f"for episode {num_episode}"
                )
                self.num_events += 1

    def __len__(self):
        return self.num_events

    def __getitem__(self, event_index):
        event_data = self.event_data[event_index]
        episode = self.episodes[event_data["episode"]]

        depth = episode["depth"][event_data["index"]] / self.depth_scale
        mask = get_mask_from_depth(depth)

        pick_pixel = np.asarray(episode["pick"][event_data["index"]], dtype=float).reshape(-1, 2)
        place_pixel = np.asarray(episode["place"][event_data["index"]], dtype=float).reshape(-1, 2)

        context = [
            {
                "rgb": episode["rgbs"][idx],
                "depth": episode["depth"][idx] / self.depth_scale,
                "mask": get_mask_from_depth(episode["depth"][idx] / self.depth_scale),
            }
            for idx in event_data["context"]
        ]

        return self.processor(
            rgb=episode["rgbs"][event_data["index"]],
            depth=depth,
            mask=mask,
            instruction=episode["instruction"][event_data["index"]],
            matrix_world_to_camera=get_matrix_world_to_camera(DENG_CAMERA_PARAMS),
            pick=pick_pixel,
            place=place_pixel,
            context=context,
        )

    @staticmethod
    def collate_fn(batch):
        """把 batch 内点数不同的 pick/place padding 到 (max_n, 2)。

        单臂样本是 1 个点、双臂样本是 2 个点，默认 collate 直接 stack 会报错。
        不足的部分用 -1 填充，下游（gaussmap / 空间增强 / 指标）会忽略无效点。
        """
        padded_batch = []
        for key in batch[0].keys():
            if key in _PAD_KEYS:
                values = [sample[key] for sample in batch]
                if all(isinstance(v, torch.Tensor) for v in values):
                    max_n = max(v.shape[0] for v in values)
                    if max_n > 1:
                        new_values = []
                        for v in values:
                            if v.shape[0] < max_n:
                                pad = torch.full(
                                    (max_n - v.shape[0], v.shape[-1]),
                                    _PAD_VALUE,
                                    dtype=v.dtype,
                                )
                                v = torch.cat([v, pad], dim=0)
                            new_values.append(v)
                        values = new_values

                # 就地替换，避免破坏其它字段的默认 collate
                for sample, v in zip(batch, values):
                    sample[key] = v

        padded_batch = batch
        try:
            return default_collate(padded_batch)
        except TypeError:
            # 某些字段（如字符串列表）默认 collate 处理不了时回退成 list
            return padded_batch
