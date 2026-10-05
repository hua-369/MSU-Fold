import torch
from torch import nn

from msufold.env import Action

from .utils import sample_from_heatmap


class Models:
    @staticmethod
    def get_by_name(cfg, *args, **kwargs):
        if cfg.name == "siglip_sequential":
            from .siglip import SiglipSequential as Model
        else:
            raise ValueError(f"Model {cfg.name} not recognized")

        del cfg.name
        return Model(*args, **cfg, **kwargs)


class Components:
    @staticmethod
    def get_by_name(name, *args, **kwargs):
        if name == "pick_place_convdecoder":
            from .pickplace import PickPlaceConvDecoder as Model
        elif name == "pick_place_transdecoder":
            from .pickplace import PickPlaceTransDecoder as Model
        elif name == "concat_transformer":
            from .fusion import ConcatTransformer as Model
        elif name == "crossattention":
            from .fusion import CrossAttention as Model
        else:
            raise ValueError(f"Model {name} not recognized")
        return Model(*args, **kwargs)


class BaseModel(nn.Module):
    def __init__(
        self,
        image_size,
        device,
        constrain_pick_mask=True,
        **kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.device = device
        self.constrain_pick_mask = constrain_pick_mask
        if not self.constrain_pick_mask:
            print("This run will not constrain the pick to the mask")

    def forward(self, x):
        raise NotImplementedError

    def frozen_submodule(self, submodule_name):
        for param in self.named_parameters():
            if submodule_name in param[0]:
                param[1].requires_grad = False

    def frozen_all(self):
        for param in self.named_parameters():
            param[1].requires_grad = False

    def get_action(self, sample, return_raw_output=False, **kwargs):
        """由共享的 pick / place 热力图解码出像素点。

        单臂步取 1 个峰、双臂步取 2 个峰的解码发生在 skills / evaluator 里
        （见 skills/bimanual_skill.py::sample_topk_peaks），这里统一只返回
        热力图的 argmax，热力图本身通过 return_raw_output 一并返回。
        """
        with torch.no_grad():
            output = self.forward(sample, **kwargs)

        pick = sample_from_heatmap(
            output["pick_heatmap"],
            sample.get("mask") if self.constrain_pick_mask else None,
        )
        action = Action(
            pick=pick,
            place=sample_from_heatmap(output["place_heatmap"]),
        )
        if return_raw_output:
            return action, output
        return action
