import math

from torch import nn

from . import Components
from .modules import ConvDecoder


class PickPlaceConvDecoder(nn.Module):
    """共享的 pick / place 热力图头（单臂与双臂步共用同一套权重）。

    双臂步的第二个抓取点不是靠额外的解码头得到的，而是在推理时从同一张
    热力图上取 Top-2 峰（见 skills/bimanual_skill.py::sample_topk_peaks）。
    """

    def __init__(
        self,
        dim,
        fusion_model,
        num_patches,
        patch_size,
        compute_mask=False,
        detach_mask=False,
        **kwargs,
    ):
        super().__init__()
        self.fusion = Components.get_by_name(fusion_model, dim=dim, **kwargs)
        self.num_patches_sqrt = int(math.sqrt(num_patches))

        if compute_mask:
            self.detach_mask = detach_mask

            self.mask_head = ConvDecoder(dim)
        else:
            self.mask_head = None

        # pick decoder
        self.pick_decoder = ConvDecoder(dim)

        # place decoder
        self.place_decoder = ConvDecoder(dim)

    def reshape_output(self, x, input_dim):
        x = x.view(
            x.size(0),
            self.num_patches_sqrt,
            self.num_patches_sqrt,
            input_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, *inputs, **kwargs):
        return_dict = {}

        dim = inputs[-1].shape[-1]

        fused_features, return_dict["attn_weights"], return_dict["out_features"] = self.fusion(
            *inputs, **kwargs
        )

        if self.mask_head is not None:
            # Obviate patch token and unpatchify
            mask = self.mask_head(self.reshape_output(inputs[-1][:, 1:, :], dim))
            return_dict["mask_heatmap"] = mask.squeeze(1).sigmoid()

        fused_features = self.reshape_output(fused_features[:, 1:, :], dim)

        if self.mask_head is None:
            return_dict["pick_heatmap"] = self.pick_decoder(fused_features).squeeze(1).sigmoid()
        else:
            if self.detach_mask:
                return_dict["pick_heatmap"] = (
                    self.pick_decoder(fused_features).squeeze(1).sigmoid()
                    * return_dict["mask_heatmap"].detach()
                )
            else:
                return_dict["pick_heatmap"] = (
                    self.pick_decoder(fused_features).squeeze(1).sigmoid()
                    * return_dict["mask_heatmap"]
                )

        return_dict["place_heatmap"] = self.place_decoder(fused_features).squeeze(1).sigmoid()
        return return_dict
