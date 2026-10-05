from .base_skill import BaseSkill
from msufold.env.softgym_demonstrators import TshirtFold


class TshirtFoldSkill(BaseSkill):
    """T恤：左右袖子先后向内折，再把下摆向上折（共 4 步的完整叠衣流程）。"""

    name = "TshirtFold"
    cloth_type = "Tshirt"
    cloth3d = True
    demonstrator_cls = TshirtFold
