from .base_skill import BaseSkill
from msufold.env.softgym_demonstrators import TrousersFold


class TrousersFoldSkill(BaseSkill):
    """裤子：左右对折后，再从腰部向下摆对折（共 3 步的完整叠裤流程）。"""

    name = "TrousersFold"
    cloth_type = "Trousers"
    cloth3d = True
    demonstrator_cls = TrousersFold
