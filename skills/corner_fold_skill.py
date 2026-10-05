from .base_skill import BaseSkill
from msufold.env.softgym_demonstrators import CornerFold


class CornerFoldSkill(BaseSkill):
    """方形布料：把四个角依次向中心折叠（共 4 步，一步一条指令）。"""

    name = "CornerFold"
    cloth_type = "Square"
    cloth3d = False
    demonstrator_cls = CornerFold
