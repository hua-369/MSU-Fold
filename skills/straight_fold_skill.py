from .base_skill import BaseSkill
from msufold.env.softgym_demonstrators import StraightFold


class StraightFoldSkill(BaseSkill):
    """长方形布料：沿边对折（共 3 步，含双臂多步与单步）。"""

    name = "StraightFold"
    cloth_type = "Rectangular"
    cloth3d = False
    demonstrator_cls = StraightFold
