from .base_skill import BaseSkill
from msufold.env.softgym_demonstrators import TriangleFold


class TriangleFoldSkill(BaseSkill):
    """方形布料：把一个角沿对角线折到对角，形成三角形（共 2 步）。"""

    name = "TriangleFold"
    cloth_type = "Square"
    cloth3d = False
    demonstrator_cls = TriangleFold
