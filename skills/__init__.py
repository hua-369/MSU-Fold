"""折叠技能库注册表。

技能的**描述与配置**写在各技能目录的 SKILL.md 里（Agent Skills 规范），
本模块负责把 SKILL.md 绑定到可执行的 Python 技能类：

    from skills import SKILLS, SKILL_DESCRIPTIONS, SKILL_SPECS
    skill = SKILLS["bimanual-tshirt-fold"]()

SKILLS 同时接受目录名（kebab-case，规范写法）与旧版 CamelCase 别名
（如 "TshirtFold"），保证已有脚本不被破坏。
"""

from typing import Dict, Type

from .base_skill import BaseSkill
from .bimanual_skill import BIMANUAL_SKILL_CLASSES
from .corner_fold_skill import CornerFoldSkill
from .loader import SkillSpec, load_skill_specs
from .straight_fold_skill import StraightFoldSkill
from .triangle_fold_skill import TriangleFoldSkill
from .trousers_fold_skill import TrousersFoldSkill
from .tshirt_fold_skill import TshirtFoldSkill

UNIMANUAL_SKILL_CLASSES: Dict[str, Type[BaseSkill]] = {
    "CornerFold": CornerFoldSkill,
    "TriangleFold": TriangleFoldSkill,
    "StraightFold": StraightFoldSkill,
    "TshirtFold": TshirtFoldSkill,
    "TrousersFold": TrousersFoldSkill,
}

#: {SKILL.md 的 name: SkillSpec}
SKILL_SPECS: Dict[str, SkillSpec] = load_skill_specs()


def _build_registry() -> Dict[str, Type[BaseSkill]]:
    registry: Dict[str, Type[BaseSkill]] = {}
    for name, spec in SKILL_SPECS.items():
        if spec.arms == "bimanual":
            cls = BIMANUAL_SKILL_CLASSES.get(spec.skill_key)
        else:
            cls = UNIMANUAL_SKILL_CLASSES.get(spec.skill_key)
        if cls is None:
            raise ValueError(
                f"SKILL.md {spec.path} 的 metadata.skill_class/dual_task="
                f"{spec.skill_key!r} 没有对应的技能类"
            )
        registry[name] = cls
        if spec.alias:
            registry[spec.alias] = cls
    return registry


#: {技能名（或别名）: 技能类}
SKILLS: Dict[str, Type[BaseSkill]] = _build_registry()

#: 供大模型做 skill 路由用的描述：{技能名: description}
SKILL_DESCRIPTIONS: Dict[str, str] = {
    name: spec.description for name, spec in SKILL_SPECS.items()
}


def get_skill_spec(skill_name: str) -> SkillSpec:
    """按技能名（或别名）取回它的 SKILL.md 规格。"""
    if skill_name in SKILL_SPECS:
        return SKILL_SPECS[skill_name]
    for spec in SKILL_SPECS.values():
        if spec.alias == skill_name:
            return spec
    raise KeyError(f"未知技能: {skill_name}")


def get_model_paths(skill_name: str):
    """取该技能默认使用的 (config.yaml, checkpoint.pth) 绝对路径。"""
    spec = get_skill_spec(skill_name)
    return spec.resolve_path("config"), spec.resolve_path("checkpoint")


__all__ = [
    "SKILLS",
    "SKILL_DESCRIPTIONS",
    "SKILL_SPECS",
    "SkillSpec",
    "get_skill_spec",
    "get_model_paths",
    "UNIMANUAL_SKILL_CLASSES",
    "BIMANUAL_SKILL_CLASSES",
]
