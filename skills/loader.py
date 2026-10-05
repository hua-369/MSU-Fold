"""SKILL.md 加载器。

每个技能是一个目录，目录里必须有 SKILL.md（YAML frontmatter + Markdown 正文），
与 Agent Skills 规范保持一致：

    skills/<skill-name>/SKILL.md
    --- 
    name: bimanual-tshirt-fold
    description: ...
    metadata:
      arms: bimanual
      ...
    ---

frontmatter 里的 metadata 是运行时的唯一事实来源（场景类型、步数、权重路径等），
skills/__init__.py 用它把 SKILL.md 绑定到对应的 Python 技能类。
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class SkillSpec:
    """一个 SKILL.md 解析出来的技能规格。"""

    name: str
    description: str
    metadata: Dict
    path: Path
    body: str = ""

    @property
    def arms(self) -> str:
        return self.metadata.get("arms", "unimanual")

    @property
    def alias(self) -> Optional[str]:
        """旧版 CamelCase 技能名（如 TshirtFold），用于向后兼容。"""
        return self.metadata.get("alias")

    @property
    def cloth_type(self) -> str:
        return self.metadata.get("cloth_type", "Square")

    @property
    def checkpoint(self) -> Optional[str]:
        return self.metadata.get("checkpoint")

    @property
    def config(self) -> Optional[str]:
        return self.metadata.get("config")

    @property
    def skill_key(self) -> str:
        """绑定到 Python 技能类的键：双臂用 dual_task，单臂用 skill_class。"""
        return self.metadata.get("dual_task") or self.metadata.get("skill_class", "")

    def resolve_path(self, key: str) -> Optional[str]:
        """把 metadata 里相对仓库根的路径转成绝对路径。"""
        value = self.metadata.get(key)
        if value is None:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        return str(path)


def parse_skill_md(path: Path) -> SkillSpec:
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER.match(text)
    if match is None:
        raise ValueError(f"{path} 缺少 YAML frontmatter（应以 --- 开头）")
    frontmatter = yaml.safe_load(match.group(1)) or {}
    body = text[match.end():]
    name = frontmatter.get("name")
    description = frontmatter.get("description")
    if not name or not description:
        raise ValueError(f"{path} 的 frontmatter 必须包含 name 和 description")
    return SkillSpec(
        name=name,
        description=description,
        metadata=frontmatter.get("metadata") or {},
        path=path,
        body=body,
    )


def load_skill_specs(root: Optional[Path] = None) -> Dict[str, SkillSpec]:
    """扫描 skills/ 下所有 */SKILL.md，返回 {skill name: SkillSpec}。"""
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    specs: Dict[str, SkillSpec] = {}
    for skill_md in sorted(root.glob("*/SKILL.md")):
        spec = parse_skill_md(skill_md)
        if spec.name != skill_md.parent.name:
            raise ValueError(
                f"{skill_md}: frontmatter 的 name({spec.name}) 必须与目录名"
                f"({skill_md.parent.name}) 一致"
            )
        specs[spec.name] = spec
    return specs
