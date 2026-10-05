"""双臂混合任务（训练用 Mixed* 任务）的评估示范器。

与 msufold/env/softgym_demonstrators.py 中单臂示范器接口一致：
    pick_speed / move_speed / place_speed / lift_height
    get_eval_instruction() -> (seen, unseen_instruction, unseen_task)

区别：指令字典里的每一步 pick/place 是**关键点索引列表**，
长度为 2 表示双臂同时抓取（对应 pick_and_place_dual），
长度为 1 表示单臂（对应 pick_and_place_single）：
    {
      "pick":  [[0, 2], [1]],        # 第 0 步双臂抓 0/2 两角, 第 1 步单臂抓 1
      "place": [[6, 8], [7]],
      "gammas": [1.0, 0.95],
      "flags": [0, 0],               # 1 = 该步由 oracle 执行 (评估用)
      "instructions": ["...", "..."],
      "primitives": ["dual", "single"],
    }

关键点布局与单臂完全一致：
    Square:  0 1 2 / 3 4 5 / 6 7 8   (0/2/6/8 四角, 1/3/5/7 边中点, 4 中心)
    Tshirt:  0/1 左右肩, 2/5 左右袖, 3/4 胸口内侧, 6/7 下摆
    Trousers: 0/3 左右腰, 4/7 左右裤脚
"""

import random

DUAL_PREFIX_TEMPLATES = [
    "Using both arms, ",
    "Using both hands, ",
    "Using two arms, ",
    "Using two robotic arms, ",
    "Using both grippers, ",
    "With both arms working together, ",
    "With two arms grasping simultaneously, ",
]

DUAL_COLON_TEMPLATES = [
    "Bimanual fold: ",
    "Dual-arm fold: ",
    "Dual-arm task: ",
    "Perform a bimanual fold: ",
    "Perform a dual-arm fold: ",
]

DUAL_VERB_TEMPLATES = ["Use both arms to ", "Use two arms to ", "Use both hands to "]

DUAL_SUFFIX_TEMPLATES = [
    " Use both arms at the same time.",
    " Use both hands at the same time.",
    " Perform this fold with both arms simultaneously.",
]


def dual_instruction(lang):
    r = random.random()
    if r < 0.45:
        return random.choice(DUAL_COLON_TEMPLATES) + lang
    elif r < 0.85:
        body = lang[0].lower() + lang[1:] if lang[:1].isupper() else lang
        return random.choice(DUAL_PREFIX_TEMPLATES) + body
    body = lang[0].lower() + lang[1:] if lang[:1].isupper() else lang
    return random.choice(DUAL_VERB_TEMPLATES) + body + random.choice(DUAL_SUFFIX_TEMPLATES)


def single_instruction(lang):
    return lang


class MixedSquareHalfFold:
    """① dual 沿一条边对折 ② single 垂直方向对折."""

    cloth_type = "Square"
    seen_tasks = ["up", "left", "right"]
    unseen_tasks = ["down"]

    def __init__(self):
        self.pick_speed = 0.006
        self.move_speed = 0.006
        self.place_speed = 0.005
        self.lift_height = 0.125

        self.position_templates = {
            "up": ["upper", "top", "uppermost"],
            "down": ["lower", "bottom", "lowermost"],
            "left": ["left", "leftmost", "left-hand"],
            "right": ["right", "rightmost", "right-hand"],
        }
        self.dual_edge_pairs = {
            "up": ([0, 2], [6, 8]),
            "down": ([6, 8], [0, 2]),
            "left": ([0, 6], [2, 8]),
            "right": ([2, 8], [0, 6]),
        }
        self.single_edge_points = {
            "up": (1, 7),
            "down": (7, 1),
            "left": (3, 5),
            "right": (5, 3),
        }
        self.opposite_edge = {"up": "down", "down": "up", "left": "right", "right": "left"}

        self.seen_lang_templates = [
            "Fold the cloth in half from the {which1} side to the {which2} side.",
            "Halve the cloth by folding it from the {which1} to the {which2}.",
            "Make a crease down the middle of the cloth from the {which1} to the {which2}.",
        ]
        self.unseen_lang_templates = [
            "Make a fold in the cloth by halving it from the {which1} to the {which2}.",
            "Fold the cloth in half, starting from the {which1} side and meeting the {which2}.",
        ]
        self.single_half_templates = [
            "Fold the cloth in half again from the {which1} side to the {which2} side.",
            "Halve the folded cloth once more, folding from {which1} to {which2}.",
        ]

    def _dual_step(self, direction, templates):
        pick_idxs, place_idxs = self.dual_edge_pairs[direction]
        lang = random.choice(templates).format(
            which1=random.choice(self.position_templates[direction]),
            which2=random.choice(self.position_templates[self.opposite_edge[direction]]),
        )
        return {
            "pick": [pick_idxs],
            "place": [place_idxs],
            "gammas": [1.0],
            "primitives": ["dual"],
            "instructions": [dual_instruction(lang)],
        }

    def _single_step(self, direction):
        pick_idx, place_idx = self.single_edge_points[direction]
        lang = random.choice(self.single_half_templates).format(
            which1=random.choice(self.position_templates[direction]),
            which2=random.choice(self.position_templates[self.opposite_edge[direction]]),
        )
        return {
            "pick": [[pick_idx]],
            "place": [[place_idx]],
            "gammas": [0.95],
            "primitives": ["single"],
            "instructions": [single_instruction(lang)],
        }

    def get_eval_instruction(self):
        def build(directions, dual_templates, flags):
            first = random.choice(directions)
            axis = "vertical" if first in ("up", "down") else "horizontal"
            second = random.choice(["left", "right"] if axis == "vertical" else ["up", "down"])

            data = self._dual_step(first, dual_templates)
            single = self._single_step(second)
            for k in ("pick", "place", "gammas", "primitives", "instructions"):
                data[k] += single[k]
            data["flags"] = list(flags)
            return data

        eval_seen_instruction = build(self.seen_tasks, self.seen_lang_templates, [0, 0])
        eval_unseen_instruction = build(self.seen_tasks, self.unseen_lang_templates, [0, 0])
        eval_unseen_tasks = build(self.unseen_tasks, self.seen_lang_templates, [1, 1])
        return eval_seen_instruction, eval_unseen_instruction, eval_unseen_tasks


class MixedSquareCornerFold:
    """① dual 一对角折向中心 ② dual 另一对角折向中心.

    两步的角组合必须**互补**（4 个角各折一次），与训练数据一致；
    若两步共享角（如 [6,8] 再 [0,6]），第二步只会提起一个新角。
    """

    cloth_type = "Square"
    # 互补组合: 两步合计折 4 个不同的角
    pair_configs = [
        ([0, 2], [6, 8]),   # 上边一对 -> 下边一对
        ([0, 6], [2, 8]),   # 左边一对 -> 右边一对
        ([0, 8], [2, 6]),   # 对角一对 -> 另一对角
    ]
    # held-out 组合（ut 用，双臂任务当前不评 ut）
    unseen_tasks = [([0, 2], [6, 8])]

    def __init__(self):
        self.pick_speed = 0.005
        self.move_speed = 0.005
        self.place_speed = 0.005
        self.lift_height = 0.1

        self.name_by_pair = {
            (0, 2): ("upper left", "upper right"),
            (6, 8): ("lower left", "lower right"),
            (0, 6): ("upper left", "lower left"),
            (2, 8): ("upper right", "lower right"),
            (0, 8): ("upper left", "lower right"),
            (2, 6): ("upper right", "lower left"),
        }

        self.seen_lang_templates = [
            "Fold the {which1} corner and the {which2} corner of the cloth towards the center.",
            "Bring the {which1} corner and the {which2} corner of the fabric to the middle.",
        ]
        self.unseen_lang_templates = [
            "Fold the {which1} corner and the {which2} corner of the cloth towards the midpoint.",
            "Bring the {which1} and {which2} corners of the fabric to the center with folds.",
        ]

    def _dual_step(self, pick_idxs, templates, flags):
        name1, name2 = self.name_by_pair[tuple(pick_idxs)]
        lang = random.choice(templates).format(which1=name1, which2=name2)
        return {
            "pick": [pick_idxs],
            "place": [[4, 4]],
            "gammas": [0.85],
            "primitives": ["dual"],
            "instructions": [dual_instruction(lang)],
            "flags": list(flags),
        }

    def get_eval_instruction(self):
        first_pair, second_pair = random.choice(self.pair_configs)
        unseen_pair_a, unseen_pair_b = random.choice(self.unseen_tasks)

        def build(pair_a, pair_b, templates_a, templates_b, flags):
            data = self._dual_step(pair_a, templates_a, flags[:1])
            other = self._dual_step(pair_b, templates_b, flags[1:])
            for k in ("pick", "place", "gammas", "primitives", "instructions", "flags"):
                data[k] += other[k]
            return data

        eval_seen_instruction = build(
            first_pair, second_pair, self.seen_lang_templates, self.seen_lang_templates, [0, 0]
        )
        eval_unseen_instruction = build(
            first_pair, second_pair, self.seen_lang_templates, self.unseen_lang_templates, [0, 0]
        )
        # 双臂任务不评 ut，这里仅为保持接口一致
        eval_unseen_tasks = build(
            unseen_pair_a, unseen_pair_b, self.seen_lang_templates, self.seen_lang_templates, [1, 1]
        )
        return eval_seen_instruction, eval_unseen_instruction, eval_unseen_tasks


class MixedTshirtFold:
    """① dual 折两只袖子 ② dual 下摆折到领口."""

    cloth_type = "Tshirt"
    seen_tasks = ["sleeves_hem"]
    unseen_tasks = ["shoulders_hem"]

    def __init__(self):
        self.pick_speed = 0.005
        self.move_speed = 0.005
        self.place_speed = 0.005
        self.lift_height = 0.125

        self.sleeve_templates = [
            "Fold both sleeves of the T-shirt towards the body.",
            "Fold the left and right sleeves inward to the halfway point.",
            "Tuck both sleeves of the T-shirt towards the center.",
        ]
        self.hem_templates = [
            "Bring the bottom of the T-shirt up towards the neckline.",
            "Fold the hem of the T-shirt towards the top after folding the sleeves.",
            "Raise the bottom of the T-shirt to the top.",
        ]
        self.unseen_lang_templates = [
            "Fold both sleeves of the T-shirt towards the midpoint of the shirt.",
            "Bring both sleeves to the center seam of the shirt.",
        ]
        self.shoulder_templates = [
            "Bring the shoulders of the T-shirt down to the hem.",
            "Fold the top of the T-shirt down towards the bottom.",
        ]

    def _dual_step(self, pick_idxs, place_idxs, templates, flags):
        lang = random.choice(templates)
        return {
            "pick": [pick_idxs],
            "place": [place_idxs],
            "gammas": [1.0],
            "primitives": ["dual"],
            "instructions": [dual_instruction(lang)],
            "flags": list(flags),
        }

    def get_eval_instruction(self):
        eval_seen_instruction = self._dual_step(
            [2, 5], [3, 4], self.sleeve_templates, [0]
        )
        hem = self._dual_step([6, 7], [0, 1], self.hem_templates, [0])
        for k in ("pick", "place", "gammas", "primitives", "instructions", "flags"):
            eval_seen_instruction[k] += hem[k]

        eval_unseen_instruction = self._dual_step(
            [2, 5], [3, 4], self.unseen_lang_templates, [0]
        )
        hem2 = self._dual_step([6, 7], [0, 1], self.hem_templates, [0])
        for k in ("pick", "place", "gammas", "primitives", "instructions", "flags"):
            eval_unseen_instruction[k] += hem2[k]

        eval_unseen_tasks = self._dual_step(
            [0, 1], [6, 7], self.shoulder_templates, [1]
        )
        hem3 = self._dual_step([6, 7], [0, 1], self.hem_templates, [1])
        for k in ("pick", "place", "gammas", "primitives", "instructions", "flags"):
            eval_unseen_tasks[k] += hem3[k]

        return eval_seen_instruction, eval_unseen_instruction, eval_unseen_tasks


class MixedTrousersFold:
    """① dual 折成一长条 ② single 长条对折."""

    cloth_type = "Trousers"
    seen_tasks = ["left", "right"]
    unseen_tasks = ["right"]

    def __init__(self):
        self.pick_speed = 0.005
        self.move_speed = 0.005
        self.place_speed = 0.005
        self.lift_height = 0.15

        self.position_templates = {
            "left": ["left", "leftmost", "left-hand"],
            "right": ["right", "rightmost", "right-hand"],
        }
        self.side_pairs = {"left": "right", "right": "left"}
        self.side_templates = {
            "left": ([0, 4], [3, 7]),
            "right": ([3, 7], [0, 4]),
        }
        self.strip_templates = [
            "Fold the Trousers in half, {which1} to {which2}, so they become one long strip.",
            "Fold the Trousers lengthwise, bringing the {which1} side onto the {which2} side.",
        ]
        self.half_templates = [
            "Fold the long strip in half by bringing the waistband down to the hem.",
            "Bring the top of the strip down to the bottom and fold it in half.",
        ]

    def get_eval_instruction(self):
        def build(fold_action, dual_flags, single_flags):
            pick_idxs, place_idxs = self.side_templates[fold_action]
            lang = random.choice(self.strip_templates).format(
                which1=random.choice(self.position_templates[fold_action]),
                which2=random.choice(self.position_templates[self.side_pairs[fold_action]]),
            )
            data = {
                "pick": [pick_idxs],
                "place": [place_idxs],
                "gammas": [1.0],
                "primitives": ["dual"],
                "instructions": [dual_instruction(lang)],
                "flags": list(dual_flags),
            }
            data["pick"] += [[2]]
            data["place"] += [[6]]
            data["gammas"] += [0.95]
            data["primitives"] += ["single"]
            data["instructions"] += [single_instruction(random.choice(self.half_templates))]
            data["flags"] += list(single_flags)
            return data

        eval_seen_instruction = build(random.choice(self.seen_tasks), [0], [0])
        eval_unseen_instruction = build(random.choice(self.seen_tasks), [0], [0])
        eval_unseen_tasks = build(random.choice(self.unseen_tasks), [1], [1])
        return eval_seen_instruction, eval_unseen_instruction, eval_unseen_tasks


DualDemonstrator = {
    "MixedSquareHalfFold": MixedSquareHalfFold,
    "MixedSquareCornerFold": MixedSquareCornerFold,
    "MixedTshirtFold": MixedTshirtFold,
    "MixedTrousersFold": MixedTrousersFold,
}

dual_task_to_cloth_type = {
    "MixedSquareHalfFold": "Square",
    "MixedSquareCornerFold": "Square",
    "MixedTshirtFold": "Tshirt",
    "MixedTrousersFold": "Trousers",
}
