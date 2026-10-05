"""Bimanual + unimanual mixed multi-step folding tasks.

设计原则:
1. 每个任务是一条**有序的多步折叠序列**, 和单臂任务一样有先后依赖关系;
2. 序列的前面几步用**双臂** (布料大, 需要两个抓手一起抓, 对应 pick_and_place_dual);
3. 折到一定程度后布料变小/变窄, **双臂不再适用**, 后面的步骤改用**单臂**
   (对应 pick_and_place_single);
4. 每一步都有独立指令: 双臂步会在指令前加双臂提示 (dual_instruction),
   单臂步保持和原来单臂数据集一样的写法 (不加前缀), 这样能和已有的
   single_data 直接融合, 模型根据指令自行选择单臂/双臂。

关键点布局 (与单臂数据完全一致):
    Square  (3x3 网格):  0 1 2 / 3 4 5 / 6 7 8
                         0=左上 2=右上 6=左下 8=右下 4=中心
                         1=上边中点 7=下边中点 3=左边中点 5=右边中点
    Tshirt  (8 点): 0=左肩 1=右肩 2=左袖 3=左胸内侧 4=右胸内侧
                    5=右袖 6=左下摆 7=右下摆
    Trousers(8 点): 0=左腰 3=右腰 4=左裤脚 7=右裤脚
                    1/2=腰部参考点 5/6=裤脚参考点
"""  # noqa: D400

import random

#################################################
###############Dual-arm templates################
#################################################

DUAL_PREFIX_TEMPLATES = [
    "Using both arms, ",
    "Using both hands, ",
    "Using two arms, ",
    "Using two robotic arms, ",
    "Using both grippers, ",
    "Using two grippers, ",
    "With both arms working together, ",
    "With both hands grasping at the same time, ",
    "With two arms grasping simultaneously, ",
]

DUAL_COLON_TEMPLATES = [
    "Bimanual fold: ",
    "Dual-arm fold: ",
    "Dual-arm task: ",
    "Bimanual task: ",
    "Perform a bimanual fold: ",
    "Perform a dual-arm fold: ",
]

DUAL_VERB_TEMPLATES = [
    "Use both arms to ",
    "Use two arms to ",
    "Use both hands to ",
]

DUAL_SUFFIX_TEMPLATES = [
    " Use both arms at the same time.",
    " Use both hands at the same time.",
    " Perform this fold with both arms simultaneously.",
    " Do this fold with two arms together.",
]


def dual_instruction(lang):
    """把一条基础折叠指令包装成**双臂**指令 (加双臂提示前缀)。"""
    r = random.random()
    if r < 0.45:
        return random.choice(DUAL_COLON_TEMPLATES) + lang
    elif r < 0.85:
        body = lang[0].lower() + lang[1:] if lang[:1].isupper() else lang
        return random.choice(DUAL_PREFIX_TEMPLATES) + body
    else:
        body = lang[0].lower() + lang[1:] if lang[:1].isupper() else lang
        return random.choice(DUAL_VERB_TEMPLATES) + body + random.choice(
            DUAL_SUFFIX_TEMPLATES
        )


def single_instruction(lang):
    """单臂步骤保持与原单臂数据集一致的写法 (不加任何前缀)。"""
    return lang


#################################################
#################Square cloth####################
#################################################
class MixedSquareHalfFold:
    """Square 布料: 双臂对折 -> 单臂垂直方向对折 -> 单臂再对折.

    类似叠毛巾: 先用两只手把它对折 (布料大), 后面随着布料变小,
    两只手施展不开, 改用一个抓手继续对折。
    """

    def __init__(self, randomize_pose=True):
        self.cloth_type = "Square"
        self.randomize_pose = randomize_pose
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
        # 双臂半折: 抓一条边的两个角, 折到对边的两个角
        self.dual_edge_pairs = {
            "up": ([0, 2], [6, 8]),
            "down": ([6, 8], [0, 2]),
            "left": ([0, 6], [2, 8]),
            "right": ([2, 8], [0, 6]),
        }
        # 单臂半折: 抓一条边的中点, 折到对边的中点 (与 StraightFold 单臂步骤一致)
        self.single_edge_points = {
            "up": (1, 7),
            "down": (7, 1),
            "left": (3, 5),
            "right": (5, 3),
        }
        self.opposite_edge = {"up": "down", "down": "up", "left": "right", "right": "left"}

        self.dual_half_templates = [
            "Fold the cloth in half from the {which1} side to the {which2} side.",
            "Halve the cloth by folding it from the {which1} to the {which2}.",
            "Bring the {which1} and {which2} sides of the cloth together to make a fold down the middle.",
            "Make a crease down the middle of the cloth from the {which1} to the {which2}.",
        ]
        self.single_half_templates = [
            "Fold the cloth in half again from the {which1} side to the {which2} side.",
            "Halve the folded cloth once more, folding from {which1} to {which2}.",
            "Fold the already folded cloth in half from the {which1} to the {which2}.",
            "Make another fold down the middle of the cloth, from the {which1} to the {which2}.",
        ]
        # 以下结构用于可选的第三步 (折两次后再对折), 当前版本不再使用
        self.final_corner_folds = {
            ("vertical", "left"): [(5, 8), (8, 5)],
            ("vertical", "right"): [(3, 6), (6, 3)],
            ("horizontal", "up"): [(4, 5), (5, 4)],
            ("horizontal", "down"): [(1, 2), (2, 1)],
        }
        self.final_templates = [
            "Fold the folded cloth in half one more time.",
            "Halve the folded cloth one last time.",
            "Make one more fold in the folded cloth.",
        ]

    def get_action_instruction(self):
        actions = []

        # ---- 第 1 步: 双臂半折 (布料最大, 需要两只手臂) ----
        first = random.choice(list(self.dual_edge_pairs.keys()))
        pick_idxs, place_idxs = self.dual_edge_pairs[first]
        axis = "vertical" if first in ("up", "down") else "horizontal"
        lang = random.choice(self.dual_half_templates).format(
            which1=random.choice(self.position_templates[first]),
            which2=random.choice(self.position_templates[self.opposite_edge[first]]),
        )
        actions.append(
            {
                "type": "dual",
                "pick": pick_idxs,
                "place": place_idxs,
                "gamma": 1.0,
                "instruction": dual_instruction(lang),
            }
        )

        # ---- 第 2 步: 单臂沿垂直方向对折 ----
        if axis == "vertical":
            second = random.choice(["left", "right"])
        else:
            second = random.choice(["up", "down"])
        pick_idx, place_idx = self.single_edge_points[second]
        lang = random.choice(self.single_half_templates).format(
            which1=random.choice(self.position_templates[second]),
            which2=random.choice(self.position_templates[self.opposite_edge[second]]),
        )
        actions.append(
            {
                "type": "single",
                "pick": pick_idx,
                "place": place_idx,
                "gamma": 0.95,
                "instruction": single_instruction(lang),
            }
        )

        # 注: 两步折完之后布料已经足够小, 不需要再做额外动作
        # (原先的第三步 "单臂再对折" 已按需求移除)

        unseen_flags = [0] * len(actions)
        return actions, unseen_flags


class MixedSquareCornerFold:
    """Square 布料: 双臂折两个角 -> 双臂折另外两个角.

    布料平整/很大时两只手各抓一个角同时折向中心; 两组角都折到中心之后
    布料已经足够小, 不需要再做额外动作。
    """

    def __init__(self, randomize_pose=True):
        self.cloth_type = "Square"
        self.randomize_pose = randomize_pose
        self.pick_speed = 0.005
        self.move_speed = 0.005
        self.place_speed = 0.005
        self.lift_height = 0.1

        # (第一对角组合, 第二对角组合)
        self.pair_configs = [
            (([0, 2], "upper left", "upper right"), ([6, 8], "lower left", "lower right")),
            (([0, 6], "upper left", "lower left"), ([2, 8], "upper right", "lower right")),
            (([0, 8], "upper left", "lower right"), ([2, 6], "upper right", "lower left")),
        ]
        self.single_edge_points = {"up": (1, 7), "down": (7, 1), "left": (3, 5), "right": (5, 3)}
        self.position_templates = {
            "up": ["upper", "top"],
            "down": ["lower", "bottom"],
            "left": ["left", "leftmost"],
            "right": ["right", "rightmost"],
        }
        self.opposite_edge = {"up": "down", "down": "up", "left": "right", "right": "left"}

        self.dual_corner_templates = [
            "Fold the {which1} corner and the {which2} corner of the cloth towards the center.",
            "Bring the {which1} corner and the {which2} corner of the fabric to the middle.",
            "Create folds from the {which1} and {which2} corners of the cloth towards the center.",
        ]
        self.dual_corner_templates_second = [
            "Fold the {which1} corner and the {which2} corner of the cloth towards the center as well.",
            "Bring the remaining {which1} and {which2} corners of the fabric to the middle too.",
            "Fold the other {which1} and {which2} corners of the cloth towards the center, after that first pair.",
        ]
        self.single_finish_templates = [
            "Fold the folded cloth in half from the {which1} side to the {which2} side.",
            "Halve the folded cloth one last time, from {which1} to {which2}.",
            "Fold the cloth in half again from the {which1} to the {which2}.",
        ]

    def get_action_instruction(self):
        actions = []
        first_pair, second_pair = random.choice(self.pair_configs)

        for pair, templates in zip(
            (first_pair, second_pair),
            (self.dual_corner_templates, self.dual_corner_templates_second),
        ):
            pick_idxs, name1, name2 = pair
            lang = random.choice(templates).format(which1=name1, which2=name2)
            actions.append(
                {
                    "type": "dual",
                    "pick": pick_idxs,
                    "place": [4, 4],
                    "gamma": 0.85,
                    "instruction": dual_instruction(lang),
                }
            )

        # 注: 两次双臂折完之后布料已经足够小, 不需要再做额外动作
        # (原先的第三步 "单臂收尾对折" 已按需求移除)

        unseen_flags = [0] * len(actions)
        return actions, unseen_flags


#################################################
####################Cloth3d######################
#################################################
class MixedTshirtFold:
    """T 恤: 双臂折两只袖子 -> 双臂把下摆折到领口.

    第一步只能是折袖子 (两只手各抓一只袖子同时向内折); 折好后把下摆
    一起拎到领口; 折两次之后衣服已经足够小, 不需要再做额外动作。
    """

    def __init__(self, randomize_pose=True):
        self.cloth_type = "Tshirt"
        self.randomize_pose = randomize_pose
        self.pick_speed = 0.005
        self.move_speed = 0.005
        self.place_speed = 0.005
        self.lift_height = 0.125

        self.sleeve_templates = [
            "Fold the {which1} sleeve and the {which2} sleeve of the T-shirt towards the inside.",
            "Fold both sleeves of the T-shirt towards the body.",
            "Bring both sleeves of the T-shirt to the center.",
            "Fold the {which1} and {which2} sleeves inward to the halfway point.",
        ]
        self.hem_templates = [
            "Bring the bottom of the T-shirt up towards the neckline after folding the sleeves.",
            "Fold the shirt's hem up towards the top after the sleeves.",
            "Fold the lower part of the T-shirt towards the top once the sleeves are done.",
            "Raise the bottom of the T-shirt to the top after that first fold.",
        ]
        self.finish_templates = [
            "Halve the folded T-shirt one last time.",
            "Fold the folded T-shirt in half one final time.",
            "Finish by folding the already folded T-shirt in half.",
        ]

    def get_action_instruction(self):
        actions = []

        # ---- 第 1 步 (双臂): 同时折两只袖子 ----
        lang = random.choice(self.sleeve_templates).format(which1="left", which2="right")
        actions.append(
            {
                "type": "dual",
                "pick": [2, 5],
                "place": [3, 4],
                "gamma": 1.0,
                "instruction": dual_instruction(lang),
            }
        )

        # ---- 第 2 步 (双臂): 两只手一起把下摆提到领口 ----
        lang = random.choice(self.hem_templates)
        actions.append(
            {
                "type": "dual",
                "pick": [6, 7],
                "place": [0, 1],
                "gamma": 1.0,
                "instruction": dual_instruction(lang),
            }
        )

        # ---- 第 3 步 (单臂): 布料已成一小团, 单臂再对折 ----
        # 注: 折两次之后布料已经足够小, 不需要再做额外动作, 该步骤已移除

        unseen_flags = [0] * len(actions)
        return actions, unseen_flags


class MixedTrousersFold:
    """裤子: 双臂把裤子折成一长条 -> 单臂折叠长条.

    第一步两条手臂分别抓住一侧的腰和裤脚, 一起把这半边翻到另一边,
    裤子变成一条窄长条; 之后长条已经很窄, 双臂施展不开,
    改用单臂把裤腰往裤脚折; 折完布料已经足够小, 不需要再做额外动作。
    """

    def __init__(self, randomize_pose=True):
        self.cloth_type = "Trousers"
        self.randomize_pose = randomize_pose
        self.pick_speed = 0.005
        self.move_speed = 0.005
        self.place_speed = 0.005
        self.lift_height = 0.15

        self.position_templates = {
            "left": ["left", "leftmost", "left-hand"],
            "right": ["right", "rightmost", "right-hand"],
        }
        self.side_pairs = {"left": "right", "right": "left"}
        # 双臂: 抓住一侧的腰+裤脚, 整个翻到另一侧 -> 折成长条
        self.side_templates = {
            "left": ([0, 4], [3, 7]),
            "right": ([3, 7], [0, 4]),
        }
        # 单臂: 腰中部 -> 裤脚中部 (折长条)
        self.length_points = {"first": (2, 6), "second": (1, 5)}

        self.strip_templates = [
            "Fold the Trousers in half, {which1} to {which2}, so they become one long strip.",
            "Fold the Trousers into a long strip, {which1} side over {which2} side.",
            "Fold the Trousers lengthwise, bringing the {which1} side onto the {which2} side.",
        ]
        self.half_templates = [
            "Fold the long strip in half by bringing the waistband down to the hem.",
            "Bring the top of the strip down to the bottom and fold it in half.",
        ]
        self.finish_templates = [
            "Fold the folded strip in half one last time.",
            "Halve the folded Trousers one final time.",
        ]

    def get_action_instruction(self):
        actions = []

        # ---- 第 1 步 (双臂): 折成长条 ----
        fold_action = random.choice(list(self.side_pairs.keys()))
        pick_idxs, place_idxs = self.side_templates[fold_action]
        lang = random.choice(self.strip_templates).format(
            which1=random.choice(self.position_templates[fold_action]),
            which2=random.choice(self.position_templates[self.side_pairs[fold_action]]),
        )
        actions.append(
            {
                "type": "dual",
                "pick": pick_idxs,
                "place": place_idxs,
                "gamma": 1.0,
                "instruction": dual_instruction(lang),
            }
        )

        # ---- 第 2 步 (单臂): 长条很窄, 双臂施展不开, 单臂对折 ----
        pick_idx, place_idx = self.length_points["first"]
        lang = random.choice(self.half_templates)
        actions.append(
            {
                "type": "single",
                "pick": pick_idx,
                "place": place_idx,
                "gamma": 0.95,
                "instruction": single_instruction(lang),
            }
        )

        # 注: 折两次之后布料已经足够小, 不需要再做额外动作
        # (原先的第三步 "单臂再折一次" 已按需求移除)

        unseen_flags = [0] * len(actions)
        return actions, unseen_flags


MixedDemonstrator = {
    "MixedSquareHalfFold": MixedSquareHalfFold,
    "MixedSquareCornerFold": MixedSquareCornerFold,
    "MixedTshirtFold": MixedTshirtFold,
    "MixedTrousersFold": MixedTrousersFold,
}


if __name__ == "__main__":
    random.seed(0)
    for name, cls in MixedDemonstrator.items():
        task = cls(randomize_pose=False)
        actions, unseen_flags = task.get_action_instruction()
        print("=" * 70)
        print(f"{name} | cloth: {task.cloth_type} | steps: {len(actions)}")
        for i, act in enumerate(actions):
            print(f"  step{i} [{act['type']}] pick={act['pick']} -> place={act['place']}"
                  f" gamma={act['gamma']}")
            print(f"        {act['instruction']}")
        print("  unseen_flags:", unseen_flags)
