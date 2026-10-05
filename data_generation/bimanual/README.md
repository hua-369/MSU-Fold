# 多步折叠演示生成（单臂/双臂混合）

为折叠机器人数据集新增的演示生成模块：每条演示是**多步折叠序列**，
每一步为 `dual`（双臂同时 pick/place）或 `single`（单臂）。
整体顺序为 **先双臂、后单臂**：布料大时用双臂，折小后双臂施展不开，收尾用单臂。

## 任务（每个任务固定 2 步）

| 任务 | 布料 | 步骤 |
|---|---|---|
| `MixedSquareHalfFold` | Square | ① dual 沿一条边对折（抓两个角折到对边两个角）② single 垂直方向再对折（抓边中点） |
| `MixedSquareCornerFold` | Square | ① dual 一对角折向中心 ② dual 另一对角折向中心 |
| `MixedTshirtFold` | Tshirt | ① dual 两只袖子向内折 ② dual 下摆向上折到领口 |
| `MixedTrousersFold` | Trousers | ① dual 折成一长条 ② single 长条对折（腰折到裤脚） |

关键点布局（沿用单臂约定）：Square 为 3×3 网格 `0 1 2 / 3 4 5 / 6 7 8`（0/2/6/8 为四角，1/3/5/7 为边中点，4 为中心）；Tshirt/Trousers 的 8 点含义见 `demonstrator.py` 头部注释。

## 文件

- `demonstrator.py` — 任务定义（每步的 pick/place 关键点、指令模板、双臂指令包装 `dual_instruction`）
- `generate_demo_fold.py` — 运行仿真生成演示（可选 mp4/gif 视频）
- `make_video.py` — 从已有演示合成步级预览图与 gif

## 指令格式

- 双臂步骤：随机加双臂提示，如 `Using both arms, ...`、`Dual-arm fold: ...`、`Use both hands to ...`（见 `dual_instruction`）。
- 单臂步骤：保持与原单臂数据一致的自然语言，不加前缀。
  这样与现有单臂数据集混训时，模型可依据指令（及输出点数）区分单/双臂。

## 生成演示

```bash
cd data_generation
conda activate MSU-Fold   # PYFLEXROOT/CLOTH3D_PATH 等已固化在该环境
python bimanual/generate_demo_fold.py --task MixedSquareHalfFold \
    --num_demonstrations 1 --record_video --seed 0
```

输出（`raw_data_mixed/<Task>/<demo>/`）：

```
rgb/0.png, 1.png, 2.png     # 每步动作前后的观测（2 步 → 3 帧）
depth/0.png, 1.png, 2.png
viz/0.png, 1.png, 2.png     # 每步 pick→place 箭头可视化
info.pkl                    # 每步 pick/place/指令/primitive("dual"|"single")/unseen_flags
demo.mp4                    # 演示视频（--record_video 时生成）
```

`info.pkl` 中每步 `pick`/`place` 形状：dual 为 `(2,2)`（两臂各一点），single 为 `(1,2)`。

## 预览

```bash
python bimanual/make_video.py --all_tasks --demo 0
# 生成 raw_data_mixed/preview/<Task>_steps.png 与 <Task>_demo.gif
```
