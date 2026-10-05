#!/usr/bin/env python3
"""Asynchronous folding inference service (HTTP + JSON) for real-robot controllers.

The service only runs inference; it never moves the arms:
    the robot controller posts one observation -> the service uses the skill's step
    instruction and the MSU-Fold model to predict this step's pick / place keypoints
    -> it returns them, and the controller converts them to robot coordinates with
    its own hand-eye calibration and executes the motion.

Endpoints (all under /v1):
    GET  /v1/health                      service and model status
    GET  /v1/skills                      available skills (steps / step instructions)
    POST /v1/sessions                    create a folding session
    POST /v1/sessions/{id}/step          submit a frame, async inference (returns job_id)
    POST /v1/sessions/{id}/step_sync     submit a frame and block for the result
    GET  /v1/jobs/{id}                   poll an async result
    GET  /v1/sessions/{id}               session progress
    DELETE /v1/sessions/{id}             release a session

Example:
    python skill_inference_server.py --host 0.0.0.0 --port 8000 --device cuda:0 \
        --skill unimanual-tshirt-fold \
        --config outputs/unimanual/config.yaml \
        --checkpoint outputs/unimanual/checkpoints/best.pth
"""

import argparse
import base64
import io
import json
import os
import random
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from msufold.data.processor import Processor  # noqa: E402
from msufold.data.utils import get_mask_from_depth  # noqa: E402
from msufold.models import Models  # noqa: E402
from msufold.models.utils import sample_from_heatmap  # noqa: E402
from skills import SKILLS, SKILL_DESCRIPTIONS, SKILL_SPECS, get_model_paths  # noqa: E402
from skills.bimanual_skill import match_places_to_picks, sample_topk_peaks  # noqa: E402

#: 与 inference_skill_softgym.py 保持一致：自然语言路由默认走本机 Ollama
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "qwen3.6:35b"
DEFAULT_CHECKPOINT = str(REPO_ROOT / "outputs" / "unimanual" / "checkpoints" / "best.pth")
#: 会话空闲多久后自动释放（秒）
SESSION_TTL = 2 * 3600
#: 异步任务结果保留多久（秒）
JOB_TTL = 3600


# --------------------------------------------------------------------------- #
# 图像 / 深度 编解码
# --------------------------------------------------------------------------- #
def b64_to_array(b64_string):
    """base64(PNG/JPEG) -> np.ndarray。"""
    raw = base64.b64decode(b64_string.split(",")[-1])
    return np.array(Image.open(io.BytesIO(raw)))


def array_to_b64_png(array):
    """np.ndarray -> base64(PNG)。"""
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def heatmap_to_b64(heatmap):
    """(H, W) float 热力图 -> base64(PNG, 8bit，按自身最大值归一化)。"""
    array = np.asarray(heatmap, dtype=np.float32)
    array = array - array.min()
    array = array / max(float(array.max()), 1e-6)
    return array_to_b64_png((array * 255).astype(np.uint8))


def _hot_colormap(values):
    """(H, W) 0~1 -> (H, W, 3) 0~255，近似 matplotlib 'hot' 伪彩色。"""
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(values / 0.375, 0.0, 1.0)
    green = np.clip((values - 0.375) / 0.375, 0.0, 1.0)
    blue = np.clip((values - 0.75) / 0.25, 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def make_overlay_b64(rgb, heatmaps, alpha=0.55):
    """RGB 底图 + 若干热力图的半透明伪彩色叠加，不画任何标记点。"""
    base = np.asarray(rgb, dtype=np.float32)
    overlay = base.copy()
    for heatmap in heatmaps:
        array = np.asarray(heatmap, dtype=np.float32)
        array = array - array.min()
        array = array / max(float(array.max()), 1e-6)
        colored = _hot_colormap(array) * 255.0
        weight = (array * alpha)[..., None]
        overlay = overlay * (1.0 - weight) + colored * weight
    return array_to_b64_png(np.clip(overlay, 0, 255).astype(np.uint8))


def build_placeholder_depth(height, width, value=0.7):
    """无深度时的占位深度：平面 0.7 + 边缘 1.0（与 scripts/infer_image_heatmaps.py 一致）。"""
    depth = np.full((height, width), value, dtype=np.float32)
    border = max(1, max(height, width) // 50)
    depth[:border, :] = 1.0
    depth[-border:, :] = 1.0
    depth[:, :border] = 1.0
    depth[:, -border:] = 1.0
    return depth


def convert_depth(array, mode="normalized", depth_min=0.4, depth_max=1.2):
    """客户端深度图 -> 训练同域的归一化深度 (0~1，>=0.996 视为背景)。

    mode:
      normalized  已是 0~255 的归一化深度（直接除以 255）
      mm          单位为毫米的 uint16 深度
      m           单位为米的深度
    """
    values = np.asarray(array, dtype=np.float32)
    if mode == "normalized":
        depth = values / 255.0
    else:
        if mode == "mm":
            values = values / 1000.0
        elif mode not in ("m", "meter", "metre"):
            raise ValueError(f"unknown depth mode: {mode}")
        invalid = ~np.isfinite(values) | (values <= 0)
        depth = (values - depth_min) / (depth_max - depth_min)
        depth[invalid] = 1.0
    return np.clip(depth, 0.0, 1.0).astype(np.float32)


def geometry_of(height, width, image_size):
    """复现 Processor 的 Resize(短边->image_size) + CenterCrop(image_size)。

    返回把模型输出坐标 (x, y) 还原到客户端原始图像像素坐标所需的参数。
    """
    short_side = min(height, width)
    scale = image_size / float(short_side)          # 原图 -> 模型输入的缩放比
    resized_h, resized_w = int(height * scale), int(width * scale)
    offset_y = int(round((resized_h - image_size) / 2.0))
    offset_x = int(round((resized_w - image_size) / 2.0))
    return {
        "scale": scale,
        "offset_xy": [offset_x, offset_y],           # 模型输入(224)坐标系下的裁剪偏移
        "scale_back": 1.0 / scale,                   # 模型坐标 -> 原图坐标的比例
        "offset_xy_original": [offset_x / scale, offset_y / scale],
    }


def model_xy_to_image_xy(xy, height, width, image_size):
    """模型输入坐标 (x 右, y 下) -> 客户端原始图像像素坐标。"""
    geom = geometry_of(height, width, image_size)
    x_img = (xy[0] + geom["offset_xy"][0]) * geom["scale_back"]
    y_img = (xy[1] + geom["offset_xy"][1]) * geom["scale_back"]
    return [float(x_img), float(y_img)]


# --------------------------------------------------------------------------- #
# 推理引擎
# --------------------------------------------------------------------------- #
class FoldEngine:
    """模型 + Processor，串行为所有会话提供推理。"""

    #: 类级默认值：即使配置里没有该字段也不会缺属性
    image_size = 224
    max_context_length = 3
    dry_run = False

    def __init__(self, config_path, checkpoint_path, device):
        cfg = OmegaConf.load(config_path)
        # 训练时 Models.get_by_name 会删除 model.name，保存的 config.yaml 里缺失
        cfg.model.name = "siglip_sequential"

        self.image_size = int(cfg.model.image_size)
        self.max_context_length = int(cfg.train_dataset.max_context_length)
        self.device = torch.device(device)

        self.processor = Processor(
            cfg=cfg.processor,
            partition="test",
            max_context_length=self.max_context_length,
            autoprocessor_name=cfg.model.automodel_name,
        )
        print(f"Building model: {cfg.model.name} (image_size={self.image_size})")
        self.model = Models.get_by_name(cfg.model, device=self.device).to(self.device)
        print(f"Loading weights: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        try:
            self.model.load_state_dict(checkpoint["model"])
        except RuntimeError as exc:
            print(f"Strict load failed ({exc}), retrying with strict=False")
            self.model.load_state_dict(checkpoint["model"], strict=False)
        self.model.eval()
        self.epoch = checkpoint.get("epoch", "unknown")
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.dry_run = False
        self.lock = threading.Lock()
        print(f"Model ready (epoch {self.epoch}) on {self.device}")

    @classmethod
    def dry(cls, image_size=224, max_context_length=3):
        """不加载权重的空引擎，仅用于连通性自测。"""
        engine = cls.__new__(cls)
        engine.image_size = image_size
        engine.max_context_length = max_context_length
        engine.device = torch.device("cpu")
        engine.dry_run = True
        engine.epoch = "dry-run"
        engine.config_path = None
        engine.checkpoint_path = None
        engine.lock = threading.Lock()
        return engine

    def predict(self, rgb, depth, mask, instruction, context, want_heatmap=False, n_arms=1):
        """一帧观测 + 本步指令 -> pick/place 热力图关键点。

        模型只有一套共享的 pick/place 热力图头（见 docs/decoding_design_notes.md）：
          n_arms=1 -> 取 1 个峰（argmax，约束在布料 mask 内）；
          n_arms=2 -> 取 top-2 峰（NMS r=12px）+ 就近配对，下发两条臂。
        n_arms 由技能步骤定义给出（skill._n_arms），不由模型预测。
        """
        with self.lock:
            if self.dry_run:
                return self._predict_dry(mask, want_heatmap, n_arms)

            sample = self.processor(
                depth=depth,
                instruction=instruction,
                rgb=rgb,
                mask=mask,
                context=context or [],
            )
            batch = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.unsqueeze(0).to(self.device)

            action, output = self.model.get_action(batch, return_raw_output=True)
            mask_batch = batch.get("mask")
            pick_map = output["pick_heatmap"][0].detach().cpu().numpy()
            place_map = output["place_heatmap"][0].detach().cpu().numpy()
            mask_np = (
                mask_batch.squeeze().detach().cpu().numpy() if mask_batch is not None else None
            )

            def value_at(heatmap, point):
                x, y = int(round(point[0])), int(round(point[1]))
                x = min(max(x, 0), heatmap.shape[1] - 1)
                y = min(max(y, 0), heatmap.shape[0] - 1)
                return float(heatmap[y, x])

            if n_arms <= 1:
                pick = np.asarray(action.pick[0], dtype=float)
                place = np.asarray(action.place[0], dtype=float)
                picks, places = [pick], [place]
            else:
                pick_peaks = sample_topk_peaks(
                    output["pick_heatmap"], n_arms, mask=mask_np
                )
                place_peaks = sample_topk_peaks(
                    output["place_heatmap"], n_arms, mask=mask_np
                )
                picks = [np.asarray(p, dtype=float) for p in pick_peaks]
                # 就近配对：避免两条臂的 pick/place 交叉把布料拧住
                places = [np.asarray(p, dtype=float) for p in
                          match_places_to_picks(pick_peaks, place_peaks)]
                # 按 x 排序：图像左侧的点归左臂，右侧归右臂
                order = sorted(range(len(picks)), key=lambda i: picks[i][0])
                picks = [picks[i] for i in order]
                places = [places[i] for i in order]

        result = {
            "n_arms": int(n_arms),
            "picks": [[float(p[0]), float(p[1])] for p in picks],
            "places": [[float(p[0]), float(p[1])] for p in places],
            "pick_confidences": [value_at(pick_map, p) for p in picks],
            "place_confidences": [value_at(place_map, p) for p in places],
            "pick_peak": float(pick_map.max()),
            "place_peak": float(place_map.max()),
        }
        if want_heatmap:
            result["pick_heatmap_png"] = heatmap_to_b64(pick_map)
            result["place_heatmap_png"] = heatmap_to_b64(place_map)
            raw_rgb = batch.get("raw_rgb")
            if raw_rgb is not None:
                base = raw_rgb[0].detach().cpu().numpy().astype(np.uint8)
                result["overlay_png"] = make_overlay_b64(base, [pick_map, place_map])
        return result

    def _predict_dry(self, mask, want_heatmap, n_arms=1):
        """dry-run：返回 mask 内的随机点，用于验证通信链路（不加载权重）。"""
        mask = np.asarray(mask)
        if mask.shape[:2] != (self.image_size, self.image_size):
            mask = np.asarray(
                Image.fromarray(mask.astype(np.float32)).resize(
                    (self.image_size, self.image_size), Image.NEAREST
                ),
                dtype=np.float32,
            )
        indices = np.argwhere(mask > 0)
        if len(indices) == 0:
            indices = np.array([[self.image_size // 2, self.image_size // 2]])
        size = self.image_size
        rows = np.linspace(0, len(indices) - 1, max(n_arms, 1)).astype(int)
        picks, places = [], []
        for i in rows:
            row, col = indices[int(i)]
            picks.append([float(col), float(row)])
            places.append([float(min(col + 20, size - 1)), float(min(row + 20, size - 1))])
        return {
            "n_arms": int(n_arms),
            "picks": picks,
            "places": places,
            "pick_confidences": [0.0] * len(picks),
            "place_confidences": [0.0] * len(places),
            "pick_peak": 0.0,
            "place_peak": 0.0,
            "pick_heatmap_png": heatmap_to_b64(np.zeros((size, size), np.float32))
            if want_heatmap else None,
            "place_heatmap_png": heatmap_to_b64(np.zeros((size, size), np.float32))
            if want_heatmap else None,
            "overlay_png": None,
        }

    def warmup(self):
        if self.dry_run:
            return
        size = self.image_size
        rgb = np.full((size, size, 3), 77, dtype=np.uint8)
        depth = build_placeholder_depth(size, size)
        self.predict(rgb, depth, get_mask_from_depth(depth), "warmup", [], False)
        print("Warmup done.")


# --------------------------------------------------------------------------- #
# 会话 / 任务
# --------------------------------------------------------------------------- #
class Session:
    """一次折叠任务：skill 的分步指令 + 已执行的步数 + 历史上下文。"""

    def __init__(self, session_id, skill_name, skill, instructions, context_length, engine,
                 plan=None):
        self.id = session_id
        self.skill_name = skill_name
        self.skill = skill
        self.instructions = instructions
        self.plan = plan  # 技能分步计划（含每步的关键点数，用于判断本步用几条臂）
        self.step_index = 0
        self.context = []
        self.context_length = context_length
        self.engine = engine
        self.routing = {"source": "client"}  # 由 ServerState.create_session 填充
        self.created_at = time.time()
        self.updated_at = time.time()
        self.lock = threading.Lock()

    @property
    def total_steps(self):
        return len(self.instructions)

    @property
    def done(self):
        return self.step_index >= self.total_steps

    def info(self):
        return {
            "session_id": self.id,
            "skill": self.skill_name,
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "done": self.done,
            "steps": self.instructions,
            "instruction": self.instructions[self.step_index] if not self.done else None,
            "routing": self.routing,
        }


class ServerState:
    def __init__(self, engine, executor, default_skill=None, ollama_url=None, llm_model=None,
                 per_skill_weights=False, prefer_arms="auto"):
        self.engine = engine
        self.executor = executor
        self.default_skill = default_skill
        self.prefer_arms = prefer_arms
        self.ollama_url = ollama_url
        self.llm_model = llm_model
        self.per_skill_weights = per_skill_weights
        self.engines = {engine.checkpoint_path: engine}
        self.sessions = {}
        self.jobs = {}
        self.lock = threading.Lock()

    def engine_for(self, skill_name):
        """取该技能要用的引擎：默认全局权重，或按 SKILL.md 声明的权重懒加载。"""
        if not self.per_skill_weights:
            return self.engine
        config_path, checkpoint_path = get_model_paths(skill_name)
        if not config_path or not checkpoint_path:
            return self.engine
        with self.lock:
            engine = self.engines.get(checkpoint_path)
        if engine is None:
            engine = FoldEngine(
                resolve_config(checkpoint_path, config_path),
                checkpoint_path,
                str(self.engine.device),
            )
            with self.lock:
                self.engines[checkpoint_path] = engine
        return engine

    @property
    def allowed_skills(self):
        """当前服务端可用的技能名。

        模型只有一套共享的 pick/place 热力图头（docs/decoding_design_notes.md），
        单臂步取 1 个峰、双臂步取 top-2 峰，因此同一份权重既能跑 unimanual-*
        也能跑 bimanual-*。
        """
        return sorted(SKILL_SPECS)

    # ---- 会话 ------------------------------------------------------------
    def create_session(self, skill_name=None, instruction=None, random_angle=0.0):
        routing = {"source": "client", "instruction": instruction}
        if not skill_name:
            if instruction:
                skill_name, routing = route_skill_with_llm(
                    instruction, self.ollama_url, self.llm_model,
                    available_skills=self.allowed_skills,
                    prefer_arms=self.prefer_arms,
                )
            elif self.default_skill:
                skill_name = self.default_skill
                routing = {"source": "server_default", "skill": skill_name}
            else:
                raise ValueError("必须提供 skill 或 instruction（或启动时用 --skill 指定默认技能）")
        if skill_name not in SKILLS:
            raise KeyError(f"未知技能 {skill_name}，可选：{sorted(SKILLS)}")

        engine = self.engine_for(skill_name)
        spec = SKILL_SPECS[skill_name]
        # 单/双臂共享同一套热力图头，双臂步用 top-2 峰 + 就近配对解码
        if spec.arms == "bimanual":
            routing.setdefault("note", "双臂技能使用共享热力图头解码（top-2 峰 + 就近配对）")
        skill = SKILLS[skill_name]()
        plan = skill.get_plan(random_angle=float(random_angle))
        session = Session(
            session_id=uuid.uuid4().hex[:12],
            skill_name=skill_name,
            skill=skill,
            instructions=[str(i) for i in plan["instructions"]],
            context_length=engine.max_context_length,
            engine=engine,
            plan=plan,
        )
        session.routing = routing
        with self.lock:
            self._gc_locked()
            self.sessions[session.id] = session
        return session

    def get_session(self, session_id):
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"会话不存在或已释放: {session_id}")
        return session

    def delete_session(self, session_id):
        with self.lock:
            return self.sessions.pop(session_id, None) is not None

    def _gc_locked(self):
        now = time.time()
        for sid in [k for k, v in self.sessions.items() if now - v.updated_at > SESSION_TTL]:
            self.sessions.pop(sid, None)
        for jid in [k for k, v in self.jobs.items() if now - v["created_at"] > JOB_TTL]:
            self.jobs.pop(jid, None)

    # ---- 任务 ------------------------------------------------------------
    def submit_step(self, session, payload):
        job_id = uuid.uuid4().hex[:12]
        future = self.executor.submit(self._run_step, session, payload)
        with self.lock:
            self.jobs[job_id] = {
                "future": future,
                "created_at": time.time(),
                "session_id": session.id,
            }
        return job_id

    def _run_step(self, session, payload):
        """在推理线程里执行：观测 -> 模型 -> 坐标（保持与 skill 一致的上下文逻辑）。"""
        started = time.time()
        engine = session.engine
        try:
            with session.lock:
                if session.done:
                    raise RuntimeError(
                        f"会话已完成全部 {session.total_steps} 步，不能再推理"
                    )
                instruction = session.instructions[session.step_index]
                step_index = session.step_index
                # 本步用几条臂：由技能步骤定义给出（不是模型预测）
                n_arms = int(session.skill._n_arms(session.plan, step_idx=step_index))

            rgb, depth, mask, meta = decode_observation(payload, engine.image_size)
            prediction = engine.predict(
                rgb=rgb,
                depth=depth,
                mask=mask,
                instruction=instruction,
                context=list(session.context),
                want_heatmap=bool(payload.get("return_heatmap", True)),
                n_arms=n_arms,
            )

            with session.lock:
                # 与 skill 执行顺序一致：先把"执行前"的观测存入历史，再前进到下一步
                session.context.append(
                    {"rgb": rgb.copy(), "depth": depth.copy(), "mask": mask.copy()}
                )
                if len(session.context) > session.context_length:
                    session.context = session.context[-session.context_length:]
                session.step_index = step_index + 1
                session.updated_at = time.time()
                done = session.done
                total_steps = session.total_steps

            height, width = rgb.shape[:2]
            crop = meta.get("crop", [0, 0])
            result = {
                "status": "done",
                "session_id": session.id,
                "skill": session.skill_name,
                "step_index": step_index,
                "step_number": step_index + 1,
                "total_steps": total_steps,
                "instruction": instruction,
                "done": done,
                "n_arms": int(prediction["n_arms"]),
                "pick": make_point(
                    prediction["picks"][0],
                    height,
                    width,
                    engine.image_size,
                    prediction["pick_confidences"][0],
                    prediction["pick_peak"],
                    crop,
                ),
                "place": make_point(
                    prediction["places"][0],
                    height,
                    width,
                    engine.image_size,
                    prediction["place_confidences"][0],
                    prediction["place_peak"],
                    crop,
                ),
                "arms": {
                    "left": {
                        "pick": make_point(
                            prediction["picks"][0], height, width, engine.image_size,
                            prediction["pick_confidences"][0], prediction["pick_peak"], crop,
                        ),
                        "place": make_point(
                            prediction["places"][0], height, width, engine.image_size,
                            prediction["place_confidences"][0], prediction["place_peak"], crop,
                        ),
                    },
                    "right": {
                        "pick": make_point(
                            prediction["picks"][1], height, width, engine.image_size,
                            prediction["pick_confidences"][1], prediction["pick_peak"], crop,
                        ),
                        "place": make_point(
                            prediction["places"][1], height, width, engine.image_size,
                            prediction["place_confidences"][1], prediction["place_peak"], crop,
                        ),
                    },
                } if prediction["n_arms"] >= 2 else None,
                "geometry": {
                    "sent_size": meta.get("sent_size", [height, width]),
                    "crop": crop,
                    "sent_height": int(height),
                    "sent_width": int(width),
                    "model_input_size": int(engine.image_size),
                    **geometry_of(height, width, engine.image_size),
                },
                "observation": meta,
                "routing": session.routing,
                "latency_ms": int((time.time() - started) * 1000),
            }
            if prediction.get("pick_heatmap_png"):
                result["heatmaps"] = {
                    "pick_png": prediction["pick_heatmap_png"],
                    "place_png": prediction["place_heatmap_png"],
                    "overlay_png": prediction.get("overlay_png"),
                }
            return result
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    def job_status(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(f"任务不存在或已过期: {job_id}")
        future = job["future"]
        if not future.done():
            return {"status": "running", "job_id": job_id, "session_id": job["session_id"]}
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        result = dict(result)
        result["job_id"] = job_id
        result.setdefault("session_id", job["session_id"])
        return result

    def wait_job(self, job_id, timeout):
        deadline = time.time() + timeout
        while True:
            result = self.job_status(job_id)
            if result.get("status") in ("done", "error"):
                return result
            if time.time() >= deadline:
                return {"status": "running", "job_id": job_id, "note": "timeout"}
            time.sleep(0.05)


def make_point(xy, height, width, image_size, confidence, peak, crop=None):
    """把一个点整理成多种坐标系的表示，方便客户端按需要取用。

    height/width 是送入模型前（已中心裁剪成正方形）的图像尺寸；
    crop 是该正方形在客户端所发原图中的左上角偏移。
    """
    x, y = float(xy[0]), float(xy[1])
    crop = list(crop) if crop else [0, 0]
    x_crop, y_crop = model_xy_to_image_xy([x, y], height, width, image_size)
    return {
        "xy_model": [x, y],                                   # 模型输入坐标系 (224x224)
        "xy_image": [x_crop + crop[0], y_crop + crop[1]],     # 客户端所发原图的像素坐标
        "uv": [x / float(image_size), y / float(image_size)],  # 归一化（相对模型输入方形）
        "confidence": float(confidence),
        "peak": float(peak),
    }


def decode_observation(payload, image_size):
    """请求体 -> (rgb, depth, mask, meta)。

    客户端图像允许非正方形：自动中心裁剪成正方形（模型的 Processor 要求正方形），
    裁剪偏移记录在 meta["crop"]，返回给客户端的 xy_image 已含该偏移。
    """
    if "rgb" not in payload:
        raise ValueError("请求体缺少 'rgb'（base64 编码的 PNG/JPEG）")
    rgb = b64_to_array(payload["rgb"])
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    rgb = rgb[:, :, :3].astype(np.uint8)
    sent_h, sent_w = rgb.shape[:2]

    # 中心裁剪到正方形（短边）
    side = min(sent_h, sent_w)
    crop_y = (sent_h - side) // 2
    crop_x = (sent_w - side) // 2
    if (sent_h, sent_w) != (side, side):
        rgb = rgb[crop_y:crop_y + side, crop_x:crop_x + side]
    height, width = rgb.shape[:2]

    meta = {
        "depth_source": "client",
        "mask_source": "client",
        "sent_size": [int(sent_h), int(sent_w)],
        "crop": [int(crop_x), int(crop_y)],  # 裁剪正方形在发送图中的左上角
        "rgb_size": [int(height), int(width)],
    }

    if payload.get("depth"):
        depth = convert_depth(
            b64_to_array(payload["depth"]),
            mode=payload.get("depth_mode", "normalized"),
            depth_min=float(payload.get("depth_min", 0.4)),
            depth_max=float(payload.get("depth_max", 1.2)),
        )
        if depth.shape[:2] != (height, width):
            depth = np.asarray(
                Image.fromarray(depth).resize((width, height), Image.BILINEAR),
                dtype=np.float32,
            )
        # 注意：仅用于送入模型；客户端拿坐标查深度时仍用自己的原始深度图，
        # meta["crop"] 始终是相对客户端所发 RGB 图的偏移
    else:
        depth = build_placeholder_depth(height, width, float(payload.get("depth_placeholder", 0.7)))
        meta["depth_source"] = "placeholder"

    if payload.get("mask"):
        mask = b64_to_array(payload["mask"]).astype(np.float32)
        mask = (mask > 0).astype(np.float32)
        if mask.shape[:2] != (height, width):
            mask = np.asarray(
                Image.fromarray(mask).resize((width, height), Image.NEAREST),
                dtype=np.float32,
            )
    else:
        mask = get_mask_from_depth(depth)
        meta["mask_source"] = "from_depth"

    return rgb, depth, mask, meta


def route_skill_with_llm(instruction, ollama_url, llm_model, available_skills=None,
                         prefer_arms="auto", timeout=180):
    """用本地 Ollama 文本大模型把一句总指令路由到某个技能。

    只把当前权重可用的技能列进候选（避免选中不可用技能）；
    prefer_arms: auto / bimanual / unimanual —— 指令没明说时用哪类技能兜底。
    返回 (skill_name, routing_info)，routing_info 含大模型的思考过程/理由/原始响应。
    """
    import re

    import requests

    if not ollama_url:
        raise RuntimeError("未提供 skill 且未配置 --ollama-url，无法做技能路由")
    available = list(available_skills) if available_skills else sorted(SKILL_DESCRIPTIONS)
    skill_lines = "\n".join(
        f"- {name}: {SKILL_DESCRIPTIONS[name]}" for name in available
    )
    prefer_rules = {
        "bimanual": (
            "If the instruction is unclear, PREFER a bimanual-* skill "
            "(two arms can cooperate on T-shirts, trousers and square cloth)."
        ),
        "unimanual": (
            "If the instruction is unclear, PREFER a unimanual-* skill."
        ),
        "auto": (
            "If the instruction is unclear, prefer unimanual-* skills "
            "(simpler and more robust)."
        ),
    }
    prompt = (
        "You are a robot folding task planner. The robot has learned several "
        "folding skills. Each skill is a complete multi-step routine.\n"
        "IMPORTANT: ONLY the following skills are valid candidates; choosing any "
        "other skill is invalid:\n"
        f"{skill_lines}\n\n"
        "Rules about the number of arms:\n"
        "1. If the instruction mentions both arms / two arms / both hands / two hands "
        "/ dual arm / bimanual, you MUST choose a skill whose name starts with "
        '"bimanual-".\n'
        "2. If it says one arm / single arm / a single robot arm, you MUST choose a "
        'skill whose name starts with "unimanual-".\n'
        f"3. {prefer_rules.get(prefer_arms, prefer_rules['auto'])}\n\n"
        f'User instruction: "{instruction}"\n\n'
        "You MUST choose exactly one skill from the list above; never answer that you "
        "cannot decide. Reply ONLY with a JSON object like "
        '{"skill": "<skill name>", "reason": "<short reason>"}.\n'
    )
    started = time.time()
    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/chat",
        json={
            "model": llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    raw = response.json()["message"]["content"]

    # 思考过程：qwen3 等思考模型会把推理放在 <think>...</think> 里
    thinking = ""
    match = re.search(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
    if match:
        thinking = match.group(1).strip()
    content = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
    if data.get("skill") not in available:
        raise RuntimeError(
            f"大模型返回的技能不可用: {data.get('skill')!r}，可用技能：{available}"
        )

    routing_info = {
        "source": "llm",
        "model": llm_model,
        "instruction": instruction,
        "thinking": thinking,                       # 大模型的推理/思考过程
        "reason": str(data.get("reason", "")),      # 简短选择理由（JSON 字段）
        "raw_response": raw[:4000],                 # 原始响应（超长截断）
        "latency_sec": round(time.time() - started, 2),
    }
    return data["skill"], routing_info


# --------------------------------------------------------------------------- #
# HTTP 层
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "FoldInferenceServer/1.0"

    @property
    def state(self) -> ServerState:
        return self.server.state  # type: ignore[attr-defined]

    # ---- 工具 ------------------------------------------------------------
    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def log_message(self, fmt, *args):  # 简化访问日志
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ---- 路由 ------------------------------------------------------------
    def do_GET(self):  # noqa: N802
        parts = urlparse(self.path).path.strip("/").split("/")
        try:
            if parts == ["v1", "health"]:
                engine = self.state.engine
                self._send(200, {
                    "status": "ok",
                    "device": str(engine.device),
                    "model_input_size": engine.image_size,
                    "max_context_length": engine.max_context_length,
                    "checkpoint": engine.checkpoint_path,
                    "config": engine.config_path,
                    "epoch": engine.epoch,
                    "dry_run": bool(engine.dry_run),
                    "allowed_skills": self.state.allowed_skills,
                    "default_skill": self.state.default_skill,
                    "sessions": len(self.state.sessions),
                    "jobs": len(self.state.jobs),
                })
            elif parts == ["v1", "skills"]:
                self._send(200, {"skills": [
                    {
                        "name": name,
                        "arms": spec.arms,
                        "cloth_type": spec.cloth_type,
                        "num_steps": spec.metadata.get("num_steps"),
                        "description": spec.description,
                        "default_checkpoint": spec.checkpoint,
                    }
                    for name, spec in SKILL_SPECS.items()
                ]})
            elif parts[:2] == ["v1", "sessions"] and len(parts) == 3:
                self._send(200, self.state.get_session(parts[2]).info())
            elif parts[:2] == ["v1", "jobs"] and len(parts) == 3:
                self._send(200, self.state.job_status(parts[2]))
            else:
                self._send(404, {"error": f"unknown route: {self.path}"})
        except Exception as exc:  # noqa: BLE001
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})

    def do_DELETE(self):  # noqa: N802
        parts = urlparse(self.path).path.strip("/").split("/")
        try:
            if parts[:2] == ["v1", "sessions"] and len(parts) == 3:
                deleted = self.state.delete_session(parts[2])
                self._send(200, {"deleted": deleted, "session_id": parts[2]})
            else:
                self._send(404, {"error": f"unknown route: {self.path}"})
        except Exception as exc:  # noqa: BLE001
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):  # noqa: N802
        parts = urlparse(self.path).path.strip("/").split("/")
        try:
            payload = self._read_json()
            if parts == ["v1", "sessions"]:
                session = self.state.create_session(
                    skill_name=payload.get("skill"),
                    instruction=payload.get("instruction"),
                    random_angle=payload.get("random_angle", 0.0),
                )
                self._send(201, session.info())
            elif parts[:2] == ["v1", "sessions"] and len(parts) == 4 and parts[3] == "step":
                session = self.state.get_session(parts[2])
                job_id = self.state.submit_step(session, payload)
                callback = payload.get("callback_url")
                if callback:
                    threading.Thread(
                        target=self._notify, args=(job_id, callback), daemon=True
                    ).start()
                self._send(202, {
                    "status": "queued",
                    "job_id": job_id,
                    "session_id": session.id,
                    "result_url": f"/v1/jobs/{job_id}",
                })
            elif parts[:2] == ["v1", "sessions"] and len(parts) == 4 and parts[3] == "step_sync":
                session = self.state.get_session(parts[2])
                job_id = self.state.submit_step(session, payload)
                result = self.state.wait_job(job_id, float(payload.get("timeout", 120)))
                self._send(200 if result.get("status") == "done" else 202, result)
            else:
                self._send(404, {"error": f"unknown route: {self.path}"})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})

    def _notify(self, job_id, callback_url):
        """异步任务完成后回调机械臂服务器（可选）。"""
        import requests

        try:
            result = self.state.wait_job(job_id, timeout=600)
            requests.post(callback_url, json=result, timeout=30)
        except Exception as exc:  # noqa: BLE001
            print(f"[callback] failed for job {job_id}: {exc!r}")


class FoldServer(ThreadingHTTPServer):
    """多线程服务：守护线程 + 更大的握手队列，避免客户端重试时连接被丢弃。"""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128


# --------------------------------------------------------------------------- #
# 启动
# --------------------------------------------------------------------------- #
def resolve_config(checkpoint_path, config_path=None):
    """未显式给 config 时，用 checkpoint 所在 run 目录的 config.yaml。"""
    if config_path:
        return config_path
    candidate = Path(checkpoint_path).resolve().parent.parent / "config.yaml"
    if not candidate.exists():
        raise FileNotFoundError(f"找不到 config.yaml: {candidate}，请用 --config 指定")
    return str(candidate)


def main():
    parser = argparse.ArgumentParser(description="折叠技能异步推理服务（供真实机械臂调用）")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="BiFold 权重路径（默认：single_sequential num_scales=8 的 last.pth）")
    parser.add_argument("--config", default=None, help="对应的 config.yaml（默认自动推断）")
    parser.add_argument("--skill", default=None,
                        help="兜底默认技能名（仅当客户端建会话时既没传 skill 也没传 instruction 时使用；"
                             "通常省略，让客户端每次自行选择要折什么）")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL,
                        help="用一句话指令自动选技能时的 Ollama 地址（默认本机 11434；传空串可禁用）")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL,
                        help="路由用的文本大模型（与 inference_skill_softgym.py 默认一致）")
    parser.add_argument("--prefer-arms", default="auto", choices=["auto", "bimanual", "unimanual"],
                        help="自然语言路由时，指令没说明几只手臂就用哪类技能兜底"
                             "（双臂服务建议 bimanual，单臂服务建议 unimanual）")
    parser.add_argument("--per-skill-weights", action="store_true",
                        help="每个技能用它自己 SKILL.md 里声明的权重（默认全部用 --checkpoint）")
    parser.add_argument("--workers", type=int, default=1, help="推理线程数（GPU 建议 1）")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="不加载模型，只验证通信链路")
    args = parser.parse_args()

    if args.dry_run:
        engine = FoldEngine.dry()
    else:
        config_path = resolve_config(args.checkpoint, args.config)
        engine = FoldEngine(config_path, args.checkpoint, args.device)
        if not args.no_warmup:
            engine.warmup()

    executor = ThreadPoolExecutor(max_workers=max(1, args.workers))
    state = ServerState(
        engine=engine,
        executor=executor,
        default_skill=args.skill,
        ollama_url=args.ollama_url,
        llm_model=args.llm_model,
        per_skill_weights=args.per_skill_weights,
        prefer_arms=args.prefer_arms,
    )
    print(f"Allowed skills for this checkpoint: {state.allowed_skills}")

    server = FoldServer((args.host, args.port), Handler)
    server.state = state  # type: ignore[attr-defined]
    print(f"Serving on http://{args.host}:{args.port}/v1  (skills: {len(SKILL_SPECS)})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down ...")
    finally:
        server.server_close()
        executor.shutdown(wait=False)


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
