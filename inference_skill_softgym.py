#!/usr/bin/env python3
"""Closed-loop skill inference (simulation or real robot).

Pipeline:
  1. Load the trained model declared by the selected skill's SKILL.md.
  2. The user gives one natural-language instruction ("Fold the T-shirt.").
  3. An LLM (local Ollama, default qwen3.6:35b) routes it to one of the 9 skills.
  4. The skill runs all of its folding steps in a loop:
     observe -> step instruction -> predict pick/place -> execute -> re-observe,
     until every step is done. Step instructions come from the Demonstrator
     templates in msufold/env/softgym_demonstrators.py (unimanual) and
     msufold/env/softgym_dual_demonstrators.py (bimanual).
  5. The whole rollout is recorded as an mp4 (gif if no encoder is available).

Examples:
    python inference_skill_softgym.py --instruction "Fold the T-shirt." --num-evals 1
    python inference_skill_softgym.py --skill bimanual-tshirt-fold --num-evals 1
    python inference_skill_softgym.py --instruction "Fold the T-shirt." \
        --ollama-url http://localhost:11434 --llm-model qwen3.6:35b
"""

import argparse
import json
import logging
import os
import pickle
import random
from pathlib import Path
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import trange

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from msufold.data.processor import Processor  # noqa: E402
from msufold.data.utils import get_mask_from_depth  # noqa: E402
from msufold.env.softgym_cloth_env import SoftgymClothEnv  # noqa: E402
from msufold.models import Models  # noqa: E402
from skills import SKILLS, SKILL_DESCRIPTIONS, SKILL_SPECS, get_model_paths  # noqa: E402
from skills.backends import RealRobotBackend, SoftgymBackend  # noqa: E402
from skills.base_skill import Observation  # noqa: E402
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "qwen3.6:35b"
#: 专门的 skill 评估输出目录：仿真视频 + run.log + llm_router.log/.jsonl
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "outputs" / "skill_eval")
#: 大模型必须给出技能，最多重试几次（不再有关键词兜底）
LLM_MAX_ATTEMPTS = 3

LOGGER = None  # 由 setup_logging 初始化


# --------------------------------------------------------------------------- #
# 日志（大模型调用全过程落盘，方便失败后排查）
# --------------------------------------------------------------------------- #
def setup_logging(output_dir, quiet=False):
    """初始化日志：同时输出到终端和 <output_dir>/run.log（追加模式）。"""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")

    logger = logging.getLogger("skill_inference")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.WARNING if quiet else logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("=" * 80)
    logger.info(f"New inference run started, log file: {log_path}")
    return logger, log_path


def write_llm_record(llm_log_path, record):
    """Record one LLM call.

    Two files are written under the output directory:
      * llm_router.jsonl - one JSON line per call (for post-processing);
      * llm_router.log   - human readable log with the full prompt, the raw
                           response, the parsed skill and any error.
    """
    os.makedirs(os.path.dirname(llm_log_path), exist_ok=True)
    record = {"time": datetime.now().isoformat(timespec="seconds"), **record}
    with open(llm_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    readable_path = os.path.splitext(llm_log_path)[0] + ".log"
    with open(readable_path, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"[{record['time']}] event={record.get('event')}\n")
        f.write(f"model: {record.get('model')}\n")
        f.write(f"endpoint: {record.get('endpoint')}\n")
        f.write(f"user_instruction: {record.get('instruction')}\n")
        if record.get("prompt"):
            f.write("prompt:\n" + record["prompt"] + "\n")
        f.write(f"http_status: {record.get('http_status')}\n")
        f.write(f"latency_sec: {record.get('latency_sec')}\n")
        f.write("raw_response:\n" + str(record.get("raw_response")) + "\n")
        f.write(f"parsed_skill: {record.get('parsed_skill')}\n")
        f.write(f"note: {record.get('note')}\n")
        f.write(f"error: {record.get('error')}\n")
    return record


def log_exception(context, exc=None):
    """把异常堆栈写入日志，方便定位失败原因。"""
    if LOGGER is None:
        return
    LOGGER.error(f"{context}: {exc!r}")
    LOGGER.error("异常堆栈:\n" + (traceback.format_exc() if exc is None else "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )))


# --------------------------------------------------------------------------- #
# 大模型 skill 路由
# --------------------------------------------------------------------------- #
def build_skill_selection_prompt(instruction: str, hint: str = None) -> str:
    """构造路由 prompt：把技能清单 + 用户指令给文本大模型，让它必须选出一个 skill。"""
    skill_lines = "\n".join(f"- {name}: {desc}" for name, desc in SKILL_DESCRIPTIONS.items())
    hint_block = f"{hint}\n\n" if hint else ""
    return (
        "You are a robot folding task planner. The robot has learned the following "
        "folding skills. Each skill is a complete multi-step routine.\n"
        f"{skill_lines}\n\n"
        f'User instruction: "{instruction}"\n\n'
        "You MUST choose exactly one skill from the list above; never answer that you "
        "cannot decide. Reply ONLY with a JSON object like "
        '{"skill": "<skill name>", "reason": "<short reason>"}.\n\n'
        f"{hint_block}"
    )


def _extract_json(text: str):
    """从大模型返回文本中提取 JSON（兼容 <think>...</think> 等推理输出）。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def select_skill_with_llm(instruction, ollama_url, llm_model, llm_log_path,
                           timeout=180, verbose=True, hint=None):
    """调用本地 Ollama 文本大模型选择 skill；失败/解析不出合法技能时返回 None。

    请求参数、原始返回、解析结果和异常堆栈都会写入 run.log 与 llm_router.log。
    """
    import requests

    logger = LOGGER
    prompt = build_skill_selection_prompt(instruction, hint=hint)
    endpoint = f"{ollama_url.rstrip('/')}/api/chat"
    payload = {
        "model": llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }

    if logger is not None:
        logger.info("[LLM] Calling LLM for skill routing")
        logger.info(f"[LLM] endpoint: {endpoint}")
        logger.info(f"[LLM] model: {llm_model}, timeout: {timeout}s")
        logger.info(f"[LLM] User instruction: {instruction}")
        logger.debug(f"[LLM] Full prompt:\n{prompt}")

    start = time.time()
    content = None
    status_code = None
    error = None
    try:
        response = requests.post(endpoint, json=payload, timeout=timeout)
        status_code = response.status_code
        response.raise_for_status()
        content = response.json()["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        log_exception("[LLM] Ollama call failed", exc)
        if verbose:
            print(f"[LLM] Ollama call failed ({exc}), will retry")
    latency = round(time.time() - start, 2)

    if logger is not None:
        logger.info(f"[LLM] HTTP status: {status_code}, latency: {latency}s")
        if content is not None:
            logger.info(f"[LLM] Raw response:\n{content}")

    skill_name = None
    parse_note = None
    if content is not None:
        data = _extract_json(content)
        if data is None:
            parse_note = "Response is not valid JSON (tried stripping <think> and regex extraction)"
        elif data.get("skill") not in SKILLS:
            parse_note = f"Parsed skill is not in the candidate list {sorted(SKILLS)}: {data.get('skill')!r}"
        else:
            skill_name = data["skill"]
            parse_note = f"reason: {data.get('reason', '')}"
            if verbose:
                print(f"[LLM] LLM selected skill: {skill_name}（{parse_note}）")
        if logger is not None:
            if skill_name is None:
                logger.warning(f"[LLM] Parse failed: {parse_note}")
            else:
                logger.info(f"[LLM] LLM selected skill: {skill_name}（{parse_note}）")

    write_llm_record(
        llm_log_path,
        {
            "event": "skill_routing",
            "endpoint": endpoint,
            "model": llm_model,
            "timeout": timeout,
            "instruction": instruction,
            "prompt": prompt,
            "http_status": status_code,
            "latency_sec": latency,
            "raw_response": content,
            "parsed_skill": skill_name,
            "note": parse_note,
            "error": error,
        },
    )
    return skill_name


def select_skill(instruction, args, llm_log_path):
    """Routing: one instruction -> the LLM must pick exactly one skill.

    There is no keyword fallback: if the LLM is unreachable or returns an
    invalid skill, we retry (up to LLM_MAX_ATTEMPTS) and finally raise, so the
    robot never executes a skill that the model did not choose.
    """
    logger = LOGGER
    if args.skill:
        assert args.skill in SKILLS, f"Unknown skill: {args.skill}, candidates: {list(SKILLS)}"
        print(f"[Router] Skill specified on the command line: {args.skill}")
        if logger is not None:
            logger.info(f"[Router] Skill specified via command line: {args.skill} (LLM not called)")
        write_llm_record(
            llm_log_path,
            {
                "event": "skill_routing",
                "instruction": instruction,
                "parsed_skill": args.skill,
                "note": "Specified by --skill, LLM not called",
            },
        )
        return args.skill, {"source": "manual"}

    hint = None
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        skill_name = select_skill_with_llm(
            instruction,
            args.ollama_url,
            args.llm_model,
            llm_log_path=llm_log_path,
            verbose=not args.quiet,
            hint=hint,
        )
        if skill_name is not None:
            if logger is not None:
                logger.info(f"[Router] Final skill: {skill_name} (source: llm, attempt {attempt})")
            return skill_name, {"source": "llm"}
        hint = (
            "Your previous reply was invalid. Reply ONLY with a JSON object "
            "{\"skill\": \"<one of: " + ", ".join(sorted(SKILL_DESCRIPTIONS)) + ">\"}."
        )
        if logger is not None:
            logger.warning(f"[Router] LLM did not return a valid skill, retrying ({attempt}/{LLM_MAX_ATTEMPTS})")

    message = (
        f"LLM failed to select a valid skill after {LLM_MAX_ATTEMPTS} attempts. "
        "Check that Ollama is running (--ollama-url) and the model (--llm-model) is available."
    )
    if logger is not None:
        logger.error(f"[Router] {message}")
    raise RuntimeError(message)


# --------------------------------------------------------------------------- #
# 模型 / 处理器加载
# --------------------------------------------------------------------------- #
def load_model_and_processor(config_path, checkpoint_path, device):
    print(f"Loading config: {config_path}")
    cfg = OmegaConf.load(config_path)
    # 训练时 Models.get_by_name 会删除 model.name，保存的 config.yaml 中缺失，这里补回
    cfg.model.name = "siglip_sequential"

    print("Building Processor ...")
    processor = Processor(
        cfg=cfg.processor,
        partition="test",
        max_context_length=cfg.train_dataset.max_context_length,
        autoprocessor_name=cfg.model.automodel_name,
    )

    print("Building model ...")
    model = Models.get_by_name(cfg.model, device=device).to(device)

    print(f"Loading weights: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as exc:
        print(f"Strict load failed ({exc}), retrying with strict=False")
        log_exception("Strict load of model weights failed", exc)
        model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    print(f"  - Trained epochs: {checkpoint.get('epoch', 'unknown')}")

    return model, processor, cfg


# --------------------------------------------------------------------------- #
# softgym 场景与视频
# --------------------------------------------------------------------------- #
def load_init_state(path):
    """加载准备好的初始布料状态（目录或目录里的 rgb.png）。

    返回 {dir, meta, config, state, cloth3d}；缺少 scene.pkl 时报错，
    因为仿真必须能精确复现这一帧初始状态。
    """
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    case_dir = path if path.is_dir() else path.parent
    scene_file = case_dir / "scene.pkl"
    meta_file = case_dir / "meta.json"
    if not scene_file.exists() or not meta_file.exists():
        raise FileNotFoundError(
            f"{case_dir} 不是有效的初始状态目录（需要 scene.pkl + meta.json），"
            "请先用 prepare_skill_states.py 生成"
        )
    with open(scene_file, "rb") as f:
        scene = pickle.load(f)
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "dir": str(case_dir),
        "meta": meta,
        "config": scene["config"],
        "state": scene["state"],
        "cloth3d": scene.get("cloth3d", False),
    }


def sample_random_angle(skill, rng):
    """与 SoftgymSingleEvaluator 一致的角度采样。"""
    if skill.name == "StraightFold":
        return rng.uniform(-80, 80)
    if skill.cloth3d:
        return rng.uniform(-40, 40)
    return rng.uniform(0, 40)


def load_softgym_cache(cfg, cloth_type):
    cache_path = os.path.join(cfg.softgym_cache, cloth_type + ".pkl")
    print(f"Loading softgym cache: {cache_path}")
    with open(cache_path, "rb") as f:
        config_data = pickle.load(f)
    return config_data["configs"], config_data["states"]


def save_video(frames, path, fps=30):
    """优先保存 mp4，编码器不可用时回退 gif。"""
    if not frames:
        return None
    try:
        with imageio.get_writer(
            path, fps=fps, codec="libx264", quality=8, macro_block_size=1
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
        return path
    except Exception as exc:  # noqa: BLE001
        print(f"[Video] Failed to save mp4 ({exc}), falling back to gif")
        log_exception(f"[Video] Failed to save mp4: {path}", exc)
        gif_path = str(path).rsplit(".", 1)[0] + ".gif"
        with imageio.get_writer(gif_path, mode="I", fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        return gif_path


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_trial(skill, backend, model, processor, device, args, trial_index, out_dir, rng,
              scene=None, init_state=None):
    """跑一个回合：复位 -> 采集初始观测 -> skill 闭环执行所有步骤。

    scene/init_state 只有仿真后端需要（初始状态 config/state）；真机后端 reset 只回拍照位。
    给定 init_state 时完全复现该状态（不再随机），否则按随机参数采样。
    """
    logger = LOGGER
    if init_state is not None:
        scene = {"config": init_state["config"], "state": init_state["state"]}
        meta = init_state["meta"]
        randomize = False
        random_angle = float(meta["angle_deg"])
        size_scale = float(meta["size_scale"])
        offset = np.asarray(meta["offset"], dtype=float)
    else:
        # 仿真里的布料随机化：随机角度 + 随机大小 + 随机摆位（真机后端不使用）
        randomize = scene is not None and not args.no_randomize
        random_angle = sample_random_angle(skill, rng) if randomize else 0.0
        size_scale = float(rng.uniform(*args.size_range)) if randomize else 1.0
        offset = rng.uniform(-args.offset_range, args.offset_range, size=2) if randomize else None

    if logger is not None:
        logger.info(
            f"[Trial {trial_index}] skill={skill.name}, backend={backend.name}, "
            f"cloth={skill.cloth_type}, {skill.num_steps} steps"
        )
        if randomize:
            logger.info(
                f"[Trial {trial_index}] cloth randomization: angle={random_angle:.2f} deg, "
                f"size_scale={size_scale:.3f}, offset={np.round(offset, 4).tolist()}"
            )
        if init_state is not None:
            logger.info(f"[Trial {trial_index}] using prepared initial state: {init_state['dir']}")

    skill.prepare_env(
        backend,
        random_angle=random_angle,
        size_scale=size_scale,
        translate=offset,
        **(scene or {}),
    )

    trial_name = f"trial{trial_index}"
    # skill 内部循环执行全部折叠步骤，每步执行后重新采集观测作为下一步输入
    final_obs, _context, step_instructions = skill.execute(
        backend=backend,
        model=model,
        processor=processor,
        device=device,
        context=[],
        random_angle=random_angle,
        log_dir=out_dir,
        trial_name=trial_name,
        verbose=not args.quiet,
    )

    # 保存整个 rollout 视频（仅仿真后端收集了全程帧）
    env = getattr(backend, "env", None)
    video_path = None
    if env is not None and getattr(env, "frames", None):
        video_path = save_video(
            env.frames, os.path.join(out_dir, f"{trial_name}_rollout.mp4")
        )

    Image.fromarray(final_obs.rgb).save(
        os.path.join(out_dir, f"{trial_name}_final_state.png")
    )

    with open(os.path.join(out_dir, f"{trial_name}_log.json"), "w") as f:
        json.dump(
            {
                "skill": skill.name,
                "cloth_type": skill.cloth_type,
                "num_steps": skill.num_steps,
                "random_angle": float(random_angle),
                "size_scale": size_scale,
                "offset": None if offset is None else np.round(offset, 4).tolist(),
                "step_instructions": step_instructions,
                "video": os.path.basename(video_path) if video_path else None,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    if logger is not None:
        for step_idx, instruction in enumerate(step_instructions, start=1):
            logger.info(f"[Trial {trial_index}] step {step_idx}/{len(step_instructions)}: {instruction}")
        logger.info(
            f"[Trial {trial_index}] finished: {len(step_instructions)} steps,"
            f"video: {video_path}, result dir: {out_dir}"
        )
    print(f"[Trial {trial_index}] finished, results saved to: {out_dir}")


def load_real_backend(factory_spec: str):
    """从 `模块:函数名` 加载真机后端。

    该函数需返回一个 ExecutionBackend，或 (camera, calibration, arms) 三元组。
    """
    import importlib

    module_name, _, func_name = factory_spec.partition(":")
    if not module_name or not func_name:
        raise ValueError(f"--robot-factory must be 'module:function', got: {factory_spec}")
    factory = getattr(importlib.import_module(module_name), func_name)
    built = factory()
    if isinstance(built, (tuple, list)):
        return RealRobotBackend(*built)
    return built


def main():
    parser = argparse.ArgumentParser(
        description="Skill 闭环折叠推理（仿真 SoftGym / 真机双臂平台通用）"
    )
    parser.add_argument("--instruction", type=str, default="Fold the T-shirt.",
                        help="一句总指令，例如: 把这件衣服折起来 / Fold the T-shirt")
    parser.add_argument("--backend", type=str, default="softgym", choices=["softgym", "real"],
                        help="执行后端：softgym=仿真评估，real=真机双臂平台")
    parser.add_argument("--robot-factory", type=str, default=None,
                        help="真机后端的构造入口，格式 '模块:函数名'")
    parser.add_argument("--skill", type=str, default=None, choices=sorted(SKILLS),
                        help="跳过大模型直接指定 skill")
    parser.add_argument("--list-skills", action="store_true",
                        help="列出所有 SKILL.md 技能（名称/手臂/场景/默认权重）后退出")
    # 默认配置与权重由所选技能的 SKILL.md metadata 决定，可用下面的参数覆盖
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-evals", type=int, default=1, help="该 skill 跑几个回合")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="结果输出目录（仿真视频 / 日志 / 大模型记录）")
    parser.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL,
                        help="路由用的文本大模型（必须由它选出技能）")
    parser.add_argument("--init-state", type=str, default=None,
                        help="准备好的初始布料状态路径（目录或其中的 rgb.png，可用相对路径），"
                             "例如 outputs/skill_eval/initial_states/Tshirt/case_00")
    parser.add_argument("--no-randomize", action="store_true",
                        help="关闭布料随机化（随机角度 / 随机大小 / 随机摆位）")
    parser.add_argument("--size-range", type=float, nargs=2, default=[0.85, 1.15],
                        help="布料随机缩放范围，默认 0.85~1.15")
    parser.add_argument("--offset-range", type=float, default=0.04,
                        help="布料随机平移范围（世界坐标），默认 0.04")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.list_skills:
        print(f"{'skill':<28}{'arms':<10}{'cloth':<14}{'steps':<6}checkpoint")
        for name, spec in SKILL_SPECS.items():
            ckpt = spec.checkpoint or "-"
            print(
                f"{name:<28}{spec.arms:<10}{spec.cloth_type:<14}"
                f"{spec.metadata.get('num_steps', '-')!s:<6}{ckpt}"
            )
        return

    global LOGGER

    # 日志目录：大模型路由发生在 skill 确定之前，因此先建在 output-dir 根下
    logger, log_path = setup_logging(args.output_dir, quiet=args.quiet)
    LOGGER = logger
    llm_log_path = os.path.join(args.output_dir, "llm_router.jsonl")
    logger.info(f"Command line arguments: {vars(args)}")
    logger.info(f"LLM call log: {llm_log_path}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    print(f"Device: {device}")
    logger.info(f"Device: {device}")

    # 1) 一句总指令 -> 文本大模型选择 skill
    skill_name, route_info = select_skill(args.instruction, args, llm_log_path)
    skill = SKILLS[skill_name]()
    print(
        f"Selected skill: {skill.name} (cloth type: {skill.cloth_type}, "
        f"{skill.num_steps} steps, routing source: {route_info['source']})"
    )
    logger.info(
        f"Selected skill: {skill.name} (cloth: {skill.cloth_type}, "
        f"{skill.num_steps} steps, routing source: {route_info['source']})"
    )

    # 2) 用该技能 SKILL.md 里声明的默认配置与权重（命令行参数可覆盖）
    default_config, default_checkpoint = get_model_paths(skill_name)
    config_path = args.config or default_config
    checkpoint_path = args.checkpoint or default_checkpoint
    assert config_path and checkpoint_path, (
        f"SKILL.md of skill {skill_name} does not declare config/checkpoint, please pass them explicitly"
    )
    logger.info(f"Skill {skill_name} uses config: {config_path}")
    logger.info(f"Skill {skill_name} uses weights: {checkpoint_path}")

    model, processor, cfg = load_model_and_processor(config_path, checkpoint_path, device)
    logger.info(f"Model weights loaded: {checkpoint_path}")

    # 4) 构造执行后端：仿真用 SoftGym，真机用 --robot-factory 注入的后端
    scenes = None
    if args.backend == "real":
        assert args.robot_factory, (
            "真机后端需要提供 --robot-factory 模块:函数名（返回 ExecutionBackend "
            "或 camera/calibration/arms 三元组）"
        )
        backend = load_real_backend(args.robot_factory)
    else:
        env = SoftgymClothEnv(
            render_dim=cfg.model.image_size,
            dump_visualizations=True,  # 收集全程帧用于生成视频
        )
        K = env.intrinsic_from_fov(height=cfg.model.image_size, width=cfg.model.image_size)
        backend = SoftgymBackend(env, K)
        configs, states = load_softgym_cache(cfg, skill.cloth_type)
        scenes = (configs, states)

    out_dir = os.path.join(args.output_dir, skill.name)
    os.makedirs(out_dir, exist_ok=True)

    # --init-state：复现指定的初始布料状态（角度 / 大小 / 摆位都来自 meta.json）
    init_state = None
    if args.init_state:
        init_state = load_init_state(args.init_state)
        state_cloth = init_state["meta"]["cloth_type"]
        if state_cloth != skill.cloth_type:
            # 不匹配也不报错：以大模型选中的技能为准，直接在该初始状态上执行
            msg = (
                f"Initial state cloth type ({state_cloth}) differs from the selected "
                f"skill cloth type ({skill.cloth_type}); executing the selected skill "
                "on this state anyway."
            )
            logger.warning(msg)
            print(f"[Router] {msg}")
        logger.info(f"Using prepared initial state: {init_state['dir']}")
        print(f"Using prepared initial state: {init_state['dir']}")

    # 5) skill 闭环执行全部折叠步骤（单回合失败不影响其它回合，异常写入日志）
    for trial_index in trange(args.num_evals, desc=f"Running {skill.name}"):
        scene = None
        if init_state is None and scenes is not None:
            rand_idx = int(rng.integers(len(scenes[0])))
            logger.info(f"[Trial {trial_index}] Using cached sample index: {rand_idx}")
            scene = {"config": scenes[0][rand_idx], "state": scenes[1][rand_idx]}
        try:
            run_trial(
                skill=skill,
                backend=backend,
                model=model,
                processor=processor,
                device=device,
                args=args,
                trial_index=trial_index,
                out_dir=out_dir,
                rng=rng,
                scene=scene,
                init_state=init_state,
            )
        except Exception as exc:  # noqa: BLE001
            log_exception(f"[Trial {trial_index}] execution failed", exc)
            print(f"[Trial {trial_index}] failed: {exc!r}, see {log_path}")

    backend.close()
    logger.info("All done.")
    print(f"All done. Full log: {log_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        log_exception("Inference pipeline aborted", exc)
        raise
