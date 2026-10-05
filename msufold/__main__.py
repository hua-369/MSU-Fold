import datetime
import os
import random
from typing import Dict, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import v2
from tqdm import tqdm

from .data import Datasets
from .losses import Losses
from .metrics import Metrics
from .models import Models
from .optim import Optimizers, Schedulers
from .utils.visualization import save_predictions, visualize_action


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)

    # 多进程 DDP 时只让主进程写配置文件/评估/保存，避免多进程互相覆盖
    if trainer.is_main:
        print(os.getcwd())
        with open("config.yaml", "w") as f:
            OmegaConf.save(cfg, f)

    try:
        if not cfg.eval_only:
            trainer.prepare_train()
            trainer.train()
        trainer.eval()
    finally:
        # 即使中途报错也要销毁进程组，否则会留僵尸进程
        trainer.finalize()


class Trainer:
    def __init__(self, cfg):
        # ---- 并行策略 ----
        # 1) 用 torchrun 启动（WORLD_SIZE>1）时走 DDP，每个进程占一张卡；
        # 2) 单进程且 device_ids 多于一张时退化为 DataParallel（注意：本机 3 卡以上 DP 会
        #    出现 cuDNN/misaligned address 报错，4 卡请用 torchrun 的 DDP 方式）。
        self.distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
        if self.distributed:
            # 默认 10 分钟超时对 softgym 评估太短，放宽到 6 小时
            torch.distributed.init_process_group(
                backend="nccl", timeout=datetime.timedelta(hours=6)
            )
            self.rank = torch.distributed.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
            self.world_size = torch.distributed.get_world_size()
            torch.cuda.set_device(self.local_rank)
        else:
            self.rank, self.local_rank, self.world_size = 0, 0, 1
        self.is_main = self.rank == 0

        device_ids = list(cfg.get("device_ids", [0]) or [0])
        self.use_cuda = torch.cuda.is_available() and not cfg.use_cpu
        if not self.use_cuda:
            device_ids = []
        else:
            n_gpu = torch.cuda.device_count()
            device_ids = [d for d in device_ids if d < n_gpu]
            if not device_ids:
                device_ids = [0]
        self.device_ids = device_ids
        self.device = (
            torch.device(f"cuda:{self.local_rank}")
            if self.distributed
            else (torch.device(f"cuda:{device_ids[0]}") if self.use_cuda else torch.device("cpu"))
        )
        if self.is_main:
            print(f"Rank {self.rank}/{self.world_size} device={self.device} "
                  f"({'DDP' if self.distributed else ('DataParallel' if len(device_ids) > 1 else 'single')})")

        if cfg.use_wandb and self.is_main and not cfg.eval_only and not cfg.debug:
            self.writer = wandb.init(
                project="msu-fold",
                group=cfg.train_dataset.name,
                name="+".join(HydraConfig.get().overrides.task),
                resume="allow",
            )
            wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        else:
            self.writer = None

        self.cfg = cfg

        self.seed_randomness()

        # raw_model 始终是未包装的原始模型（含 get_action / device 等接口），
        # self.model 在多卡时是 DataParallel 包装，仅用于训练前向
        self.raw_model = Models.get_by_name(self.cfg.model, device=self.device).to(self.device)
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.raw_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                # SigLIP 的 logit_scale/pooler 不参与 loss，需要允许未使用参数
                find_unused_parameters=True,
            )
        elif len(self.device_ids) > 1:
            print(f"Using DataParallel on GPUs {self.device_ids}")
            self.model = torch.nn.DataParallel(self.raw_model, device_ids=self.device_ids)
        else:
            self.model = self.raw_model

        self.train_dataloader, self.test_dataloader, self.input_processor = (
            Datasets.get_dataloaders(self.cfg)
        )
        self.metrics = Metrics(self.cfg.metrics)

    def barrier(self):
        if self.distributed:
            torch.distributed.barrier()

    def train(self):
        if self.start_epoch < self.cfg.epochs:
            iterator = list(range(self.start_epoch, self.cfg.epochs))
            pbar = tqdm(iterator, desc="Train", disable=not self.is_main)
            for epoch in pbar:
                # DDP 下每个 epoch 要换一次 shuffle 顺序
                sampler = getattr(self.train_dataloader, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

                self.train_epoch(epoch)
                self.barrier()

                if self.cfg.eval_epochs and (epoch + 1) % self.cfg.eval_epochs == 0:
                    # 评估与保存只在主进程做
                    if self.is_main:
                        has_improved, _ = self.eval_epoch(epoch)
                        if has_improved:
                            self.save_model(epoch, is_best=True)
                    self.barrier()
                if self.cfg.save_epochs and (epoch + 1) % self.cfg.save_epochs == 0:
                    if self.is_main:
                        self.save_model(epoch)
                    self.barrier()

            epoch = self.cfg.epochs - 1
            if self.is_main:
                self.save_model(epoch)
            self.barrier()

    def eval(self):
        self.load_model(load_best=self.cfg.load_best)
        if self.distributed:
            # 仿真评估（softgym）动辄几十分钟到数小时，远超 NCCL 默认 10 分钟的
            # watchdog 超时：非主进程卡在 barrier 上会被看门狗整体杀掉。
            # 评估阶段只用到 raw_model（非 DDP 包装），不再有集合通信，
            # 因此先同步一次，然后销毁进程组：只留 rank0 做评估，其它 rank 直接退出。
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
            self.distributed = False  # 之后的 barrier() 变成空操作
        if not self.is_main:
            return
        eval_file = f"eval_{self.cfg.test_dataset.name}.yaml"
        _, metric_dict = self.eval_epoch()
        for k, val in metric_dict.items():
            if isinstance(val, dict):
                for sub_k, sub_val in val.items():
                    print(f"{k} {sub_k}:\t{sub_val:.2f}")
            else:
                print(f"{k}:\t{val:.2f}")

        if os.path.isfile(eval_file):
            print("Found YAML file")
            with open(eval_file, "r") as f:
                old_results = yaml.load(f, Loader=yaml.Loader)
            for k, val in old_results.items():
                if k not in metric_dict:
                    metric_dict[k] = val
                else:
                    if val != metric_dict[k]:
                        print(f"Old value for {k} = {val} ; New value {metric_dict[k]}")
        with open(eval_file, "w") as f:
            yaml.dump(metric_dict, f)
        self.barrier()

    def finalize(self):
        if self.distributed:
            torch.distributed.destroy_process_group()

    def seed_randomness(self):
        # 多进程时给每个 rank 不同的随机种子，保证数据增强/打乱不同
        seed = self.cfg.seed + self.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def prepare_train(self):
        n_parallel = self.world_size if self.distributed else len(self.device_ids)
        if n_parallel > 1 and self.cfg.batch_size % n_parallel != 0:
            print(
                f"Warning: batch_size={self.cfg.batch_size} 不能被并行数 {n_parallel} 整除，"
                "建议设为它的整数倍（注意 DDP 下 batch_size 是每张卡的批量）"
            )
            print(
                f"Warning: batch_size={self.cfg.batch_size} < {len(self.device_ids)} GPUs, "
                "建议把 batch_size 设为 GPU 数量的整数倍"
            )
        params_non_frozen = filter(lambda p: p.requires_grad, self.raw_model.parameters())
        self.loss_fn = Losses.get_by_name(cfg=self.cfg.loss)
        self.optimizer = Optimizers.get_by_name(cfg=self.cfg.optim, params=params_non_frozen)
        assert self.train_dataloader is not None
        self.scheduler = Schedulers.get_by_name(
            cfg=self.cfg.scheduler,
            optimizer=self.optimizer,
            max_iters=len(self.train_dataloader) * self.cfg.epochs,
        )
        self.load_model()

    def train_epoch(self, epoch=None):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_dataloader)
        for i, sample in enumerate(pbar):
            sample = self.move_sample_to_device(sample)

            # Visualize inputs to model
            if self.cfg.debug and self.cfg.visualize_model_inputs:
                self.visualize_model_inputs(sample)

            output = self.model(sample)

            loss, intermediate_losses = self.loss_fn(output, sample)

            self.optimizer.zero_grad()
            assert loss is not None
            loss.backward()

            if self.cfg.debug:
                for name, param in self.raw_model.named_parameters():
                    if param.requires_grad and param.grad is None:
                        raise ValueError(f"Parameter {name} might not have gradient attached!")

            if self.cfg.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.cfg.gradient_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            if self.writer:
                self.writer.log({"loss": loss, "epoch": epoch})
                for k, val in intermediate_losses.items():
                    self.writer.log({k: val})
                for j, param_group in enumerate(self.optimizer.param_groups):
                    self.writer.log({f"lr_{j}": param_group["lr"]})

            assert loss is not None
            total_loss += loss.item()
            pbar.set_description("Training loss:{}".format(total_loss / (i + 1)))

    def eval_epoch(self, epoch=None) -> Tuple[Optional[bool], Dict[str, torch.Tensor]]:
        self.model.eval()
        has_improved = None
        if epoch is not None or self.cfg.simulator is None:
            # During training or in testing if simulator is not specified
            has_improved, metric_dict = self.eval_epoch_pixel(epoch)
        elif self.cfg.simulator == "softgym":
            metric_dict = self.eval_epoch_softgym(epoch)
        else:
            raise ValueError(f"Simulator {self.cfg.simulator} not supported")

        if self.writer:
            for k, val in metric_dict.items():
                self.writer.log({k: val})

        if epoch is not None:
            assert (
                has_improved is not None
            ), "has_improved boolean must be specified during training"
        return has_improved, metric_dict

    def eval_epoch_pixel(self, epoch):
        self.metrics.reset()
        num_samples = 0
        for sample in tqdm(self.test_dataloader, desc="Evaluate"):
            sample = self.move_sample_to_device(sample)

            # Pick and place prediction
            ret_get_action = self.raw_model.get_action(sample, return_raw_output=True)
            assert isinstance(ret_get_action, Tuple)
            action, raw_output = ret_get_action

            if any("pick" in k for k in sample.keys()):
                # If samples are available
                self.metrics(action=action, sample=sample, raw_output=raw_output)

            if self.cfg.visualize_predictions:
                out_folder = (
                    os.path.join(
                        "eval_background",
                        self.cfg.test_dataset.name,
                        "pixel_metrics",
                        "best" if self.cfg.load_best else "last",
                    )
                    if epoch is None
                    else os.path.join("eval", "pixel_metrics", f"epoch_{epoch}")
                )
                visualizations = visualize_action(sample, action)
                for j in range(len(visualizations)):
                    kwargs = {
                        "rgb": sample["raw_rgb"][j].cpu().numpy(),
                        "depth": sample["depth"][j].squeeze().cpu().numpy(),
                        "viz": visualizations[j],
                    }
                    if raw_output is not None:
                        for k, val in raw_output.items():
                            if "heatmap" in k:
                                kwargs[k] = val[j]
                    save_predictions(
                        out_folder=out_folder,
                        out_file_name=f"{num_samples}_{sample['raw_instruction'][j]}.png",
                        **kwargs,
                    )
                    num_samples += 1

        has_improved, metric_dict = self.metrics.summary()
        if self.writer:
            for k, val in metric_dict.items():
                self.writer.log({k: val})

        return has_improved, metric_dict

    def eval_epoch_softgym(self, epoch):
        """闭环仿真评估：按训练集决定评单臂任务还是双臂任务。"""
        from .env._pyflex_silence import silence_c_stdout
        from .env.softgym_evaluator import SoftgymSingleEvaluator

        evaluator = SoftgymSingleEvaluator(
            cfg=self.cfg,
            model=self.raw_model,
            processor=self.input_processor,
        )
        # 双臂数据集（mixed_sequential loader）→ 只评 4 个 Mixed* 任务
        dual_mode = self.cfg.train_dataset.get("loader") == "mixed_sequential"
        # pyflex 的 "[dbg] depth_bits=..." 会刷屏，评估期间屏蔽，只保留进度条
        with silence_c_stdout():
            if dual_mode:
                # 只评估双臂任务（与 dual_data_sequential 数据集对应，只评 si/usi，不评 ut）
                from .env.softgym_dual_evaluator import SoftgymDualEvaluator

                dual_evaluator = SoftgymDualEvaluator(
                    self.cfg,
                    model=self.raw_model,
                    processor=self.input_processor,
                    env=evaluator.env,  # 复用环境，避免 pyflex 二次初始化
                )
                dual_tasks = [
                    "MixedSquareHalfFold",
                    "MixedSquareCornerFold",
                    "MixedTshirtFold",
                    "MixedTrousersFold",
                ]
                # eval_tasks='[MixedSquareCornerFold]' 可只评指定任务
                filter_tasks = self.cfg.get("eval_tasks")
                if filter_tasks:
                    if isinstance(filter_tasks, str):
                        import json

                        try:
                            filter_tasks = json.loads(filter_tasks)
                        except Exception:
                            filter_tasks = [filter_tasks]
                    filter_tasks = list(filter_tasks)
                    dual_tasks = [t for t in dual_tasks if t in filter_tasks]
                for task in dual_tasks:
                    dual_evaluator.evaluate(num_evals=self.cfg.num_evals, task=task)
                # 只保留双臂任务的指标，average_success 只统计这 4 个任务
                evaluator.success = dual_evaluator.success
                evaluator.additional_metrics = dual_evaluator.additional_metrics
            else:
                for task in [
                    "CornerFold",
                    "TriangleFold",
                    "StraightFold",
                    "TshirtFold",
                    "TrousersFold",
                ]:
                    evaluator.evaluate(num_evals=self.cfg.num_evals, task=task)
            evaluator.close()

        return evaluator.summary()

    def load_model(self, load_best=False):
        if load_best and os.path.isfile(os.path.join("checkpoints", "best.pth")):
            checkpoint_file = os.path.join("checkpoints", "best.pth")
        elif os.path.isfile(os.path.join("checkpoints", "last.pth")):
            checkpoint_file = os.path.join("checkpoints", "last.pth")
        else:
            if self.cfg.eval_only and not self.cfg.debug:
                raise FileNotFoundError("Cannot evaluate with untrained model")
            self.start_epoch = 0
            return

        # PyTorch >= 2.6 的 torch.load 默认 weights_only=True，无法反序列化 numpy 对象
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        self.start_epoch = checkpoint["epoch"]
        random.setstate(checkpoint["random_states"][0])
        np.random.set_state(checkpoint["random_states"][1])
        torch.set_rng_state(checkpoint["random_states"][2].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint["random_states"][3].cpu())

        self.raw_model.load_state_dict(checkpoint["model"])

        if not self.cfg.eval_only:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            if self.scheduler is not None and "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])

            if "best_eval" in checkpoint:
                self.metrics.best_eval = checkpoint["best_eval"]

        print(f"Loaded model from checkpoint {checkpoint_file}")

    def save_model(self, epoch, is_best=False):
        os.makedirs("checkpoints", exist_ok=True)
        if is_best:
            model_path = os.path.join("checkpoints", "best.pth")
        else:
            model_path = os.path.join("checkpoints", "last.pth")
        state_dict = {
            "epoch": epoch + 1,
            "random_states": (
                random.getstate(),
                np.random.get_state(),
                torch.get_rng_state(),
                (torch.cuda.get_rng_state() if torch.cuda.is_available() else None),
            ),
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()
        if self.metrics.best_eval is not None:
            state_dict["best_eval"] = self.metrics.best_eval
        torch.save(state_dict, model_path)

    def move_sample_to_device(self, sample):
        # Move all inputs to device
        for k, val in sample.items():
            if isinstance(val, torch.Tensor) or k == "graph":
                sample[k] = val.to(self.device)
        return sample

    def visualize_model_inputs(self, sample):
        input_visualizations = {}
        if self.writer is None:
            fig, axes = plt.subplots(ncols=4, nrows=3)
            flat_axes = axes.flat
            i = 0
        else:
            fig, flat_axes, i = None, None, None

        for k, val in sample.items():
            if any(subk in k for subk in ["rgb", "depth", "heatmap"]):
                if "rgb" in k:
                    transform = v2.Compose([
                        v2.Normalize(
                            (0.0, 0.0, 0.0),
                            (
                                1 / 0.26862954,
                                1 / 0.26130258,
                                1 / 0.27577711,
                            ),
                        ),
                        v2.Normalize(
                            (-0.48145466, -0.4578275, -0.40821073),
                            (1.0, 1.0, 1.0),
                        ),
                        v2.ToPILImage(),
                    ])
                else:
                    transform = v2.ToPILImage()

                if len(val.shape) > 2:
                    if len(val.shape) < 5:
                        imgs = [transform(val[0])]
                        names = [k]
                    else:
                        imgs = [transform(v) for v in val[0]]
                        names = [f"{k}_{i}" for i in range(len(imgs))]

                    for img, name in zip(imgs, names):
                        if self.writer:
                            input_visualizations[name] = wandb.Image(
                                img,
                                caption=sample["raw_instruction"][0],
                            )
                        else:
                            assert flat_axes is not None
                            assert i is not None
                            flat_axes[i].imshow(img)
                            if k == "depth":
                                for k_in, val_in in sample.items():
                                    if "heatmap" in k_in and len(val_in[0].shape) > 1:
                                        flat_axes[i].imshow(transform(val_in[0]), alpha=0.5)
                            flat_axes[i].set_title(name)
                            i += 1

        if self.writer:
            self.writer.log(input_visualizations)
        else:
            assert fig is not None
            fig.suptitle(sample["raw_instruction"][0])
            plt.show()


if __name__ == "__main__":
    main()
