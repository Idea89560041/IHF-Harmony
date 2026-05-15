from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml

from model.trainers.hf_trainer import Trainer
from model.utils.log_helper import get_logger, setup_logging


class Config(dict):
    """Dictionary with attribute access for nested YAML configs."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def as_config(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({k: as_config(v) for k, v in value.items()})
    if isinstance(value, list):
        return [as_config(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IHF-Harmony training and inference")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="YAML config path")
    parser.add_argument("--eval-only", action="store_true", help="run inference only")
    parser.add_argument("--load-path", type=str, default="", help="checkpoint path for resume/eval")
    parser.add_argument("--seed", type=int, default=0, help="global random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-steps", type=int, default=None, help="override train.max_steps")
    parser.add_argument("--num-workers", type=int, default=None, help="override train.num_workers")
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
    parser.add_argument("--no-eval-after-train", action="store_true", help="skip final evaluation")
    return parser.parse_args()


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as handle:
        return as_config(yaml.safe_load(handle))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed(local_rank: int) -> tuple[bool, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1

    backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
    return True, rank, world_size


def select_device(requested: str, local_rank: int, distributed: bool) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available.")
        return torch.device(f"cuda:{max(local_rank, 0)}" if distributed else "cuda")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{max(local_rank, 0)}" if distributed else "cuda")
    return torch.device("cpu")


def prepare_output(cfg: Config, config_path: str, rank: int) -> None:
    task_dir = Path(cfg.output) / cfg.task_name
    cfg.task_dir = str(task_dir)
    cfg.image_dir = str(task_dir / "img_save")
    cfg.model_dir = str(task_dir / "model_save")
    cfg.eval_dir = str(task_dir / "eval_results")
    cfg.pred_dir = str(Path(cfg.eval_dir) / "pred")
    cfg.compare_dir = str(Path(cfg.eval_dir) / "cat_img")

    if rank != 0:
        return

    for folder in [cfg.image_dir, cfg.model_dir, cfg.pred_dir, cfg.compare_dir]:
        Path(folder).mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, task_dir / "cfg.yaml")


def apply_overrides(cfg: Config, args: argparse.Namespace) -> None:
    cfg.eval_mode = bool(args.eval_only)
    if args.load_path:
        cfg.train.load_path = args.load_path
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.no_eval_after_train:
        cfg.train.eval_after_train = False


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    set_seed(args.seed)

    distributed, rank, world_size = init_distributed(args.local_rank)
    setup_logging(rank=rank)
    logger = get_logger(__name__)
    device = select_device(args.device, args.local_rank, distributed)

    torch.backends.cudnn.benchmark = bool(getattr(cfg.train, "cudnn_benchmark", True))
    prepare_output(cfg, args.config, rank)

    if rank == 0:
        logger.info("Using device=%s distributed=%s world_size=%s", device, distributed, world_size)
        logger.info("Config loaded from %s", args.config)

    trainer = Trainer(
        cfg=cfg,
        device=device,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
        local_rank=args.local_rank,
    )

    if cfg.eval_mode:
        trainer.evaluate()
    else:
        trainer.train()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
