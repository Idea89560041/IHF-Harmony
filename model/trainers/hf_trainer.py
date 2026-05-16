from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image

from model.losses import VGGLoss
from model.network.hf import IHFHarmony
from model.utils.dataset import get_dataset
from model.utils.log_helper import get_logger


try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None


logger = get_logger(__name__)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def reduce_for_logging(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return value.detach()
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced / world_size


def display_range(images: torch.Tensor, value_range: str) -> torch.Tensor:
    if value_range == "minus_one_one":
        return (images + 1.0) * 0.5
    return images


class Trainer:
    def __init__(
        self,
        cfg: Any,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        distributed: bool = False,
        local_rank: int = -1,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.distributed = distributed
        self.local_rank = local_rank
        self.value_range = getattr(cfg.dataset, "intensity_range", "zero_one")
        self.global_step = 0

        model = IHFHarmony(
            pad_size=cfg.network.pad_size,
            in_channel=cfg.network.in_channel,
            out_channels=cfg.network.out_channels,
            weight_type=cfg.network.weight_type,
            style_dim=cfg.network.style_dim,
            style_base_channels=cfg.network.style_base_channels,
            use_squeeze=getattr(cfg.network, "use_squeeze", False),
        ).to(device)

        if distributed:
            if device.type == "cuda":
                model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
            else:
                model = DistributedDataParallel(model)
        self.model = model

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.betas),
            weight_decay=cfg.optimizer.weight_decay,
        )
        self.scheduler = self._build_scheduler()
        self.criterion = VGGLoss(cfg.loss.vgg_encoder, input_range=self.value_range).to(device)
        self.amp_enabled = bool(cfg.train.amp and device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.writer = self._build_writer()

        load_path = getattr(cfg.train, "load_path", "")
        if cfg.eval_mode or (cfg.train.resume and load_path):
            self.load_checkpoint(load_path)

        if rank == 0:
            parameter_count = sum(p.numel() for p in unwrap_model(self.model).parameters())
            logger.info("Model parameters: %.2fM", parameter_count / 1e6)

    def _build_writer(self):
        if self.rank != 0 or SummaryWriter is None:
            return None
        return SummaryWriter(log_dir=str(Path(self.cfg.task_dir) / "runs"))

    def _build_scheduler(self):
        scheduler_type = self.cfg.lr_scheduler.type.lower()
        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.cfg.train.max_steps),
                eta_min=self.cfg.lr_scheduler.eta_min,
            )
        if scheduler_type == "none":
            return None
        raise ValueError(f"Unsupported lr_scheduler.type: {self.cfg.lr_scheduler.type}")

    def make_loader(self, split: str, training: bool) -> DataLoader:
        dataset_cfg = self.cfg.dataset.train if training else self.cfg.dataset.test
        dataset = get_dataset(dataset_cfg, intensity_range=self.value_range)
        sampler = None
        if self.distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=training,
                drop_last=training,
            )
        return DataLoader(
            dataset,
            batch_size=dataset_cfg.batch_size,
            shuffle=training and sampler is None,
            sampler=sampler,
            num_workers=self.cfg.train.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=training,
        )

    def train(self) -> None:
        loader = self.make_loader("train", training=True)
        sampler = loader.sampler if isinstance(loader.sampler, DistributedSampler) else None
        max_steps = int(self.cfg.train.max_steps)
        epoch = 0

        self.model.train()
        while self.global_step < max_steps:
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in loader:
                self.global_step += 1
                self.train_step(batch)
                if self.global_step >= max_steps:
                    break
            epoch += 1

        if self.rank == 0:
            self.save_checkpoint("final.ckpt")
        if self.cfg.train.eval_after_train:
            self.evaluate()

    def train_step(self, batch: tuple[torch.Tensor, ...]) -> None:
        content = batch[0].to(self.device, non_blocking=True)
        target = batch[1].to(self.device, non_blocking=True)

        autocast = torch.cuda.amp.autocast if self.amp_enabled else nullcontext
        with autocast():
            output = self.model(content, target)
            output = output.clamp(-1.0, 1.0) if self.value_range == "minus_one_one" else output.clamp(0.0, 1.0)
            loss_anatomy, loss_artifact = self.criterion(
                content,
                target,
                output,
                keep_ratio=self.cfg.loss.keep_ratio,
            )
            loss = self.cfg.loss.anatomy_weight * loss_anatomy + self.cfg.loss.artifact_weight * loss_artifact

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

        self.log_step(content, target, output, loss, loss_anatomy, loss_artifact)

    def log_step(
        self,
        content: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor,
        loss: torch.Tensor,
        loss_anatomy: torch.Tensor,
        loss_artifact: torch.Tensor,
    ) -> None:
        loss_log = reduce_for_logging(loss, self.world_size)
        anatomy_log = reduce_for_logging(loss_anatomy, self.world_size)
        artifact_log = reduce_for_logging(loss_artifact, self.world_size)

        if self.rank != 0:
            return

        lr = self.optimizer.param_groups[0]["lr"]

        if self.writer is not None:
            self.writer.add_scalar("train/lr", lr, self.global_step)
            self.writer.add_scalar("train/loss", loss_log.item(), self.global_step)
            self.writer.add_scalar("train/loss_anatomy", anatomy_log.item(), self.global_step)
            self.writer.add_scalar("train/loss_artifact", artifact_log.item(), self.global_step)

        if self.global_step % self.cfg.train.print_freq == 0 or self.global_step == 1:
            logger.info(
                "step=%d lr=%.6g loss=%.5f anatomy=%.5f artifact=%.5f",
                self.global_step,
                lr,
                loss_log.item(),
                anatomy_log.item(),
                artifact_log.item(),
            )

        if self.global_step % self.cfg.train.image_freq == 0 or self.global_step == 1:
            grid = torch.cat([content[:1], target[:1], output[:1]], dim=0)
            grid = display_range(grid.detach().cpu(), self.value_range)
            save_image(grid, Path(self.cfg.image_dir) / f"{self.global_step:07d}.png", nrow=1)

        if self.global_step % self.cfg.train.save_freq == 0:
            self.save_checkpoint(f"{self.global_step:07d}.ckpt")

    @torch.no_grad()
    def evaluate(self) -> None:
        loader = self.make_loader("test", training=False)
        self.model.eval()

        for batch_id, batch in enumerate(loader):
            content = batch[0].to(self.device, non_blocking=True)
            target = batch[1].to(self.device, non_blocking=True)
            names = batch[2] if len(batch) > 2 else [f"sample_{batch_id:05d}_{i}.png" for i in range(content.size(0))]

            output = self.model(content, target)
            output = output.clamp(-1.0, 1.0) if self.value_range == "minus_one_one" else output.clamp(0.0, 1.0)
            output_cpu = display_range(output.detach().cpu(), self.value_range)
            content_cpu = display_range(content.detach().cpu(), self.value_range)
            target_cpu = display_range(target.detach().cpu(), self.value_range)

            for idx, name in enumerate(names):
                save_image(output_cpu[idx].unsqueeze(0), Path(self.cfg.pred_dir) / name)
                if idx == 0:
                    comparison = torch.stack([content_cpu[idx], target_cpu[idx], output_cpu[idx]], dim=0)
                    save_image(comparison, Path(self.cfg.compare_dir) / name, nrow=1)

            if self.rank == 0 and batch_id % 10 == 0:
                logger.info("evaluated batch %d", batch_id)

        if self.rank == 0:
            logger.info("Evaluation outputs saved under %s", self.cfg.eval_dir)
        self.model.train()

    def save_checkpoint(self, name: str) -> None:
        if self.rank != 0:
            return
        path = Path(self.cfg.model_dir) / f"{name}.pth.tar"
        checkpoint = {
            "step": self.global_step,
            "model": unwrap_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        torch.save(checkpoint, path)
        logger.info("saved checkpoint: %s", path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        if not checkpoint_path:
            raise ValueError("A checkpoint path is required for resume/eval.")
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(str(path), map_location=self.device)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        cleaned_state = {key.replace("module.", "", 1): value for key, value in state.items()}
        unwrap_model(self.model).load_state_dict(cleaned_state, strict=False)

        if "optimizer" in checkpoint and not self.cfg.eval_mode:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint.get("scheduler") is not None and not self.cfg.eval_mode:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.global_step = int(checkpoint.get("step", 0))
        logger.info("loaded checkpoint %s at step %d", path, self.global_step)
