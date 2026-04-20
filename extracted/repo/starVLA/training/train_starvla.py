# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import re
import time
from pathlib import Path
from datetime import timedelta
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, normalize_dotlist_args
from deployment.model_server.tools.trace_tools import should_trace, trace

deepspeed_plugin = DeepSpeedPlugin()
# Avoid default c10d 30-minute timeout (1800000ms) on slow/imbalanced steps.
# You can override via env: TORCH_DIST_TIMEOUT_MINUTES=1440
dist_timeout_minutes = int(os.environ.get("TORCH_DIST_TIMEOUT_MINUTES", "1440"))
process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=dist_timeout_minutes))
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, kwargs_handlers=[process_group_kwargs])
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def _dist_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _dist_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _safe_len(obj):
    try:
        return len(obj)
    except Exception:
        return None


def _model_runtime_summary(model):
    try:
        first_param = next(model.parameters())
        dtype = str(first_param.dtype)
        device = str(first_param.device)
    except StopIteration:
        dtype = None
        device = None
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "type": type(model).__name__,
        "device": device,
        "dtype": dtype,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def _dataloader_summary(dataloader):
    dataset = getattr(dataloader, "dataset", None)
    collate_fn = getattr(dataloader, "collate_fn", None)
    return {
        "type": type(dataloader).__name__,
        "dataset_type": type(dataset).__name__ if dataset is not None else None,
        "dataset_len": _safe_len(dataset),
        "dataloader_len": _safe_len(dataloader),
        "batch_size": getattr(dataloader, "batch_size", None),
        "num_workers": getattr(dataloader, "num_workers", None),
        "collate_fn": getattr(collate_fn, "__name__", repr(collate_fn)),
    }


def _optimizer_summary(optimizer):
    groups = []
    for group in optimizer.param_groups:
        params = group.get("params", [])
        groups.append(
            {
                "name": group.get("name"),
                "lr": group.get("lr"),
                "weight_decay": group.get("weight_decay"),
                "num_param_tensors": len(params),
                "num_params": sum(p.numel() for p in params),
                "trainable_params": sum(p.numel() for p in params if getattr(p, "requires_grad", False)),
            }
        )
    return {"type": type(optimizer).__name__, "groups": groups}


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    trace(
        "train.setup_directories",
        output_dir=str(output_dir),
        checkpoint_dir=str(output_dir / "checkpoints"),
        rank=_dist_rank(),
        world_size=_dist_world_size(),
    )
    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    trace(
        "train.prepare_data.start",
        data_root_dir=cfg.datasets.vla_data.get("data_root_dir", None),
        data_mix=cfg.datasets.vla_data.get("data_mix", None),
        dataset_py=cfg.datasets.vla_data.get("dataset_py", None),
        per_device_batch_size=cfg.datasets.vla_data.get("per_device_batch_size", None),
        output_dir=str(output_dir),
        rank=_dist_rank(),
    )
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    trace(
        "train.prepare_data.end",
        dataloader=_dataloader_summary(vla_train_dataloader),
        dispatch_batches=accelerator.dataloader_config.dispatch_batches,
        rank=_dist_rank(),
    )
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    trace(
        "train.optimizer_scheduler",
        model=_model_runtime_summary(model),
        optimizer=_optimizer_summary(optimizer),
        lr_scheduler_type=cfg.trainer.lr_scheduler_type,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        max_train_steps=cfg.trainer.max_train_steps,
        rank=_dist_rank(),
    )
    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)
        trace(
            "train.prepare_training.start",
            rank=rank,
            world_size=_dist_world_size(),
            seed=seed,
            total_batch_size=self.total_batch_size,
            gradient_accumulation_steps=self.accelerator.gradient_accumulation_steps,
            accelerator_state=str(self.accelerator.state),
        )

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)
        trace(
            "train.prepare_training.after_freeze",
            freeze_modules=freeze_modules,
            model=_model_runtime_summary(self.model),
            resume_from_checkpoint=getattr(self, "resume_from_checkpoint", None),
            completed_steps=self.completed_steps,
            rank=rank,
        )

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )
        trace(
            "train.prepare_training.after_accelerate_prepare",
            model=_model_runtime_summary(self.model),
            optimizer=_optimizer_summary(self.optimizer),
            dataloader=_dataloader_summary(self.vla_train_dataloader),
            rank=rank,
        )

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                trace(
                    "train.checkpoint.resume",
                    checkpoint_dir=self.checkpoint_dir,
                    resume_from_checkpoint=self.resume_from_checkpoint,
                    completed_steps=self.completed_steps,
                    rank=_dist_rank(),
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
            trace(
                "train.checkpoint.pretrained_loaded",
                pretrained_checkpoint=pretrained_checkpoint,
                reload_modules=reload_modules,
                completed_steps=self.completed_steps,
                rank=_dist_rank(),
            )
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0
            trace("train.checkpoint.none", checkpoint_dir=self.checkpoint_dir, completed_steps=self.completed_steps, rank=_dist_rank())

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
            trace(
                "train.checkpoint.save",
                checkpoint_path=checkpoint_path,
                save_format=save_format,
                completed_steps=self.completed_steps,
                summary_data=summary_data,
                rank=_dist_rank(),
            )

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                full_cfg_path = output_dir / "config.full.yaml"
                logger.info(f"📦 Saving full merged configuration to `{full_cfg_path}`...")
                self.config.save_full_config(full_cfg_path, resolve=True)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and dist.get_rank() == 0:
            metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")
            trace("train.metrics", step=self.completed_steps, metrics=metrics, rank=_dist_rank())

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        if should_trace("train.batch", step=self.completed_steps):
            trace("train.batch", step=self.completed_steps, batch=batch_vla, batch_size=len(batch_vla), rank=_dist_rank())
        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        trace(
            "train.loop.start",
            max_train_steps=self.config.trainer.max_train_steps,
            completed_steps=self.completed_steps,
            eval_interval=self.config.trainer.eval_interval,
            save_interval=self.config.trainer.save_interval,
            logging_frequency=self.config.trainer.logging_frequency,
            total_batch_size=self.total_batch_size,
            dataloader=_dataloader_summary(self.vla_train_dataloader),
            rank=_dist_rank(),
        )
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            if should_trace("train.step.timing", step=self.completed_steps):
                trace(
                    "train.step.timing",
                    step=self.completed_steps,
                    sync_gradients=self.accelerator.sync_gradients,
                    data_time=step_metrics["data_time"],
                    model_time=step_metrics["model_time"],
                    metrics=step_metrics,
                    lr=self.lr_scheduler.get_last_lr(),
                    rank=_dist_rank(),
                )
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        trace("train.eval_action_model.input", step=self.completed_steps, examples=examples, actions=actions, rank=_dist_rank())
        output_dict = self.model.predict_action(examples=examples, use_ddim=True, num_ddim_steps=20)

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots
            trace(
                "train.eval_action_model.output",
                step=self.completed_steps,
                normalized_actions=normalized_actions,
                actions=actions,
                score=score,
                mse_score=step_metrics["mse_score"],
                rank=_dist_rank(),
            )

        del examples
        dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        do_trace = should_trace("train.train_step", step=self.completed_steps)
        if do_trace:
            trace(
                "train.train_step.input",
                step=self.completed_steps,
                batch=batch_vla,
                sync_gradients=self.accelerator.sync_gradients,
                rank=_dist_rank(),
            )
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            if do_trace:
                trace("train.train_step.zero_grad", step=self.completed_steps, rank=_dist_rank())

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss
            if do_trace:
                trace(
                    "train.train_step.forward_output",
                    step=self.completed_steps,
                    output_dict=output_dict,
                    action_loss=action_loss,
                    total_loss=total_loss,
                    rank=_dist_rank(),
                )

            self.accelerator.backward(total_loss)
            if do_trace:
                trace("train.train_step.backward_done", step=self.completed_steps, total_loss=total_loss, rank=_dist_rank())

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
                if do_trace:
                    trace(
                        "train.train_step.grad_clip",
                        step=self.completed_steps,
                        gradient_clipping=self.config.trainer.gradient_clipping,
                        rank=_dist_rank(),
                    )

            self.optimizer.step()
            if do_trace:
                trace(
                    "train.train_step.optimizer_step",
                    step=self.completed_steps,
                    sync_gradients=self.accelerator.sync_gradients,
                    lr=self.lr_scheduler.get_last_lr(),
                    rank=_dist_rank(),
                )
            # Only step the scheduler when an actual optimizer update occurs.
            # Inside accelerator.accumulate(), optimizer.step() is a no-op on
            # non-sync micro-steps, but lr_scheduler.step() always advances
            # the internal counter.  This caused the scheduler to advance
            # gradient_accumulation_steps times faster than intended, leading
            # to premature LR decay and incorrect LR on resume (#204).
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
                if do_trace:
                    trace("train.train_step.scheduler_step", step=self.completed_steps, lr=self.lr_scheduler.get_last_lr(), rank=_dist_rank())

        return {
            "action_dit_loss": action_loss.item(),
        }

    def _finalize_training(self):
        """Training end processing."""
        if getattr(self.config.trainer, "skip_final_save", False):
            trace(
                "train.finalize.skip_save",
                reason="trainer.skip_final_save",
                completed_steps=self.completed_steps,
                rank=_dist_rank(),
            )
            if self.accelerator.is_main_process:
                wandb.finish()
            self.accelerator.wait_for_everyone()
            return

        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")
            trace(
                "train.finalize.save",
                final_checkpoint=final_checkpoint,
                save_format=save_format,
                completed_steps=self.completed_steps,
                rank=_dist_rank(),
            )

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")
    trace(
        "train.main.start",
        framework=cfg.framework.get("name", cfg.framework.get("framework_py", None)),
        run_root_dir=cfg.get("run_root_dir", None),
        run_id=cfg.get("run_id", None),
        seed=cfg.get("seed", None),
        max_train_steps=cfg.trainer.get("max_train_steps", None),
        rank=_dist_rank(),
        world_size=_dist_world_size(),
    )

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    trace("train.main.framework_built", model=_model_runtime_summary(vla), framework=type(vla).__name__, rank=_dist_rank())
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
