#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/starvla-official-qwen3vl-20260411/repo
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export STARVLA_TRACE=1
export STARVLA_TRACE_FIRST_N=3
export STARVLA_TRACE_EVERY=20
export STARVLA_TRACE_MAX_ITEMS=4
export STARVLA_TRACE_MAX_DEPTH=3
export STARVLA_TRACE_MAX_TEXT=320
export STARVLA_TRACE_MAX_STATS_NUMEL=50000
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

/root/autodl-tmp/starvla-official-qwen3vl-20260411/envs/starvla/bin/accelerate launch   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml   --num_processes 1   starVLA/training/train_starvla.py   --config_yaml ./examples/LIBERO/train_files/starvla_cotrain_libero.yaml   --framework.name QwenOFT   --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct   --framework.qwenvl.attn_implementation sdpa   --datasets.vla_data.data_root_dir playground/Datasets/LEROBOT_LIBERO_DATA   --datasets.vla_data.data_mix libero_goal   --datasets.vla_data.per_device_batch_size 1   --datasets.vla_data.video_backend torchvision_av   --trainer.pretrained_checkpoint /root/autodl-tmp/starvla-official-qwen3vl-20260411/checkpoints/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt   --trainer.freeze_modules qwen_vl_interface   --trainer.max_train_steps 2   --trainer.save_interval 1000   --trainer.logging_frequency 1   --trainer.eval_interval 1000   --trainer.skip_final_save true   --run_root_dir ./results/TraceTrain   --run_id train_trace_qwen3oft_20260421_022458   --wandb_project starVLA_Trace   --wandb_entity local
