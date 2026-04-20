# StarVLA Qwen3-VL-OFT Training Dataflow Trace Map

This note explains the training-side trace points added for studying the StarVLA Qwen3-VL-OFT LIBERO run.

All trace rows are printed as JSON lines with this prefix:

```text
[STARVLA_TRACE] {...}
```

Tracing is opt-in:

```bash
export STARVLA_TRACE=1
export STARVLA_TRACE_FIRST_N=3
export STARVLA_TRACE_EVERY=50
export STARVLA_TRACE_MAX_ITEMS=4
export STARVLA_TRACE_MAX_DEPTH=3
```

## Current Runtime

The remote container is reachable and GPU access is restored.

```text
host: autodl-container-af19428472-6c01572e
gpu: NVIDIA GeForce RTX 4080 SUPER
memory.total: 32760 MiB
driver: 580.105.08
torch: 2.6.0+cu124
torch.cuda.is_available(): True
torch.cuda.device_count(): 1
```

## Main Training Entry

Primary Qwen3-VL-OFT training path:

```text
run_libero_train.sh
  -> accelerate launch
  -> starVLA/training/train_starvla.py
  -> build_framework(cfg)
  -> build_dataloader(cfg)
  -> VLATrainer.prepare_training()
  -> VLATrainer.train()
  -> VLATrainer._train_step()
  -> QwenOFT.forward()
  -> Qwen3VL backbone + action head + L1 loss
  -> accelerator.backward()
  -> optimizer.step()
  -> lr_scheduler.step()
  -> checkpoint save
```

The original command package is at:

```text
extracted/checkpoints/Qwen3-VL-OFT-LIBERO-4in1/run_libero_train.sh
```

## Trace Labels

`train.main.start`

Records the merged high-level config before model/data construction: framework name, `run_root_dir`, `run_id`, seed, max steps, distributed rank/world size.

`train.setup_directories`

Records output and checkpoint directories. This confirms where checkpoints, `summary.jsonl`, and config snapshots will be written.

`train.main.framework_built`

Records the model class, first parameter device/dtype, total parameter count, and trainable parameter count immediately after `build_framework(cfg)`.

For Qwen3-VL-OFT this should be `Qwenvl_OFT`.

`train.prepare_data.start`

Records dataset config before construction: `data_root_dir`, `data_mix`, `dataset_py`, per-device batch size.

`dataloader.build_vla.start`

Records VLA dataset-level options before `get_vla_dataset()`.

`dataloader.build_vla.dataset`

Records the built dataset type and length. For LIBERO mixtures this should be a `LeRobotMixtureDataset`.

`dataloader.build_vla.dataloader`

Records DataLoader type, dataset length, dataloader length, batch size, and worker count.

`train.prepare_data.end`

Records the final DataLoader summary after construction and Accelerate dataloader config.

`dataset.single.item`

Emitted when a `LeRobotSingleDataset` sample is packed. It shows raw data keys, transformed keys, and the final sample schema.

`dataset.mixture.item`

Emitted when a `LeRobotMixtureDataset` sample is selected from a sub-dataset. It shows source dataset, trajectory id, step, retry info, raw/transformed keys, and packed sample.

The packed training sample normally contains:

```text
{
  "image": List[PIL.Image],
  "lang": str,
  "language": str,
  "action": np.ndarray shaped [T, action_dim],
  "robot_tag": EmbodimentTag
}
```

If `include_state` is enabled, it also contains:

```text
"state": np.ndarray shaped [T, state_dim]
```

`train.optimizer_scheduler`

Records optimizer type, parameter groups, group learning rates, trainable parameter count per group, scheduler type, warmup steps, and total train steps.

`train.prepare_training.start`

Records seed, rank, world size, total batch size, gradient accumulation, and Accelerate state before checkpoint/freeze/distributed preparation.

`train.checkpoint.resume`

Emitted if resume mode finds a valid checkpoint.

`train.checkpoint.pretrained_loaded`

Emitted if a configured pretrained checkpoint is loaded.

`train.checkpoint.none`

Emitted if training starts without a checkpoint.

`train.prepare_training.after_freeze`

Records which modules are frozen and how many parameters remain trainable.

`train.prepare_training.after_accelerate_prepare`

Records model, optimizer, and DataLoader summaries after `accelerator.prepare(...)`.

`train.loop.start`

Records final loop parameters: max steps, current completed steps, eval interval, save interval, logging frequency, total batch size, and DataLoader summary.

`train.batch`

Records the actual Python batch returned by DataLoader. This is the most important line for understanding the interface between data and policy.

Expected batch type:

```text
List[dict]
```

Expected per-example keys:

```text
image, lang, language, action, robot_tag
```

`train.train_step.input`

Records the batch at the start of one optimization micro-step, plus whether Accelerate is syncing gradients.

`train.train_step.zero_grad`

Marks `optimizer.zero_grad()`.

`qwen_oft.forward.input`

Records the model-facing examples: images, instructions, and target action chunks.

`qwen_oft.forward.prompt`

Records the action-token prompt suffix. For QwenOFT this adds repeated action tokens and asks the model to predict the next action chunk.

`qwen_oft.forward.qwen_inputs`

Records tokenized multimodal Qwen inputs.

`qwen_oft.forward.qwenvl_outputs`

Records hidden state count and final hidden tensor summary.

`qwen_oft.forward.action_queries`

Records the hidden states gathered at action-token positions.

`qwen_oft.forward.pred_actions`

Records predicted normalized actions shaped roughly:

```text
[batch_size, chunk_len, action_dim]
```

`qwen_oft.forward.loss`

Records aligned target action chunks and the scalar L1 action loss.

`train.train_step.forward_output`

Records returned model output dict, action loss, and total loss.

`train.train_step.backward_done`

Marks `accelerator.backward(total_loss)`.

`train.train_step.grad_clip`

Marks gradient clipping.

`train.train_step.optimizer_step`

Marks optimizer step. With gradient accumulation, this may be a no-op on non-sync micro-steps.

`train.train_step.scheduler_step`

Marks scheduler step. It only runs when `accelerator.sync_gradients` is true.

`train.step.timing`

Records data loading time, model time, current metrics, current LR, and gradient sync state.

`train.metrics`

Records logged metrics at `logging_frequency`.

`train.eval_action_model.input`

Records the simple training-time action evaluation batch.

`train.eval_action_model.output`

Records prediction, target actions, distance score, and `mse_score`.

`train.checkpoint.save`

Records checkpoint path and format when periodic checkpoint save runs.

`train.finalize.save`

Records final model save path and format at the end of training.

`train.finalize.skip_save`

Records an intentional final checkpoint skip when `trainer.skip_final_save=true`. This is useful for trace smoke runs where we want real forward/backward/optimizer dataflow without writing a multi-GB Qwen checkpoint.

## How To Read One Training Step

For one real training step, read labels in this order:

```text
dataset.mixture.item
train.batch
train.train_step.input
qwen_oft.forward.input
qwen_oft.forward.prompt
qwen_oft.forward.qwen_inputs
qwen_oft.forward.qwenvl_outputs
qwen_oft.forward.action_queries
qwen_oft.forward.pred_actions
qwen_oft.forward.loss
train.train_step.forward_output
train.train_step.backward_done
train.train_step.optimizer_step
train.train_step.scheduler_step
train.step.timing
```

This sequence answers:

```text
Which dataset sample was chosen?
What raw/transformed fields existed?
What exact Python dict went into the model?
How images/language/actions became Qwen inputs?
Which hidden states became action queries?
What action tensor was predicted?
What target action tensor was used?
What loss was computed?
Did backward/optimizer/scheduler actually run?
How long data loading and model forward/backward took?
```

## Recommended Smoke Command After GPU Returns

Use a tiny one-process run first:

```bash
cd /root/autodl-tmp/starvla-official-qwen3vl-20260411/repo
export STARVLA_TRACE=1
export STARVLA_TRACE_FIRST_N=2
export STARVLA_TRACE_EVERY=20
export WANDB_MODE=disabled

accelerate launch \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  --datasets.vla_data.data_root_dir playground/Datasets/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_goal \
  --datasets.vla_data.per_device_batch_size 1 \
  --trainer.max_train_steps 2 \
  --trainer.save_interval 1000 \
  --trainer.logging_frequency 1 \
  --trainer.eval_interval 1000 \
  --trainer.skip_final_save true \
  --run_root_dir ./results/TraceTrain \
  --run_id qwen3oft_train_trace_smoke
```

Redirect stderr/stdout to a log:

```bash
... 2>&1 | tee logs/train_trace_smoke.log
```

Then inspect:

```bash
grep 'STARVLA_TRACE' logs/train_trace_smoke.log | less
```
