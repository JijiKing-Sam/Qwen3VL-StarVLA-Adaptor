# StarVLA Qwen3-VL Repro Study Package

这是从远端独立复现目录导出的本地学习包，用来学习 StarVLA Qwen3-VL-OFT + LIBERO eval 的全过程数据流。

## 目录结构

- `archives/`: 从远端下载的 manifest、校验文件和 patch。压缩包本体在本地存在，但 GitHub 版通过 `.gitignore` 排除，因为其中有超过 GitHub 单文件限制的大二进制文件。
- `extracted/repo/`: StarVLA 源码，包含本次加入的 trace 输出代码；已排除 8.3GB Qwen3-VL base model、数据集 symlink 和 `.git`。
- `extracted/logs/`: 复现、下载、LIBERO eval、trace smoke 的完整日志。
- `extracted/results/`: 本次生成的视频结果，包含 `libero_goal` 500 个 full eval mp4、10 个 smoke mp4、10 个 trace smoke mp4。
- `extracted/checkpoints/`: 官方 `Qwen3-VL-OFT-LIBERO-4in1` 的 README、config、dataset statistics 等元信息；不含 9.2GB `.pt` 权重。
- `extracted/data/`: LIBERO 数据集的 metadata；不含 parquet/mp4 训练数据。
- `extracted/third_party/LIBERO/`: LIBERO 官方代码与资产，排除了 `.git` 和 pycache。
- `notes/`: 本地学习路线和大文件说明。

## 推荐学习入口

1. 先看 `notes/01_dataflow_map.md`，理解 eval 到模型输出的调用链。
2. 再看 `extracted/logs/trace_smoke_20260411_215258/server_trace.log`，它展示 server 侧模型输入、Qwen hidden state、action head 输出。
3. 对照 `extracted/logs/trace_smoke_20260411_215258/eval_trace.log`，看 LIBERO 观测、client resize、动作反归一化、env step。
4. 对照源码从这几个文件读：`extracted/repo/examples/LIBERO/eval_files/eval_libero.py`、`extracted/repo/examples/LIBERO/eval_files/model2libero_interface.py`、`extracted/repo/deployment/model_server/tools/websocket_policy_server.py`、`extracted/repo/starVLA/model/framework/QwenOFT.py`、`extracted/repo/starVLA/model/modules/action_model/MLP_ActionHeader.py`。

## 本地包不包含的大文件

为了避免盲目下载 18GB+ 模型权重和 16GB conda env，本包没有包含：

- Qwen3-VL base model: `repo/playground/Pretrained_models/Qwen3-VL-4B-Instruct`，约 8.3GB。
- StarVLA checkpoint `.pt`: `checkpoints/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt`，约 9.2GB。
- LIBERO 训练数据 parquet/mp4: `data/libero`，约 1.8GB。
- conda envs: `envs/`，约 16GB。

这些路径和大文件清单在 `archives/REMOTE_ARTIFACT_INDEX.md` 和 `archives/REMOTE_MANIFEST.txt` 里有记录。若从 GitHub clone，本地不会自动包含 `archives/*.tar.gz`。

## Trace 开关

trace 默认关闭。远端运行时可用这些环境变量开启：

```bash
export STARVLA_TRACE=1
export STARVLA_TRACE_FIRST_N=2
export STARVLA_TRACE_EVERY=50
export STARVLA_TRACE_MAX_ITEMS=3
export STARVLA_TRACE_MAX_DEPTH=4
export STARVLA_TRACE_MAX_STATS_NUMEL=64
```

trace 行都以 `[STARVLA_TRACE]` 开头，是 JSON 摘要，主要记录 shape、dtype、device、min/max/mean、少量 sample 和接口字段。
