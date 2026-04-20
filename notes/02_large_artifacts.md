# Large Artifacts

本地学习包为了便于快速阅读，没有拉取以下大文件。如果后续要在本地真正运行推理，需要单独下载。

## 远端路径

- 远端工作目录：`/root/autodl-tmp/starvla-official-qwen3vl-20260411`
- Qwen3-VL base model：`/root/autodl-tmp/starvla-official-qwen3vl-20260411/repo/playground/Pretrained_models/Qwen3-VL-4B-Instruct`
- StarVLA checkpoint：`/root/autodl-tmp/starvla-official-qwen3vl-20260411/checkpoints/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt`
- LIBERO training data：`/root/autodl-tmp/starvla-official-qwen3vl-20260411/data/libero`
- conda envs：`/root/autodl-tmp/starvla-official-qwen3vl-20260411/envs`

## 已下载到本地的替代元信息

- Checkpoint config/statistics：`../extracted/checkpoints/Qwen3-VL-OFT-LIBERO-4in1/`
- Dataset metadata：`../extracted/data/libero/`
- 大文件索引：`../archives/REMOTE_ARTIFACT_INDEX.md`
- 远端 manifest：`../archives/REMOTE_MANIFEST.txt`

## 如果之后要全量下载

优先单独下载 checkpoint `.pt` 和 Qwen3-VL base model，不建议下载 conda env。conda env 应在本地重建，因为跨机器复制环境通常不可移植。

如果要下载大文件，建议另开一个目录，比如：

```text
F:\starvla-qwen3vl-repro-study-20260412\large_artifacts
```

再用 SFTP/rsync 按需拉取，不要和当前源码学习包混在一起。
