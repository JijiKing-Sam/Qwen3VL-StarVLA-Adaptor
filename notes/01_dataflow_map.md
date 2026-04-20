# Dataflow Map

当前实际复现跑的是官方 `Qwen3-VL-OFT-LIBERO-4in1`，框架类是 `Qwenvl_OFT`，不是 `QwenAdapter`。

## 推理链路

1. `eval_libero.py`
   LIBERO 环境 reset，读取 `agentview_image`、`robot0_eye_in_hand_image`、eef pose、quat、gripper state，构造 `example_dict = {"image": [primary, wrist], "lang": task_description}`。

2. `model2libero_interface.py`
   `ModelClient.step()` 将两张图 resize 到 `224x224`，按 action chunk 频率请求 server。server 返回 `normalized_actions` 后，使用 checkpoint 的 `dataset_statistics.json` 反归一化为 7 维动作。

3. `websocket_policy_client.py`
   将 `{"examples": [example], "do_sample": False, "use_ddim": True, "num_ddim_steps": 10}` 用 msgpack 发给 policy server。

4. `websocket_policy_server.py`
   接收请求，调用 `policy.predict_action(**msg)`，返回 `{"status": "ok", "data": {"normalized_actions": ...}}`。

5. `QwenOFT.py`
   将 numpy 图像转 PIL，拼接 action token prompt，构造 Qwen3-VL 输入，取最后层 hidden states，再 gather 8 个 action token 的 hidden states。

6. `MLP_ActionHeader.py`
   `L1RegressionActionHead.predict_action()` 把 `[B, 8, 2560]` action token hidden states 送入 MLP，输出 `[B, 8, 7]` normalized actions。

7. `model2libero_interface.py`
   normalized actions clip 到 `[-1, 1]`，第 7 维 gripper 二值化，再按 `min/max` 反归一化，得到 `world_vector`、`rotation_delta`、`open_gripper`。

8. `eval_libero.py`
   将 gripper 从 open/close 映射到 LIBERO 控制格式，拼接成 `delta_action = [x, y, z, rx, ry, rz, gripper]`，执行 `env.step(delta_action.tolist())`。

## 关键 trace label

- `eval.step.observation`: LIBERO 原始观测图、wrist 图、state、obs keys。
- `model_client.step.resized`: client 侧 resize 后输入。
- `websocket.client.send` / `websocket.server.recv`: 网络接口请求体。
- `qwen_oft.predict.qwen_inputs`: Qwen3-VL 输入摘要。
- `qwen_oft.predict.qwenvl_outputs`: hidden state 数量、last hidden shape。
- `qwen_oft.predict.action_queries`: gather 后的 action query tensor，通常 `[1, 8, 2560]`。
- `action_head.l1.output`: action head 输出，通常 `[1, 8, 7]`。
- `model_client.unnormalize`: normalized action 到 raw action 的反归一化。
- `eval.step.env_result`: LIBERO step 结果，包括 reward、done、info。

## 对照日志

- Server 侧模型链路：`../extracted/logs/trace_smoke_20260411_215258/server_trace.log`
- Eval/client/env 链路：`../extracted/logs/trace_smoke_20260411_215258/eval_trace.log`
- 完整 `libero_goal` 50 trials/task 结果：`../extracted/logs/libero_eval_20260411_054831/eval_goal50.log`
