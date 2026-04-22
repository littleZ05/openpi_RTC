# OpenPI_RTC 真机使用手册_zrm

适用仓库：`/share/project/Ruimeng/openpi_RTC`  
适用场景：`JAX` 模式下 `pi05 + RTC` 真机推理联调

## 1. 目的

本文档说明如何基于当前仓库启动 `pi05 + RTC` 真机推理链路，包括：

- 服务端模型加载与 RTC server 启动
- 机器人端环境准备与 client 启动
- `orange` 与 `fruit` 两套任务的启动方式
- 联调顺序
- 常见问题排查
- 上真机前的安全检查项

当前已验证范围：

- `pi05` JAX checkpoint 可正常加载
- `RTC websocket server` 可正常启动
- 离线 parquet 回放可完整跑通 `obs -> infer -> action chunk`
- 当前验证到的输出动作形状为 `(50, 14)`

## 2. 链路说明

整条链路分为两端。

服务端：

- 加载 `pi05` JAX checkpoint
- 启动 RTC websocket server
- 接收机器人端发来的观测
- 调用 `policy.infer(obs)`
- 返回 action chunk

机器人端：

- 采集图像和机器人状态
- 打包观测发送给服务端
- 接收 action chunk
- 将动作送入控制执行

当前仓库对应入口：

- RTC 服务端：`examples/real_robot_rtc/serve_rtc_pi0.py`
- websocket server：`examples/real_robot_rtc/websocket_server.py`
- websocket client 参考：`examples/real_robot_rtc/websocket_client.py`
- 离线测试参考：`examples/real_robot_rtc/test_rtc_main.py`

## 3. 环境要求

服务端机器需要满足：

- 可以访问仓库 `/share/project/Ruimeng/openpi_RTC`
- 可以访问模型 checkpoint
- 可以使用参考运行时：
  - `/share/project/wujiling/openpi/.venv`
  - `/share/project/wujiling/envs/.uvpy/python/cpython-3.11.13-linux-x86_64-gnu`

机器人端需要满足：

- 与服务端处于同一网络
- 可以 `ping` 通服务端 IP
- 已配置好 CAN、相机及运行环境
- 能正常启动机器人 client 脚本

## 4. 当前推荐运行时

不要直接使用：

```bash
/share/project/Ruimeng/openpi_RTC/.venv
```

原因：

- 当前 `.venv` 运行 RTC 链路时会遇到 `PyAV` 动态库问题
- 典型报错为 `libavformat.so.61` 缺失

当前推荐运行方式：

- 使用 `/share/project/wujiling/openpi/.venv` 作为运行时
- 使用当前仓库 `openpi_RTC` 作为实际代码来源

服务端统一环境变量如下：

```bash
source /share/project/wujiling/openpi/.venv/bin/activate
export ROOT=/share/project/wujiling/envs/.uvpy/python/cpython-3.11.13-linux-x86_64-gnu
export PYTHONHOME="$ROOT"
export PYTHONPATH=/share/project/Ruimeng/openpi_RTC/src:/share/project/Ruimeng/openpi_RTC/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}
```

## 5. 模型与配置

当前已确认可用的配置名：

- `pi05_agilex_orange`
- `pi05_agilex_fruit`

配置定义位置：

- `src/openpi/training/config.py`

当前已确认可用的 `orange` checkpoint：

```bash
/share/project/wujiling/checkpoints/finetune/pi05_pnp_orange/20000
```

当前仓库 `serve_rtc_pi0.py` 中默认写了 `fruit` 的配置占位，但路径是另一台机器上的：

```bash
/home/admin123/Desktop/pi05_fruit/20k
```

如果要上 `fruit` 真机，请先确认你实际可用的 `fruit` checkpoint 路径，再替换下面命令中的 `--policy.dir`。

说明：

- 模型内部配置并不等于最终控制输出维度
- 当前验证过的 `orange` 返回给机器人端的是 `(50, 14)` 的 action chunk

## 6. 服务端启动

建议在算力机上使用 `tmux`。

进入目录：

```bash
cd /share/project/Ruimeng/openpi_RTC/examples/real_robot_rtc
```

加载环境：

```bash
source /share/project/wujiling/openpi/.venv/bin/activate
export ROOT=/share/project/wujiling/envs/.uvpy/python/cpython-3.11.13-linux-x86_64-gnu
export PYTHONHOME="$ROOT"
export PYTHONPATH=/share/project/Ruimeng/openpi_RTC/src:/share/project/Ruimeng/openpi_RTC/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}
```

### 6.1 Orange 服务端启动命令

```bash
python serve_rtc_pi0.py --port 8000 policy:checkpoint --policy.config pi05_agilex_orange --policy.dir /share/project/wujiling/checkpoints/finetune/pi05_pnp_orange/20000
```

### 6.2 Fruit 服务端启动命令

将下面命令中的 `<fruit_checkpoint_dir>` 替换为你实际可用的 checkpoint 根目录。该根目录下应包含 `params/` 和 `assets/`。

```bash
python serve_rtc_pi0.py --port 8000 policy:checkpoint --policy.config pi05_agilex_fruit --policy.dir <fruit_checkpoint_dir>
```

如果你确认当前环境里就要用 `serve_rtc_pi0.py` 中写死的那份路径，则命令为：

```bash
python serve_rtc_pi0.py --port 8000 policy:checkpoint --policy.config pi05_agilex_fruit --policy.dir /home/admin123/Desktop/pi05_fruit/20k
```

启动成功后应看到类似日志：

```text
INFO:root:Loading model...
INFO:absl:Restoring checkpoint ...
INFO:websockets.server:server listening on 0.0.0.0:8000
```

说明服务端已正常监听 `8000` 端口。

## 7. 机器人端启动

机器人端先完成基础初始化：

```bash
bash ~/info.sh
conda activate dora
```

确认网络：

```bash
ping <服务端IP>
```

启动机器人 client：

```bash
python /home/agilex/robobrain-robotics/client_move_from_server_eef.py
```

注意：

- 机器人 client 需要连接到服务端实际 IP 和端口 `8000`
- 如果脚本中 host/port 写死，需要先检查脚本配置
- `orange` 和 `fruit` 如果共用同一个 client，需要额外确认 prompt、动作语义和任务选择逻辑是否同步切换

## 8. 服务端输入输出格式

当前 RTC server 期望客户端发送原始消息，核心字段包括：

- `state`
- `images`
- `eef_pose`
- `prompt`

其中 `images` 需要包含：

- `cam_high`
- `cam_left_wrist`
- `cam_right_wrist`

服务端最终重构出的模型输入为：

- 三路图像 tensor
- `state[0]` 转为 `float32`
- 一个 prompt

重要说明：

- 当前 RTC server 实际送入模型的是 `state`
- `eef_pose` 当前主要用于接口兼容，不是当前模型主输入

## 9. 联调推荐顺序

建议按以下顺序联调：

1. 启动服务端 `serve_rtc_pi0.py`
2. 确认服务端已监听端口
3. 在机器人端完成 `info.sh` 和 `conda activate dora`
4. 确认网络联通
5. 启动机器人 client
6. 观察服务端日志是否持续出现观测重建与推理成功信息
7. 先低速、低风险测试
8. 再逐步进入真实任务动作测试

服务端正常联调日志通常会包含：

```text
Observation reconstructed
Infer result type=<class 'dict'>, keys=['actions', 'policy_timing']
Sending inference result to client
```

## 10. 动作与状态约定

上真机前必须确认以下约定与机器人控制端完全一致：

- `eef` 与 `qpos/state` 的定义
- 左右臂顺序
- gripper 维度位置
- 14 维动作每一维的物理含义

已知注意事项：

- 当前约定是“先右后左”
- 若机器人控制端按不同顺序解释动作，机械臂会直接错位
- 当前 RTC server 用 `state` 作为模型输入，不应误以为是 `eef_pose`

## 11. prompt 注意事项

当前 `examples/real_robot_rtc/websocket_server.py` 中，服务端会在重构观测时写入 prompt。

这意味着：

- 客户端传入的 prompt 可能被覆盖
- 上真机前需要确认最终以谁为准
- 如果任务从 `orange` 切到 `fruit` 或其他任务，需要检查 prompt 是否同步切换

## 12. infer 与 replay 的区分

当前仓库存在两类服务方式：

HTTP 风格：

- `examples/real_robot/serve_robobrain_robotics_pi0.py`

RTC websocket 风格：

- `examples/real_robot_rtc/serve_rtc_pi0.py`

本文档描述的是 RTC websocket 链路。  
不要将 HTTP 的 `infer/replay` 端口逻辑直接套到 RTC websocket 方案上。

## 13. 常见问题排查

服务端启动失败：

- 检查是否使用了推荐运行时
- 检查 `PYTHONPATH` 是否指向当前仓库
- 检查 checkpoint 路径是否存在且包含 `params/` 和 `assets/`

报 `libavformat.so.61` 或类似动态库错误：

- 说明运行时不对
- 切换到 `/share/project/wujiling/openpi/.venv` 方案

服务端已启动但机器人端无法连接：

- 检查服务端 IP
- 检查端口 `8000`
- 检查网络连通性
- 检查 client 配置中的 host/port

服务端出现 `Observation reconstructed` 但没有 `Infer result`：

- 问题多半在模型推理层
- 检查输入格式、checkpoint、config 是否匹配

服务端出现 `Infer result` 但机器人不动作：

- 问题多半在机器人 client 或控制执行层
- 检查动作维度解释、左右臂顺序、执行接口

## 14. 安全检查清单

上真机前至少确认以下项目：

- 已先完成离线链路验证
- 机器人端与服务端网络稳定
- 端口、IP、checkpoint、config 均已确认
- 动作维度与机器人控制语义一致
- 左右臂顺序确认无误
- gripper 维度确认无误
- prompt 与任务一致
- 初次动作测试采用低风险姿态与空场景
- 有人工急停与接管条件
- 先小幅测试，再逐步放开动作

## 15. 最小可执行命令

### 15.1 Orange

服务端：

```bash
cd /share/project/Ruimeng/openpi_RTC/examples/real_robot_rtc

source /share/project/wujiling/openpi/.venv/bin/activate
export ROOT=/share/project/wujiling/envs/.uvpy/python/cpython-3.11.13-linux-x86_64-gnu
export PYTHONHOME="$ROOT"
export PYTHONPATH=/share/project/Ruimeng/openpi_RTC/src:/share/project/Ruimeng/openpi_RTC/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}

python serve_rtc_pi0.py --port 8000 policy:checkpoint --policy.config pi05_agilex_orange --policy.dir /share/project/wujiling/checkpoints/finetune/pi05_pnp_orange/20000
```

机器人端：

```bash
bash ~/info.sh
conda activate dora
python /home/agilex/robobrain-robotics/client_move_from_server_eef.py
```

### 15.2 Fruit

服务端：

```bash
cd /share/project/Ruimeng/openpi_RTC/examples/real_robot_rtc

source /share/project/wujiling/openpi/.venv/bin/activate
export ROOT=/share/project/wujiling/envs/.uvpy/python/cpython-3.11.13-linux-x86_64-gnu
export PYTHONHOME="$ROOT"
export PYTHONPATH=/share/project/Ruimeng/openpi_RTC/src:/share/project/Ruimeng/openpi_RTC/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}

python serve_rtc_pi0.py --port 8000 policy:checkpoint --policy.config pi05_agilex_fruit --policy.dir <fruit_checkpoint_dir>
```

如果就是使用当前代码里默认提到的路径，则替换为：

```bash
python serve_rtc_pi0.py --port 8000 policy:checkpoint --policy.config pi05_agilex_fruit --policy.dir /home/admin123/Desktop/pi05_fruit/20k
```

机器人端：

```bash
bash ~/info.sh
conda activate dora
python /home/agilex/robobrain-robotics/client_move_from_server_eef.py
```

## 16. 当前手册的可信范围

本文档中，以下内容是已实际验证过的：

- 当前仓库 RTC server 可启动
- `pi05_agilex_orange` 的 JAX checkpoint 可加载
- 使用真实 parquet 样本可成功返回 `(50, 14)` action chunk
- websocket RTC loopback 能连续运行

以下内容仍需在真机环境中二次确认：

- `fruit` 任务实际 checkpoint 路径
- 机器人 client 的 host/port 和字段映射
- `eef/qpos` 与 14 维动作的控制语义
- 真机动作执行安全边界
- prompt 是否应由服务端写死
