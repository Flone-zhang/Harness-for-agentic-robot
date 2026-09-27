# HarnessVLA
[论文展示页 / Project Page](https://flone-zhang.github.io/Harness-for-agentic-robot/)
HarnessVLA 是一个面向 Piper 机械臂的、证据驱动且离线优先的 VLA 任务执行框架。它将受约束的 Qwen 规划、PI0 技能执行、双相机视觉验证和 SQLite 事件记录组合成闭环，并提供无需硬件的模拟路径。

> [!WARNING]
> 真实机器人运动具有风险。默认示例关闭运动；在完成机械限位、急停、工作空间和现场监督验证前，不要将模型动作发送给机器人。

## 功能

- 白名单技能与前置条件约束，规划器不能直接输出关节动作。
- 每个技能都有动作预算、超时和视觉完成条件。
- Qwen 失败、未知视觉结论或黑帧不会被当作成功。
- 正常停止、超时和等待姿态停滞会走受控回位流程。
- SQLite、JSONL 和图像证据支持恢复、复核和重放。
- 离线 mock 模式无需模型、相机或机器人。

## 环境与安装

要求 Python 3.10 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

PI0 真机部署还需要与目标硬件匹配的 LeRobot、PyTorch、gRPC、RealSense 和 Piper 驱动环境；这些硬件依赖没有被强行加入基础安装。

## 快速开始

运行全部离线测试：

```bash
python -m unittest discover -s tests -v
```

运行 mock T1：

```bash
python -m harnessvla.cli --db runs/demo.sqlite3 start --config config/mock.json
# 或使用统一 YAML 入口
cp config/piper_harness.example.yaml config/piper_harness.yaml
python run_harness.py --config_path=config/piper_harness.yaml
```

测试扰动恢复、回放和离线 PI0 动作检查：

```bash
python -m harnessvla.cli --db runs/demo.sqlite3 start \
  --scenario config/scenario_empty_grasp.json
python -m harnessvla.cli --db runs/demo.sqlite3 replay <run_id>
python -m harnessvla.cli pi0-preview \
  --state-json config/pi0_preview_state.synthetic.json \
  --chunk-json config/pi0_preview_chunk.synthetic.json
python -m harnessvla.qwen_planner validate \
  --file config/t1_qwen_plan.example.json
```

## 统一配置

`run_harness.py` 读取一个 YAML 文件。先复制公开模板，再只在本地文件中填写设备参数和密钥：

```bash
cp config/piper_harness.example.yaml config/piper_harness.yaml
```

`config/piper_harness.yaml` 已被 `.gitignore` 排除。不要在示例文件、命令历史、日志或 issue 中提交 API Key。

可用运行模式：

| 模式 | 作用 | 是否发送电机命令 |
| --- | --- | --- |
| `mock_t1` | 离线任务编排与事件记录 | 否 |
| `pi0_check` | 检查 checkpoint 契约 | 否 |
| `pi0_server` | 在回环地址启动本地 PI0 服务 | 否 |
| `pi0_preview` | 读取状态/图像并记录预测 | 否 |
| `pi0_execute` | 有界执行单个技能 | 是，需显式开启 |
| `qwen_plan` | 仅请求受约束技能计划 | 否 |
| `qwen_pi0` | 规划、执行、视觉确认闭环 | 是，需显式开启 |

真实运动需要同时选择执行模式并设置 `pi0.motion_enabled: true`。这只是软件门控，不能替代硬件急停和现场风险评估。

## PI0 权重

大型 checkpoint 不随仓库分发。按照 [weights/pi0/README.md](weights/pi0/README.md) 放置文件，然后先执行只读检查：

```bash
python -m harnessvla.pi0_deploy check --hash-model
```

更完整的部署顺序见 [docs/pi0-deployment.md](docs/pi0-deployment.md)。

## 项目结构

```text
harnessvla/                 核心编排、规划、安全门控和部署代码
config/                     可公开的 mock、技能和 YAML 示例
tests/                      离线自动化测试
weights/pi0/README.md       checkpoint 放置说明（不含权重）
run_harness.py              统一 YAML 启动入口
```

## 当前边界

- 自动化测试主要覆盖编排、契约、故障路径和安全门控，不构成真机安全认证。
- 完成判定依赖双相机视觉；模型输出动作本身不代表任务成功。
- T3 扰动恢复当前只在模拟执行器中验证。
- 仓库不包含模型权重、真实运行记录、硬件标定或 API Key。

## 许可证

本仓库当前未附带开源许可证。在仓库所有者明确选择许可证之前，默认版权规则适用。

