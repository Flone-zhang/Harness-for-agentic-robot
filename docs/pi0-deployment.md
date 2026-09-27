# PI0 部署指南

本文描述从只读检查到受监督执行的建议顺序。所有命令都应在项目根目录运行。

## 1. 准备 checkpoint

将完整导出放入 `weights/pi0/`，目录结构见 `weights/pi0/README.md`。不要提交权重到 Git；发布模型时应使用专门的模型存储和独立许可。

先执行不连接机器人和相机的检查：

```bash
python -m harnessvla.pi0_deploy check --hash-model
python -m harnessvla.pi0_deploy check --processors --device cpu \
  --task "pick up the red block"
python -m harnessvla.pi0_deploy check --smoke-inference --device cuda:0 \
  --task "pick up the red block"
```

第二、三条命令需要兼容的 LeRobot/PyTorch 环境。

## 2. 启动本地推理服务

```bash
python -m harnessvla.pi0_deploy server \
  --device cuda:0 \
  --expected-model-sha256 <checkpoint-sha256>
```

服务必须只绑定回环地址。LeRobot 握手格式不应暴露到不可信网络。

## 3. 只读预览

在另一个终端运行：

```bash
python -m harnessvla.pi0_deploy preview-client \
  --task "pick up the red block" \
  --can-port <can-interface> \
  --top-camera-serial <top-camera-id> \
  --wrist-camera-serial <wrist-camera-id>
```

预览客户端读取状态和相机、记录预测与拒绝原因，但不使能机器人、不回零、也不发送动作。预览通过不代表动作安全。

## 4. 受监督执行前检查

- 核对 checkpoint 哈希、动作维度、绝对/相对动作语义和单位。
- 独立验证关节限位、夹爪范围、最大步长、控制频率和安全位。
- 验证相机身份、安装方向、时间同步和遮挡条件。
- 验证硬件急停，不依赖 Python 进程处理急停。
- 确认现场人员、清空工作空间，并从保守动作预算开始。
- 将 `config/piper_harness.example.yaml` 复制为被忽略的本地配置，填写本机参数；不要修改公开模板来保存密钥。

只有上述检查完成后，才应考虑将模式设为 `pi0_execute` 或 `qwen_pi0`，并显式设置 `pi0.motion_enabled: true`。

