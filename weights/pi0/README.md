# 在此放入新的 PI0 权重导出

此目录是 HarnessVLA 部署入口默认使用的检查点位置。公开仓库不含权重。请将已验证且有权使用的新权重完整导出放到本目录，并保留原有相对结构，例如：

```text
weights/pi0/
  config.json
  model.safetensors
  policy_preprocessor.json
  policy_postprocessor.json
  policy_preprocessor_step_6_normalizer_processor.safetensors
  policy_postprocessor_step_0_unnormalizer_processor.safetensors
  tokenizer/
    tokenizer.json
    tokenizer_config.json
```

`train_config.json` 可选，但应随新权重保存以追溯训练数据。所有权重内容被 `.gitignore` 排除，只有本说明文件保留在项目中。复制后先运行 `python3 -m harnessvla.pi0_deploy check`，再考虑服务端加载；`preview-client` 永不发送动作。
