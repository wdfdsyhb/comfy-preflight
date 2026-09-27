# comfy-preflight 中文说明

**先体检，再下载。** ComfyUI 工作流静态预检工具：不装任何东西、不启动
ComfyUI，就能告诉你一个 workflow.json——

- 缺哪些自定义节点包（内置核心节点清单 + 精选热门包映射 + 在线 Registry 查询）
- 需要哪些模型文件、各自该放进 `ComfyUI/models/` 的哪个子目录
- 显存粗估 + 判定：你的显卡到底跑不跑得动

## 用法

```bash
python comfy_preflight.py workflow.json --vram 6        # 6GB 卡
python comfy_preflight.py workflow.json --markdown > report.md
```

- 支持两种格式：UI 格式（画布"保存"的 json）和 API 格式（"Save (API Format)"）
- `--offline` 跳过在线查询；`--family sdxl|flux|...` 手动指定模型家族
- 单文件、纯标准库，Python ≥ 3.9，无需安装

## 显存估算怎么算

按模型家族基准权重（SDXL≈6.9GB、FLUX fp16≈23.8GB…），识别 GGUF/fp8/NF4
量化折扣（看文件名或加载器的 `weight_dtype`），加上 latent 分辨率/批量的激活
内存和 LoRA 开销，给出**区间**：`low`（激进 offload）到 `high`（全驻留）。
判定留了余量：TIGHT ≈ 需要 lowvram，NO ≈ 权重都塞不下。

是经验公式不是基准测试，实际占用受注意力后端、ComfyUI 版本影响——把区间
当数量级参考即可。

## 限制（v0.1）

- 核心节点清单可能落后于最新版 ComfyUI（会标注"可能是新核心节点"）
- 在线查询是关键词搜索，给出的包是候选不是定论（精选映射表命中才是权威）
- 多阶段工作流（高清修复、放大链）目前按最大单阶段估算

详细文档见 [README.md](README.md)。MIT 许可。
