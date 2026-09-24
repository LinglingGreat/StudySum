---
title: LLaDA 系列模型全景
created: 2026-09-23
tags:
  - 扩散语言模型
  - dLLM
  - 多模态
type: 模型调研
institution:
  - 人大高瓴 GSAI
  - 蚂蚁 inclusionAI
---

## 一句话总结

LLaDA 是掩码扩散语言模型（dLLM）家族：人大高瓴李崇轩组起源（与蚂蚁合作），2.x 主线由蚂蚁 inclusionAI 接手，人大线仍单独在发（LLaDA-o、MoE v2）。截至 2026-09-23 公开 13 个型号（文本 8、多模态 5），**无公开 LLaDA2.5**，最新文本版是 2.2；**全系列无语音输入**。

> 日期口径：优先 arXiv 日期，否则 HF 仓库创建日期（私有转公开时会比官宣早几天）。数据于 2026-09-23 从 HF API 核实。

## 纯文本

| 时间 | 模型 | 规模 | 发布方 | 要点 |
|---|---|---|---|---|
| 2025-02 | LLaDA-8B-Base / Instruct | 8B dense | GSAI-ML | 首个从零训练的大规模掩码扩散 LM（arXiv 2502.09992） |
| 2025-05 | LLaDA 1.5 | 8B | GSAI-ML | VRPO 偏好优化（扩散版 DPO 降方差，2505.19223） |
| 2025-09 | LLaDA-MoE-7B-A1B-Base / Instruct | 7B-A1B | inclusionAI | 首个 MoE dLLM，从零训练（2509.24389）；10 月加 -TD（轨迹蒸馏加速） |
| 2025-10 | LLaDA2.0-mini / flash-preview | 16B / 100B | inclusionAI | 预览版 |
| 2025-11 | LLaDA2.0-mini / flash | 16B-A1B / 100B-A6B | inclusionAI | 首个 100B dLLM（2512.15745）；12 月出 -CAP 加速版（flash-CAP ~535 TPS） |
| 2026-02 | LLaDA2.1-mini / flash | 16B / 100B | inclusionAI | Token editing 可纠错、Speed/Quality 双模式、100B dLLM 上做 RL（2602.08676） |
| 2026-07 / 09 | LLaDA2.2-flash / mini | 100B / 16B | inclusionAI | Levenshtein 编辑（DELETE/INSERT）、128K 上下文、agentic RL（L-EBPO） |
| 2026-08 | LLaDA-MoE-v2-30B-A3B-Base / Instruct | 30B-A3B | GSAI-ML | 从零 23.5T tokens（2608.03457） |

命名规律：mini = 16B 级、flash = 100B 级；CAP / TD 后缀 = 推理加速版。

## 多模态

| 时间 | 模型 | 规模 | 发布方 | 能力 |
|---|---|---|---|---|
| 2025-05 | LLaDA-V | 8B | GSAI-ML | 图文理解（LLaDA-8B + 视觉指令微调，2505.16933） |
| 2026-03 | LLaDA-o | ~15B（按权重体积估） | GSAI-ML | omni：理解 + 文生图 + 编辑，长度自适应（2603.01068） |
| 2026-04 | LLaDA2.0-Uni（5 月 FP8） | 16B MoE | inclusionAI | 统一理解与生成，SigLIP-VQ 离散视觉 token（2604.20796） |
| 2026-09 | LLaDA-Image / -Turbo（+FP8） | 6B | inclusionAI | 文生图 + 编辑，Turbo 4 步（Twin-DMD 蒸馏，2609.03796） |
| 2026-09 | LLaDA-UI | ~16.7B MoE | inclusionAI | 块扩散 GUI agent，底座 LLaDA2.0-mini-base（2609.13287） |

### 图像理解对比（LLaDA2.0-Uni 论文 Table 2，作者自报）

| 模型 | MMStar | MMBench-CN | HallusionBench | RealWorldQA | OCRBench |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B（参照） | 63.9 | 83.4 | 51.9 | 68.5 | 84.2 |
| LLaDA2.0-Uni | 64.1 | 81.2 | 50.2 | 66.7 | 75.7 |
| LLaDA-o | 58.0 | 69.9 | 47.4 | 60.8 | 74.6 |
| LLaDA-V | 60.1 | 70.1 | 39.2 | 63.2 | 63.2 |

## 选型结论：图文（+语音）输入、文本输出的聊天场景

- LLaDA 系列内只有 **LLaDA2.0-Uni** 可选，但不推荐直接用于聊天：
  - 无语音；dLLM 做语音理解只有第三方 [DIFFA](https://github.com/NKU-HLT/DIFFA)（南开，基于 LLaDA-8B，只有音频）
  - 官方接口是单轮 `understand_image(image_tokens, h, w, question=...)`，无 system prompt / 多轮历史，需自己拼模板并验证
  - 上下文 8K（SFT 扩到 16K），底座停在 LLaDA2.0-mini
  - 理解能力 ≈ Qwen2.5-VL-7B，OCR 差 8 分多
- 做产品：蚂蚁 [Ming-flash-omni-2.0](https://huggingface.co/inclusionAI/Ming-flash-omni-2.0)（2026-02，100B-A6B，MIT，文本/图像/音频/视频输入 → 文本/语音输出），自回归而非扩散。
- 研究 dLLM 聊天：LLaDA2.0-Uni + 多轮/人设 SFT；语音先走 ASR 级联，后续再参照 DIFFA 接音频 encoder。

### LLaDA2.0-Uni 做聊天：代码层核实（2026-09-23，HF `modeling_llada2uni_moe.py` + GitHub 源码）

- **两套对话格式不一致**：tokenizer 自带 Ling 风格 chat_template（`<role>SYSTEM</role>…detailed thinking off<|role_end|>`，支持多轮 / system / tools）；但官方 `understand_image()` 手拼 `<role>SYSTEM</role> {system} <role>HUMAN</role>{图+问题}<role>ASSISTANT</role>`，**无 `<|role_end|>`、system 写死 "You are a multimodal understanding assistant."、单轮**。论文只说 SFT 含 single/multi-turn，没给格式 → 多轮该用哪套需实测
- **变长输出没问题**：`generate_bd` 是块扩散（block 32，块间因果、块内双向），EOS 确认即提前停止；块从左到右生成，可按 32 token 粒度流式（现有代码未实现）
- **效率**：不开 Sprint 时每个去噪步都重算整个前缀（无 KV cache），多轮历史越长越慢；Sprint = KV 复用 + 剪枝；已有 SGLang Omni 支持
- **图像 token 开销**：patch 14 × merge 2 = 32px/token；单图上限 800×800 → ≤625 token，多图各 448×448 → 196 token
- **上下文**：config `max_position_embeddings` 8192，论文 SFT 阶段扩到 16K
- **文本能力未知**：SFT 80B token、纯文本:多模态 = 1:5，论文**没报任何纯文本 benchmark**，也没和 LLaDA2.0-mini 对比
- 架构 = LLaDA2.0-mini（20 层、256 专家选 8），词表 157,184 + 16,384 个图像 token

## 部署框架（2026-09-24 在 B300 上实测）

| 模型 | 框架 | 备注 |
|---|---|---|
| LLaDA2.0 / 2.1 | SGLang（0.5.20 已含 `llada2.py`），`--dllm-algorithm LowConfidence / JointThreshold` | 2.1 用 JointThreshold（M2T + T2T 编辑） |
| LLaDA2.2 | **只能用官方 remote code 的 `model.generate`** | SGLang 0.5.20 没有 DELETE/SPLIT 编辑解码；架构名相同能加载但解码错误 |
| LLaDA2.0-Uni | sglang-omni（`sgl-omni serve`，OpenAI 兼容） | 🔴 预处理会丢弃 system 消息 |
| 任意 LLaDA | ❌ vLLM | 不支持扩散模型 |

部署细节与冒烟结果：`/Users/liling/Career/xinchen/docs/模型部署/LLaDA系列-B300部署与聊天冒烟_20260924.md`

## 参考资料

- HF 组织：[inclusionAI](https://huggingface.co/inclusionAI)、[GSAI-ML](https://huggingface.co/GSAI-ML)
- [inclusionAI/LLaDA2.X](https://github.com/inclusionAI/LLaDA2.X)、[LLaDA2.0-Uni](https://github.com/inclusionAI/LLaDA2.0-Uni)
- [LLaDA2.0-Uni 论文](https://arxiv.org/html/2604.20796)
- 库内相关：[HF 周榜 W17（LLaDA2.0-Uni）](../../../News/papers/HuggingFace_2026-W17.md)、[HF 周榜 W32（LLaDA MoE v2）](../../../News/papers/HuggingFace_2026-W32.md)
