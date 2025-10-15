---
title: nanochat
created: 2025-10-15
tags:
---
https://github.com/karpathy/nanochat

- 全流程集成：
    
    - 使用Rust实现训练分词器；
        
    - 在FineWeb数据集上预训练Transformer语言模型，并评估CORE指标；
        
    - 中期训练阶段引入SmolTalk数据集，涵盖用户-助手对话、多选问答及工具使用；
        
    - 执行监督微调（SFT），并在ARC-E/C、MMLU、GSM8K、HumanEval等任务上评估性能；
        
    - 可选使用GRPO算法在GSM8K上进行强化学习训练；
        
    - 推理引擎支持KV缓存、prefill/decode分离、工具调用等现代LLM推理特性。
        
- 轻量化设计：整个系统仅约8000行代码，依赖极少，强调代码可读性与模块可替换性，便于教学与实验。