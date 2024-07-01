---
title: README
created: 2024-06-17
tags:
  - 推理加速
---


[GitHub - microsoft/DeepSpeed-MII: MII makes low-latency and high-throughput inference possible, powered by DeepSpeed.](https://github.com/microsoft/DeepSpeed-MII)

[Break the Sequential Dependency of LLM Inference Using Lookahead Decoding | LMSYS Org](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)

[GitHub - FasterDecoding/Medusa: Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads](https://github.com/FasterDecoding/Medusa)

[语言大模型推理加速指南](https://mp.weixin.qq.com/s/B3TD2p_5HKoYkzzupLoUxQ)

[进我的收藏夹吃灰吧：大模型加速超全指南来了](https://mp.weixin.qq.com/s/4USwSMIiudFCdy9C5pN1dQ)


[[Prefill优化][万字]🔥原理&图解vLLM Automatic Prefix Cache(RadixAttention): 首Token时延优化 - 知乎](https://zhuanlan.zhihu.com/p/693556044)

[LLM后端推理引擎性能大比拼](https://mp.weixin.qq.com/s/dPd84P_VdKog8v2IcHDOrQ) 对比了vLLM、LMDeploy、MLC-LLM、TensorRT-LLM 和 Hugging Face TGI.

[大模型压缩量化方案怎么选？无问芯穹Qllm-Eval量化方案全面评估：多模型、多参数、多维度](https://mp.weixin.qq.com/s/BxMT1CZk35yMP8qnhoFNFw)
1. Weight-only量化可以显著加速decoding阶段，从而改善端到端延迟。
    
2. 关于prefill阶段，Weight-only量化可能实际上会增加延迟。
    
3. 随着批量大小和输入长度的增加，Weight-only量化所带来的加速效果逐渐减小。
    
4. 对于较大的模型，Weight-only量化提供了更大的益处，因为较大模型尺寸的内存访问开销显著增加。

[大模型训练及推理经典必读：FP8的what，why，how及其带来的机会？](https://mp.weixin.qq.com/s/Hyb04agEpGpwM4inn6CibA)


[月之暗面kimi底层推理系统方案揭秘](https://mp.weixin.qq.com/s/To97I4bU30fQssqkESTOGA)



