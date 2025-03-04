---
title: README
created: 2024-06-17
tags:
  - 推理加速
---
## 推理加速

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

[Towards 100x Speedup: Full Stack Transformer Inference Optimization](https://yaofu.notion.site/Towards-100x-Speedup-Full-Stack-Transformer-Inference-Optimization-43124c3688e14cffaf2f1d6cbdf26c6c)

[OSDI 24 Serverless LLM：性能提升200倍](https://mp.weixin.qq.com/s/DoxSI5M-jcdlSg000VEIng)

[DeepSpeed Inference全栈优化，延迟降低7.3倍，吞吐提升1.5倍](https://mp.weixin.qq.com/s/fvJaREiR6FGuwWBRFafvbw)

[[TensorRT-LLM][5w字]🔥TensorRT-LLM 部署调优-指北](https://zhuanlan.zhihu.com/p/699333691)

##  参数量、计算量、显存等

[【Transformer】参数量、计算量、显存等分析](https://mp.weixin.qq.com/s/zZh1CaeozXBffImxnBTPtg)

[语言模型的训练时间：从估算到 FLOPs 推导](https://zhuanlan.zhihu.com/p/646905171)

[大模型训练需要花费多长时间：FLOPs的简单计算方法及calflop开源实现](https://mp.weixin.qq.com/s/nB-ldVgWJTJhwI-f7rO7IQ)

https://huggingface.co/spaces/Jellyfish042/UncheatableEval
压缩能力榜单

## 训练


[【分布式训练技术分享七】聊聊字节 AML 万卡工作 MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs - 知乎](https://zhuanlan.zhihu.com/p/684619370)

[OneFlow —— 让每一位算法工程师都有能力训练 GPT - 知乎](https://zhuanlan.zhihu.com/p/371499074?utm_psn=1749361005107462144)



## DeepSeek开源项目

[FlashMLA](FlashMLA/FlashMLA.md)

[DeepEP](DeepEP/DeepEP.md)

[DeepGEMM](DeepGEMM/DeepGEMM.md)

[DualPipe&EPLB](DualPipe/DualPipe&EPLB.md)

[DeepSeek-V3 / R1 推理系统概览](https://zhuanlan.zhihu.com/p/27181462601)

**优化目标是：更大的吞吐，更低的延迟。**

