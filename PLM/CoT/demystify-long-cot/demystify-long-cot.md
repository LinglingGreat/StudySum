---
title: demystify-long-cot
created: 2025-03-03
tags:
  - longcot
type: 论文
papername: Demystifying Long Chain-of-Thought Reasoning in LLMs
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2025
institution:
  - 清华
  - CMU
---

## 论文基本信息

标题：Demystifying Long Chain-of-Thought Reasoning in LLMs

作者：Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig,  Xiang Yue

链接：

代码： https://github.com/eddycmu/demystify-long-cot

框架图：


## 背景

long CoT的定义：

![](img/Pasted%20image%2020250303194927.png)





## 相关研究




## 核心亮点



## 实验

基座模型：Llama-3.1-8B和Qwen2.5 -7B-Math

训练数据：SFT和RL都用的是MATH的7,500条训练样本的prompt

评估benchmark：in-domain (MATH-500 test set)和out-of-domain (AIME 2024, TheoremQA, MMLU-Pro-1k).

### Impact of SFT on Long CoT

SFT数据来自拒绝采样：
- long CoT数据蒸馏自QwQ-32B-Preview
- short CoT数据蒸馏自Qwen2.5-Math-72B-Instruct

PPO训练：
- PPO算法，规则验证结果
- 采取余弦长度缩放奖励和重复惩罚

基座模型是Llama-3.1-8B

可以看到用long CoT数据SFT后模型能取得更高的性能上限，并且更容易通过RL进一步提升性能。

![](img/Pasted%20image%2020250303195412.png)

两种获取long CoT数据的方法：（1）Construct：通过提示短COT模型生成原始动作并顺序组合来构建长的COT轨迹； （2）Emergent：从现有的长COT模型中提取long CoT轨迹，这些模型已经有long CoT模式。

实验证明，高质量的涌现的的long CoT模式会有更好的泛化性和RL收益。

![](img/Pasted%20image%2020250303200428.png)

### Impact of Reward Design on Long CoT

探讨reward的设计对CoT长度和性能的影响。

Classic Reward：基于规则的reward，正确答案给1分。

微调时设置上下文长度为16K，使用Classic Reward。

我们观察到，这两种模型在训练过程中都增加了CoT的长度，最终达到了上下文窗口限制。由于COTS超过允许的窗户尺寸，这导致训练准确性下降。此外，不同的基本模型表现出不同的缩放行为。与QWEN-2.5-MATH-7B相比,Llama-3.1-8B长度的波动更大。

![](img/Pasted%20image%2020250303201210.png)



### Scaling up Verifiable Reward



## 未来方向



## 主要收获


## 参考资料
