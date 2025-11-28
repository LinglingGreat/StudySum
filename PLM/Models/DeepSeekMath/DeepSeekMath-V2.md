---
title: DeepSeekMath-V2
created: 2025-11-28
tags:
  - 数学大模型
type: 论文
papername:
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2025
institution:
  - DeepSeek
---

## 论文基本信息

标题：

作者：

链接：

代码：

框架图：


## 背景
论文试图解决什么问题？这是否是一个新的问题？

这篇文章要验证一个什么科学假设？

论文中提到的解决方案之关键是什么？


## 相关研究
有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？



## 核心亮点

该论文提出了一套包含**证明生成器（Generator）**、**证明验证器（Verifier）**和**[元验证器](https://zhida.zhihu.com/search?content_id=758529994&content_type=Answer&match_order=1&q=%E5%85%83%E9%AA%8C%E8%AF%81%E5%99%A8&zhida_source=entity)（Meta-Verifier）**的完整框架。通过强化学习（RL），模型不仅学会了生成证明，还具备了自我验证和自我修正的能力。关键创新包括：

1. **验证器训练**：训练模型识别证明中的逻辑漏洞并打分。
2. **元验证机制**：引入元验证器来评估验证者提出的批评是否合理，从而减少验证者的幻觉（即对正确步骤的错误批评）。
3. **[自验证强化](https://zhida.zhihu.com/search?content_id=758529994&content_type=Answer&match_order=1&q=%E8%87%AA%E9%AA%8C%E8%AF%81%E5%BC%BA%E5%8C%96&zhida_source=entity)**：在生成器的训练中加入自省环节，若模型能诚实地指出自身证明的缺陷，将获得奖励。
4. **推理时计算扩展**：利用训练好的验证器指导推理阶段的搜索（Sequential Refinement 和 High-Compute Search）。


- **1分**：完全正确，逻辑严密，所有步骤均有正当理由。
- **0.5分**：整体逻辑成立，但存在细节遗漏或轻微错误。
- **0分**：存在根本性缺陷，包含致命逻辑错误或关键缺失。
  
作者：0xC001  
链接：https://www.zhihu.com/question/1977490844794782603/answer/1977515161389655161  
来源：知乎  
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

## 实验
论文中的实验是如何设计的？

用于定量评估的数据集是什么？代码有没有开源？

论文中的实验及结果有没有很好地支持需要验证的科学假设？



## 未来方向



## 主要收获


## 参考资料
