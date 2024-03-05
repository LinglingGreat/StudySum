---
title: ResonanceRoPE
created: 2024-03-04
tags:
  - 长文本
type: 论文
papername: Resonance RoPE Improving Context Length Generalization of Large Language Models
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - 华为诺亚方舟
  - Mila
  - 蒙特利尔大学DIRO
---

## 论文基本信息

标题：Resonance RoPE: Improving Context Length Generalization of Large Language Models

作者：

链接：http://arxiv.org/abs/2403.00071

代码：

框架图：


## 背景
这篇论文试图解决的问题是在大型语言模型（LLMs）中，特别是在配备了旋转位置嵌入（Rotary Position Embedding, RoPE）的模型中，训练短序列（train-short）与测试长序列（test-long）场景（TSTL）下的泛化挑战。在这种场景中，预训练模型在较短序列上学习到的位置嵌入在面对更长序列时（即超出分布，OOD）会出现性能下降。具体来说，论文提出了一种名为RESONANCE ROPE的新方法，旨在通过改进RoPE特征在OOD位置的插值来缩小TSTL场景下的泛化差距，从而显著提高模型性能，同时不增加在线计算成本。此外，论文还提出了一个新的合成基准测试POSGEN，用于在TSTL场景中对模型的行为进行细粒度分析。


## 相关研究
相关研究主要集中在以下几个方面：

1. **RoPE Position Encoding Scaling**:
    
    - 近期的研究集中在扩展大型语言模型（LLMs）的上下文窗口，特别是通过操作位置嵌入（PE），尤其是RoPE。主要策略包括嵌入缩放和随机化标记位置。
        
    - 嵌入缩放策略调整位置嵌入以匹配预训练范围，避免特征外推。例如，Chen等人（2023）通过压缩位置索引来扩展LLaMA的上下文长度。
        
    - Liu等人（2023b）和Rozière等人（2023）修改RoPE的旋转基，并在扩展序列上进行微调，称为Adjusted Base Frequency (ABF) 或 "NTK-aware" 缩放。
        
    - YaRN（Peng等人，2023）通过分段RoPE特征并应用定制的外推策略，改进了NTK-aware缩放。
        
2. **Long Context Evaluations**:
    
    - 基于Transformer的LLMs的长上下文能力评估分为两类：针对长度外推策略的合成任务评估和LLM规模的真实世界任务评估。
        
    - 合成评估针对简单任务，如长序列分类和算术语言建模。
        
    - LLM规模评估测量在广泛文本语料库中的困惑度（PPL）和复杂任务，包括摘要、问答和数学推理。
        
3. **RoPE Critical Dimensions**:
    
    - Liu等人（2023b）发现RoPE特征中可能存在两个“关键维度”，这些维度对应于满足特定波长条件的特征维度，这些维度在TSTL场景中的行为与其它维度不同。
4. **YaRN (Yet another RoPE extensioN)**:
    
    - YaRN是当前最先进的RoPE缩放方法，它引入了“NTK-by-parts”缩放策略，根据RoPE特征的时间波长应用不同的缩放策略。

这些研究为本文提出的RESONANCE ROPE方法提供了背景和基础，特别是在RoPE特征的缩放和长上下文处理方面。



## 核心亮点

论文通过以下主要方法解决TSTL场景下大型语言模型（LLMs）的泛化问题：

1. **RESONANCE ROPE**:
    
    - 提出了RESONANCE ROPE，这是一种新颖的方法，旨在通过改进RoPE特征在OOD（Out-Of-Distribution）位置的插值来缩小TSTL场景下的泛化差距。
        
    - 通过识别最佳的角频率，使得对应的波长与原始波长紧密匹配，同时强制波长对齐到整数，从而减少预训练和OOD位置之间的特征差距。
        
    - 该方法通过将RoPE特征的波长四舍五入到最近的整数来实现，确保每个RoPE特征在特定的跨度长度上重复，从而在预训练和OOD位置之间消除插值差距。
        
2. **POSGEN Benchmark**:
    
    - 引入了一个新的合成基准测试POSGEN，专门设计用于在TSTL场景中对模型的行为进行细粒度分析。
        
    - POSGEN通过控制生成令牌的难度，确保在序列中的所有位置的难度相同，从而有效地区分了由于生成更复杂令牌的算法难度增加和由于未能识别新令牌位置而导致的失败。
        
3. **实验验证**:
    
    - 在POSGEN基准测试上对RESONANCE ROPE进行了严格的测试，展示了其在OOD位置的性能提升，超过了不包含RESONANCE ROPE的现有方法。
        
    - 在LLMs的上游语言建模任务和涉及长文本的下游应用中，应用RESONANCE ROPE到当前最先进的RoPE缩放方法YaRN上，也显示出了优越的性能。
        
4. **兼容性与资源效率**:
    
    - RESONANCE ROPE与RoPE及其基于RoPE的缩放技术兼容，可以在不增加在线计算资源需求的情况下增强它们在TSTL情况下的性能。

通过这些方法，论文不仅提高了模型在处理长序列时的性能，还为未来在其他基础模型上探索RESONANCE ROPE的性能以及识别更优的RoPE特征波长组合提供了可能性。

## 实验
论文进行了以下实验来验证RESONANCE ROPE方法的有效性：

1. **合成任务评估（POSGEN）**:
    
    - 在POSGEN基准测试上应用RESONANCE ROPE，并评估模型在未见过的位置上识别性能。
        
    - 使用模块化加法任务作为示例，训练一个两层Transformer模型，并在不同长度的序列上进行评估。
        
    - 比较了使用RoPE、YaRN以及结合RESONANCE技术的模型在不同子任务（递归、CoT、半递归）上的性能。
        
2. **大型语言模型（LLM）微调评估**:
    
    - 将RESONANCE ROPE应用于当前最先进的RoPE缩放方法YaRN，并在LLaMA2-Chat模型上进行微调。
        
    - 在PG19数据集上进行微调，并在不同长度的序列上测试模型的语言建模性能。
        
    - 在L-Eval基准测试套件上评估微调后的LLaMA2-Chat模型在真实世界长文本任务上的性能。
        
3. **长序列困惑度（Perplexity）评估**:
    
    - 在GovReport和Proofpile数据集上评估不同位置嵌入策略下的LLaMA-Chat模型的困惑度。
        
    - 随机选择数据集中的样本，并报告在逐渐增加长度的文本片段上的最终困惑度。
        
4. **真实世界任务评估**:
    
    - 在L-Eval的闭环任务套件上测试LLaMA2-Chat 7B和13B模型在不同RoPE缩放策略下的性能。
        
    - 使用两种不同的微调策略：在较短序列上进行更多轮次的微调，以及在较长序列上进行较少轮次的微调。
        

这些实验涵盖了从合成任务到真实世界应用的广泛场景，旨在全面评估RESONANCE ROPE在不同上下文中的性能和适用性。实验结果表明，RESONANCE ROPE能够有效提高模型在处理长序列时的性能，并且在不同的任务和模型大小上都显示出了优越的性能。



## 未来方向
论文提出了几个可以进一步探索的点：

1. **RESONANCE ROPE在其他模型上的性能**:
    
    - 虽然RESONANCE ROPE在LLaMA模型上表现出色，但其在其他类型的大型语言模型或不同的基础架构上的性能尚未得到验证。未来的工作可以探索RESONANCE ROPE在这些模型上的应用和效果。
2. **更优的波长组合**:
    
    - 论文中提到，RESONANCE ROPE通过将波长四舍五入到最近的整数来减少插值差距。未来的研究可以探索是否存在更优的波长组合，以进一步提高模型在TSTL场景下的性能。
3. **性能与效率的平衡**:
    
    - 应用LLMs处理长文本序列时需要考虑性能和效率的平衡。未来的工作可以探索如何将RESONANCE ROPE应用于高效的Transformer模型，以同时提升性能和效率。
4. **全面的长文本基准测试**:
    
    - 目前缺乏一个全面的长文本基准测试来彻底测试LLMs的性能，特别是在长序列任务上。开发一个更全面的长文本基准测试将有助于提高实验结果的有效性和可靠性。
5. **RESONANCE ROPE与现有RoPE缩放方法的结合**:
    
    - 论文中提到RESONANCE ROPE主要解决RoPE的预关键维度的插值问题，而没有解决后关键维度的外推问题。未来的研究可以探索如何将RESONANCE ROPE与现有的RoPE缩放方法（如YaRN）结合，以解决外推问题，从而在TSTL场景中实现LLMs的全面性能提升。
6. **模型泛化能力的深入理解**:
    
    - 通过RESONANCE ROPE和POSGEN基准测试，可以更深入地理解模型在长文本处理中的泛化能力。未来的研究可以进一步分析模型在不同上下文中的泛化行为，以及如何通过改进位置嵌入来提高这种泛化能力。

这些探索点为未来的研究提供了方向，旨在进一步提升大型语言模型在处理长文本时的性能和泛化能力。


## 主要收获


## 参考资料
