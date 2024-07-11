---
title: Gemma
created: 2024-06-29
tags:
  - 大模型
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - 谷歌
---

## 论文基本信息

标题：

作者：

链接：

代码：

框架图：


## 背景

除了Gemini模型外，Gemma这一系列轻量级的SOTA开放模型似乎与我们距离更近。它基于Gemini模型相同的研究和技术构建，旨在让每个人都拥有构建AI的工具。谷歌持续扩展Gemma家族，包括CodeGemma、RecurrentGemma和PaliGemma——每个模型都为不同的AI任务提供独特的能力，并且可以通过与Hugging Face、NVIDIA和Ollama等合作伙伴轻松访问。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gW8MLCL9iaF5aaZoibMeibp99bA4IDaaSiazJXfiakS0d8nRjqL7rx75CaPcf49IHrFGX6Yian0WibCHLy14w/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

现在，Gemma家族迎来新成员——Gemma 2，延续短小精悍传统。Gemma 2此次提供的90亿（9B）和270亿（27B）参数的两个版本，其推理性能和效率均优于第一代，并具有显著的安全性改进。事实上，270亿参数版本可以与体积超过其两倍的模型进行同等级别的竞争，并且提供了此前只有专有模型才能实现的性能，而这种性能现在可以在单个NVIDIA H100 Tensor Core GPU或TPU主机上实现，从而大大降低了部署成本。

谷歌团队在重新设计的架构上构建了Gemma 2，使得这位Gemma家族的新成员既能提供卓越的性能，又具有高效的推理能力。简要概括一下，性能、成本、推理是它的突出特点：

-   性能卓越：Gemma 2 27B模型在其同体积类别中提供了最佳性能，甚至可以与体积超过其两倍的模型竞争。9B Gemma 2模型也在其同等体积类别中表现出色，并超越了Llama 3 8B和其他同类开放模型。
    
-   高效率、低成本：27B Gemma 2模型设计用于在单个Google Cloud TPU主机、NVIDIA A100 80GB Tensor Core GPU或NVIDIA H100 Tensor Core GPU上以全精度高效运行推理，在保持高性能的同时大幅降低成本。这使得AI部署更加便捷和经济实惠。
    
-   超高速推理：Gemma 2经过优化，能够在各种硬件上以惊人的速度运行，无论是强大的游戏笔记本、高端台式机，还是基于云的设置。使用者可以在Google AI Studio上尝试全精度运行Gemma 2，也可以在CPU上使用Gemma.cpp的量化版本解锁本地性能，或者通过Hugging Face Transformers在家用电脑上使用NVIDIA RTX或GeForce RTX进行尝试。
    

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/KmXPKA19gW8MLCL9iaF5aaZoibMeibp99bAyID7e2I0FRCvguuFVWXdPUkuEawq5FldfAFbvbwV7apkuUicK8buwNQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

以上是 Gemma2 与 Llama3、Grok-1 的得分数据对比。  

其实从各项得分数据来看，此次开源的 9B 大模型优势不是特别明显。近1个月前智谱AI 开源的国产大模型 GLM-4-9B 更具有优势。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/KmXPKA19gW8MLCL9iaF5aaZoibMeibp99bAsCBQSnf4se5VWRicwp7ylPTPPYjibHRLnGDFRtLndUZXHc2Yzl2796mw/640?wx_fmt=jpeg&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

此外，Gemma 2不仅更强大，还设计得更易于集成到工作流程中。谷歌为开发者提供了更多的可能性，让他们能够更轻松地构建和部署AI解决方案。

-   开放且易于访问：与原始Gemma模型一样，Gemma 2允许开发者和研究人员共享和商业化创新成果。
    
-   广泛的框架兼容性：Gemma 2兼容主要的AI框架，如Hugging Face Transformers，以及通过Keras 3.0、vLLM、Gemma.cpp、Llama.cpp和Ollama原生支持的JAX、PyTorch和TensorFlow，使其能够轻松与用户偏好的工具和工作流程结合。此外，Gemma已通过NVIDIA TensorRT-LLM优化，可以在NVIDIA加速的基础设施上运行，或作为NVIDIA NIM推理微服务运行，未来还将优化NVIDIA的NeMo，并且可以使用Keras和Hugging Face进行微调。除此之外，谷歌正在积极升级微调能力。
    
-   轻松部署：从下个月开始，Google Cloud客户将能够在Vertex AI上轻松部署和管理Gemma 2。
    

谷歌还提供了由一系列实用示例和指南构成的新Gemma Cookbook，旨在帮助构建使用者自己的应用程序并针对特定任务微调Gemma 2模型。

Gemma Cookbook链接：https://github.com/google-gemini/gemma-cookbook

与此同时，谷歌还向开发者提供了前段时间在I/O大会上官宣的Gemini 1.5 Pro的200万上下文窗口访问权限、Gemini API的代码执行功能，并在Google AI Studio中添加了Gemma 2。

-   在最新的博客中，谷歌宣布向所有开发者开放了Gemini 1.5 Pro的200万token上下文窗口访问权限。但是，随着上下文窗口的增加，输入成本也可能增加。为了帮助开发者减少使用相同token的多prompt任务成本，谷歌贴心地在Gemini API中为Gemini 1.5 Pro和1.5 Flash推出了上下文缓存功能。
    
-   为解决大型语言模型在处理数学或数据推理时需要生成和执行代码来提高准确性，谷歌在Gemini 1.5 Pro和1.5 Flash中启用了代码执行功能。开启后，模型可以动态生成并运行Python代码，并从结果中迭代学习，直到达到所需的最终输出。执行沙盒不连接互联网，并标配一些数值库，开发者只需根据模型的输出token进行计费。这是谷歌在模型功能中首次引入代码执行的步骤，今天即可通过Gemini API和Google AI Studio中的「高级设置」使用。
    
-   谷歌希望让所有开发者都能接触到AI，无论是通过API密钥集成Gemini模型，还是使用开放模型Gemma 2。为了帮助开发者动手操作Gemma 2模型，谷歌团队将在Google AI Studio中提供其用于实验。
    

以下是Gemma2的技术实验报告，我们可以从多个角度深度解析了技术细节。


-   论文地址：https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf
    
-   博客地址：https://blog.google/technology/developers/google-gemma-2/
    

## 核心亮点

### 模型结构

与之前的 Gemma 模型类似，Gemma 2 模型也是基于仅解码器的transformer架构。表 1 总结了模型的主要参数和架构选择。

![](img/Pasted%20image%2020240629104120.png)

![](img/Pasted%20image%2020240629104248.png)

部分结构要素与第一版 Gemma 模型相似，即上下文长度为 8192 个 token、使用旋转位置嵌入（RoPE）和近似 GeGLU 非线性。Gemma 1 和 Gemma 2 有一些不同之处，包括使用了更深的网络。主要差异总结如下：

-   局部滑动窗口和全局注意力。研究团队在每隔一层中交替使用局部滑动窗口注意力和全局注意力。局部注意力层的滑动窗口大小设置为4096个token，而全局注意力层的跨度设置为8192个token。
    
-   Logit软封顶。根据Gemini 1.5的方法，研究团队在每个注意力层和最终层限制logit，使得logit的值保持在−soft\_cap和+soft\_cap之间。logits ← soft_cap ∗ tanh(logits/soft_cap).
    
-   对于9B和27B模型，研究团队将注意力对数封顶设置为50.0，最终对数封顶设置为30.0。截至本文发表时，注意力logit软封顶与常见的FlashAttention实现不兼容，因此他们已从使用FlashAttention的库中移除了此功能。研究团队对模型生成进行了有无注意力logit软封顶的消融实验，发现大多数预训练和后期评估中，生成质量几乎不受影响。本文中的所有评估均使用包含注意力logit软封顶的完整模型架构。然而，某些下游性能可能仍会受到此移除的轻微影响。
    
-   使用RMSNorm进行post-norm 和pre-norm。为了稳定训练，研究团队使用RMSNorm对每个变换子层、注意力层和前馈层的输入和输出进行归一化。 
    
-   分组查询注意力。27B和9B模型均使用GQA，num\_groups = 2，基于消融实验表明在保持下游性能的同时提高了推理速度。
    

### 预训练

**数据**

他们在主要为英文数据的13万亿token上对Gemma 2 27B进行了训练，并对9B模型进行了8万亿token的训练，对2.6B模型则进行了2万亿token的训练。这些token来自各种数据源，包括网页文档、代码和科学文章。模型并不是多模态的，也没有专门为最先进的多语言能力进行训练。

最终的数据混合通过类似于Gemini 1.0的消融研究所确定。通过对较小模型进行消融来确定的。分阶段进行训练，以在训练期间改变混合物的组成——在训练结束时增加领域相关数据的权重。

Tokenizer和Gemma1, Gemini一致：a SentencePiece tokenizer with split digits, preserved whitespace, and byte-level encodings。256k大小。

和Gemma 1一样的数据过滤策略。过滤不安全数据、敏感数据、个人信息、评估数据

**知识蒸馏**

![](img/Pasted%20image%2020240629105039.png)


**infra**

研究团队使用TPUv4、TPUv5e和TPUv5p进行模型训练，细节如下方表3所示。

![](img/Pasted%20image%2020240629104538.png)

### Post-Training

在后训练中，谷歌将预训练模型微调为指令调整模型。

-   首先，在混合的纯文本、纯英文合成和人工生成的prompt-响应对上应用监督微调（SFT）。
    
-   然后，在这些模型上应用基于奖励模型（RLHF）的强化学习，奖励模型训练基于token的纯英文偏好数据，策略则与SFT阶段使用相同的prompt。
    
-   最后，通过平均每个阶段获得的模型以提高整体性能。最终的数据混合和训练后方法，包括调优的超参数，都是基于在提高模型有用性的同时最小化与安全性和幻觉相关的模型危害来选择的。 
    
Gemma 2扩展了 Gemma 1.1 的后训练数据，混合了内部和外部公共数据。使用了LMSYS-chat-1M的prompt但没有用answer.

- SFT：根据合成和真实的prompts进行行为克隆，并且主要由教师生成responses，这是一个更大的模型。We also run distillation from the teacher on the student’s distribution
- RLHF：和Gemma v1.1类似的RLHF算法，但是不同的奖励模型，比policy模型大一个数量级。新的奖励模型也更注重对话能力，特别是多轮对话。
- 模型合并：我们对使用不同超参数运行的实验的模型进行平均
- 数据过滤：使用合成数据时，我们会运行多个阶段的过滤，以删除显示某些个人信息、不安全或有毒的模型输出、错误的自我识别数据和重复的示例。继 Gemini 之后，我们发现，包含鼓励更好的上下文归因、对冲和拒绝（in-context attribution, hedging, and refusals）以最大限度地减少幻觉的数据子集可以提高事实性指标的性能，而不会降低模型在其他指标上的性能。

Gemma 2模型的微调采用了与Gemma 1模型不同的格式模式。谷歌使用了相同的控制token，具体如表4所述，表5中则提供了对话示例。

![](img/Pasted%20image%2020240629110044.png)


### 消融实验

Distillation versus from scratch. 在表6中可以发现，与从头开始训练相比，从更大的模型中提炼出来的结果提高了性能。需要注意的是，500B个token是2.6B模型最佳计算token数的10倍。研究团队从7B模型进行蒸馏，以保持与从27B模型蒸馏到9B模型相似的比例。

![](img/Pasted%20image%2020240629111343.png)


Impact of distillation w.r.t. model size. 在表7中，谷歌团队测量了随着模型规模增加进行蒸馏的影响。可以观察到，随着模型规模的扩大，这种增益仍然存在。在此消融实验中，研究团队保持教师模型的规模为7B，并训练较小的模型以模拟最终教师和学生模型规模之间的差距。

![](img/Pasted%20image%2020240629111425.png)


GQA versus MHA. 在表 8 中，我们将 9B 的 MHA 或 GQA 进行了比较。根据多个基准测试，我们观察到两种模型之间的性能总体上变化不大。我们选择 GQA，因为它需要更少的参数并且推理速度更快

![](img/Pasted%20image%2020240629111814.png)


Wide versus deep. 在表 9 中，我们表明，对于相同数量的参数，更深的 9B 网络略好于更宽的 9B 网络。尽管差距很小，但它在各个基准测试中是一致的。

![](img/Pasted%20image%2020240629111943.png)


Changing sliding window size. 在表 10 中，我们表明我们可以在推理过程中改变模型局部注意力层的滑动窗口大小，对困惑度产生中等影响。因此，调整滑动窗口的大小可以成为稍微提高推理速度的杠杆。

![](img/Pasted%20image%2020240629112040.png)

Impact of formatting. 此外，谷歌考虑到prompt/评估格式变化的影响，测量了在MMLU上的性能方差，如表11所示。Gemma 2B模型在格式稳健性方面略逊于较大的模型。值得注意的是，Mistral 7B在稳健性方面显著低于Gemma系列模型。

![](img/Pasted%20image%2020240629112131.png)

### 评估

研究团队还评估了在13万亿token上训练的27B模型（未经过蒸馏）的性能，并与类似规模的Qwen1.5 34B模型以及规模大2.5倍的LLaMA-3 70B模型在HuggingFace评估套件上的表现进行了比较，在表12中列出了评估结果。模型的选择依据基于其在HuggingFace排行榜上的排名。总体来看，Gemma-2 27B模型在其规模类别中表现最佳，甚至可以与训练时间更长的大模型进行同级别竞争。

![](img/Pasted%20image%2020240629113000.png)


我们将通过蒸馏训练的新 2.6B 和 9B 与我们之前的模型和几个标准开放模型进行比较。我们观察到，与之前的版本相比，我们的模型总体上有了巨大的改进，在 9B 模型的某些基准测试中高达 10%。这两个 2.6B 模型使用相似数量的 token（v2 为 2T，v1.0 为 3T）进行训练，我们仍然观察到新模型的显着改进。这证实了即使在相同数量的标记上进行训练，蒸馏也能显着提高模型的质量。

![](img/Pasted%20image%2020240629113514.png)



Gemma-2 27B和9B指令微调模型在Chatbot Arena中进行了盲测评估，由人类评估员与其他SOTA模型进行对比。研究团队在图1中报告了ELO评分。初步结果表明，Gemma 27B 模型为开放权重模型设定了新的技术水平，略微超过了更大的 Llama3-70BInstruct 和 Nemotron-4-340B-Instruct 模型。在相同参数范围内，Gemma 9B 的性能远远优于所有其他型号。

![](img/Pasted%20image%2020240629113749.png)

我们还提交 Gemma IT 模型用于并行人类评估研究（独立于 Chatbot Arena）。我们使用了针对安全和指令遵循 (IF) 的单轮提示集合。我们使用 gpt4o-2024-05-13 作为基本模型，并观察到与较旧的 Gemma v1.1 7B 模型相比，获胜率和偏好分数有很大改进。我们将安全性报告为相对于 GPT4o 的胜负比，并将单边指令遵循分数报告为遵循所有指令的提示比率。特别是，我们发现 Gemma 2 9B 和 27B 模型在保留安全提示集上产生的提示比 GPT4o 更安全、更合适。

![](img/Pasted%20image%2020240629113935.png)

除此之外，研究团队通过让人类评估员与模型进行对话，并遵循指定的场景进行测试，评估了Gemma 1.1 7B、Gemma 2 9B和27B模型的多轮对话能力。

谷歌使用了一个包含 500 个场景的多样化保留集合，每个场景描述了对模型的一系列请求，包括头脑风暴、制定计划或学习新知识。用户平均交互次数为8.4次。最终发现，与Gemma 1.1相比，用户对Gemma 2模型的对话满意度和对话目标实现率的评价显著更高（见表15）。此外，Gemma 2模型在从对话开始到后续轮次中，相比于Gemma 1.1 7B能够更好地保持高质量的回应。

在 Llama-3中观察到，指令微调可以提高模型在few-shot基准上的性能，尽管没有经过训练目标few-shot的能力。在表 16 中，我们的模型显示了类似的改进。总体而言，我们观察到了几个百分点的改进。我们的推测是，我们的 IT 模型更擅长理解格式化问题，因为众所周知，预先训练的模型对格式化很敏感。

![](img/Pasted%20image%2020240629114208.png)




## 未来方向



## 主要收获


## 参考资料

[In this post, we take a deep dive into the architectural components of Gemma 2 such as Grouped Query Attention, Sliding Window Attention, RoPE Embeddings, Logit soft-capping & Model-merging!](https://amaarora.github.io/posts/2024-07-07%20Gemma.html)

