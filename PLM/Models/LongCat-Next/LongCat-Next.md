---
title: LongCat-Next
created: 2026-03-29
tags:
  - 多模态
type: 论文
papername:
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2026
institution:
  - 美团
---

## 论文基本信息

标题：

作者：

技术报告地址：

https://github.com/meituan-longcat/LongCat-Next/blob/main/tech_report.pdf

GitHub地址：

https://github.com/meituan-longcat/LongCat-Next

HuggingFace地址：

https://huggingface.co/meituan-longcat/LongCat-Next

Demo体验：

https://longcat.chat/longcat-next

框架图：


# 一个多模态模型，理解也要，生成也要

原创 rumor 李rumor

 _2026年3月27日 09:18_ _北京_

卷友们好，我是rumor。

CV世界长期存在两类「各自为政」的模型：

- **理解派**（LLaVA、InternVL 等）：看懂图、回答问题，但**画不出图**。
    
- **生成派**（Stable Diffusion、DALL-E 等）：画出漂亮的图，但**看不懂图**。
    

这件事对文本来说是天然的。LLM 一个next-token prediction就同时搞定了理解和生成。凭什么？因为文本有三个天然优势：

1. **同一个 token，读和写是同一回事。**tokenizer是双向无损的——token化后可以完美还原原文。
    
2. **只在一个语义层次上工作。**「猫」既是理解的最小单元，也是生成的最小单元，不存在「高层语义的猫」和「低层像素的猫」两种表示。而图像不一样，
    
3. **一维、离散、有限。** 天然匹配Transformer因果注意力，词表有限。
    

图像恰好在这三个层面**全部冲突**：

|维度|文本|图像|
|---|---|---|
|Tokenization|BPE，无损|VQ-VAE/VAE，有损|
|语义层次|读和写完全相同|理解要高层语义，生成要低层像素|
|数据结构|1D离散有限|2D连续无限|

这就引出了一个根本矛盾：**一个编码如果足够抽象能理解「这是猫」，就必然丢失了猫毛的像素细节；一个编码如果保留了所有像素细节，就包含了大量对理解无用的冗余信息**。

所以众多工作尝试统一理解与生成的工作，本质上在做两件事：**为图像找到一套类似文本BPE的「理解-生成统一表示」，或者设计架构来绕过「图像不存在这种统一表示」这个事实。**

# 主流方案

当前的原生多模态统一模型可以沿**两个维度**来看：**LLM 怎么画图（生成范式）以及理解和生成是否共享视觉编码器（编码策略）**。

**生成范式**

参考综述Unified Multimodal Understanding and Generation Models[1]的分类体系，生成范式可以分为四类：

|范式|原理|代表模型|
|---|---|---|
|**纯扩散（Diffusion-based）**|在扩散/Flow框架内同时完成理解和生成|UniModel[2]、FUDOKI[3]|
|**纯自回归（Auto-Regressive）**|把图像变成离散token，逐个预测next-token|Lumina-mGPT[4]、Janus-Pro[5]、OneCAT[6]|
|**AR + 外部扩散模型**|LLM管思考和推理，独立的扩散模型管画图|BLIP3-o[7]、OmniGen2[8]、UniWorld-V1[9]|
|**AR + Diffusion/Flow 融合**|同一个Transformer内，文字用AR loss，图像用扩散/Flow loss|Mogao[10]、BAGEL[11]、Show-o2[12]、InternVL-U[13]|

**编码策略**

在每种生成范式内部，各模型在视觉编码上的选择也不同：

|策略|原理|代表模型|
|---|---|---|
|**共享编码**|理解和生成用同一个VQ tokenizer|Lumina-mGPT、OneCAT|
|**解耦编码**|理解用ViT/SigLIP（语义级），生成用VAE/VQ（像素级）|Janus-Pro、BAGEL、Mogao、InternVL-U|
|**语义-像素融合**|一个编码器内部同时提取语义和像素特征|Show-o2、NEO|
|**可学习查询桥接**|LLM输出learnable query，送入外部扩散模型生成|BLIP3-o|

一个清晰的趋势是：**绝大多数工业级模型都放弃了「一套token打天下」，转向了解耦或融合方案**——理解和生成各用各的编码器（或在一个编码器内部做融合），在LLM内部汇合。与此同时，「AR+Diffusion/Flow 融合」成了生成范式的主流选择。原因很直觉：文字天然适合自回归，图像天然适合扩散。

但LongCat-Next[14]的作者们，做了一个不同的选择。

# LongCat-Next

面对离散自回归 vs. 连续扩散，LongCat-Next坚定地选择离散自回归方案。

作者们的核心信念是**「表征的对称性」（representational symmetry）**。他们认为，一个真正的原生多模态模型，图像token和文本token应该在模型内部具有**对等的地位**——相同的表征形式、相同的训练范式、相同的损失函数。

这不仅是审美偏好，也有工程上的深刻考量：

- **范式统一**：如果图像也是离散token，那整个模型就是一个纯粹的next-token prediction 系统。不需要在同一个Transformer里同时维护两套 loss（交叉熵 + 扩散），不需要处理两种loss的权重平衡、学习率差异、梯度冲突。
    
- **Loss 一致**：连续方案（扩散/Flow Matching）的loss本质上是回归问题（MSE/velocity prediction），和文本的分类问题（交叉熵）是两种完全不同的优化目标。把它们硬塞进一个模型，需要精心调参才能让两边都收敛好。离散方案则没有这个问题——所有token统一用交叉熵，干净利落。
    
- **架构简洁**：不需要额外的DiT生成头、adaLN时间步调制、双流注意力等为扩散量身定做的组件。整个模型就是一个 decoder-only Transformer，和GPT没有本质区别。
    

选定离散方向后，作者需要直面一个根本问题：把图像变成离散token，到底难在哪里？

文字天然是离散的紧凑单元，「猫」这个token已经足够精确地传达「猫」的含义。但图像是高维、连续的信号。业界普遍怀疑：把丰富的视觉信息压进有限的codebook，会导致不可避免的信息损失，形成性能天花板。这也是为什么大多数团队转向了扩散/Flow Matching 等连续方案。

作者们把它**解构为两个串联的瓶颈**：

1. **表征能力瓶颈**：用什么encoder来提取视觉特征？
    
2. **离散化信息损失瓶颈**：如何降低量化中信息的损失？
    

接下来，我们看作者分别给出了怎样的解法。

## 视觉编码器SAE

面对「用什么编码器提取视觉特征」这个问题，作者系统考察了业界现有的几类方案：

1. **重建型编码器（VAE）**：VAE天然为图像重建设计，像素级高保真。但问题是它们缺乏高层语义理解能力。一个VAE可以完美重建猫毛的纹理，但不「知道」这是一只猫。用这类编码器做理解任务，效果远不如语义编码器。
    
2. **自监督语义型编码器（DINOv2、SigLIP 等）**：这类编码器通过自监督或对比学习获得了强大的特征提取能力。但作者指出它们各有缺陷：DINOv2学到的特征更偏向视觉结构（适合分割等任务），但缺乏与语言的语义对齐；SigLIP 通过图文对比学习获得了一定的语义对齐，但对比学习的目标是「区分不同图文对」而非「完整理解图像内容」，缺乏生成所需的细粒度语义。
    
3. **无编码器的原始像素**：最简单的方案——直接把像素当输入。但像素极度冗余：一张 256×256 的图有 65536 个像素，其中大量是相邻像素间几乎相同的冗余信息。序列太长，计算开销爆炸。
    

作者提出了一个被忽视的第四类——**SAE（Semantic-and-Aligned Encoder，语义对齐编码器）**。这类编码器的独特之处在于：它们不是通过自监督或对比学习训练的，而是通过**大规模视觉-语言联合预训练**（即作为多模态大模型的视觉前端，直接参与语言建模任务的训练）。这种训练方式赋予了SAE两个关键属性：

- **语义完备性（Semantic Completeness）**：不仅知道「这是猫」，还理解猫的姿态、场景关系、上下文语义等丰富信息——因为这些正是回答语言问题所需要的。
    
- **语言亲和性（Language Affinity）**：特征空间天然与LLM的文本空间对齐——因为训练时就是和LLM联合优化的，不需要额外的适配层来弥合模态鸿沟。
    

同时，一个发现进一步坚定了作者选择SAE的信心。像QwenViT这样的SAE，虽然训练目标完全是语义理解（而非像素重建），但其Transformer架构中的**残差连接**天然保留了一条低层信号传播的隐含通路。输入的patch embedding通过残差连接层层传递，即使中间层在做高度抽象的语义计算，原始的低层视觉信息也不会完全消失。

这意味着，**即使没有任何重建监督，SAE的特征中也隐含了相当的像素级信息**。

换句话说，SAE不仅能**理解**图像，还为**生成**图像提供了基础——低层通路保留的像素信息，可以被后续的量化和解码步骤利用。这让SAE成为了同时服务理解和生成的理想基础——正好契合作者追求的「表征对称性」。

## RVQ:解决离散化信息损失

编码器选好了，接下来要把这些连续的特征向量变成离散token。这就是第二个瓶颈——量化过程中的信息损失。

传统VQ（Vector Quantization）的做法很直接：连续特征向量z → 在codebook中找最近邻 → 替换为最近的码字 → 离散ID

问题在于，一次量化只能用**一个**码字来近似原始向量。如果 z 恰好落在两个码字之间，就只能「四舍五入」到其中一个——信息就这样丢了。码本越小，这种「四舍五入」越粗暴。**这种单步量化是一个 hard ceiling——码本大小决定了信息上限，无法通过其他手段弥补。**

作者选择了**残差向量量化（RVQ, Residual Vector Quantization）**来突破这个瓶颈。

RVQ 的核心思想很直觉：**一次量化不够准就多量化几次，每次编码上一次「没搞定」的残差**。

`第 1 级：z  → 在 codebook₁ 中量化 → 码字 c₁        残差 r₁ = z - c₁   第 2 级：r₁ → 在 codebook₂ 中量化 → 码字 c₂        残差 r₂ = r₁ - c₂   第 3 级：r₂ → 在 codebook₃ 中量化 → 码字 c₃        残差 r₃ = r₂ - c₃   ...   重建：z ≈ c₁ + c₂ + c₃ + ...   `

每一级都有自己独立的codebook，专门负责编码上一级「遗漏」的细节。像「残差的残差」一样层层递进。

打个比方。假设你要用有限的词汇来描述一种颜色：

- **VQ 的方式**：只能说「红色」——最接近的那一个词。但如果这个颜色其实是偏橙的暖红，你只能说「红色」，细微的色调信息丢了。
    
- **RVQ 的方式**：先说「红色」（第一级），然后补充「偏橙」（第二级编码「红色」和实际颜色的差异），再补充「暖色调」（第三级编码剩余的细微差异）。层层递进，信息保留得越来越完整。
    

这种分层机制的精妙之处在于：

- **第一级**捕捉最粗粒度的全局信息（高层语义）→ 已经足够支撑理解任务
    
- **后续各级**逐层补充细节信息（纹理、边缘等）→ 生成高质量图像时才需要
    
- 层数越多，重建越精确，**理论上可以无限逼近连续表征的效果**
    

RVQ 在保持离散性（每级仍然是从有限码本中选一个 ID）的同时，通过多级叠加大幅提升了信息密度——既保留了高层语义（用于理解），也保留了细节信息（用于生成），实现了信息密度和压缩率的平衡。

## dNaViT: 统一的视觉Tokenizer

作者把SAE和RVQ组合到一起，构建了 **dNaViT（discrete Native-resolution Vision Transformer）**——一个统一的视觉 tokenizer：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/2oicuz5vRaSt3lVkpLpA04EgRugc7QDYFueyI0VwTZPwMpSwQreg6LqFLEqsqEgUEGmvhyhfZWFvvVn7GNKCyoQXqEcBw9tYyuXuCCVt0luw/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=0)

dNaViT 的关键能力：

- **Tokenization**：把任意分辨率的图像编码成离散 ID 序列，最高 28× 压缩比
    
- **De-tokenization**：从离散token序列重建回图像
    
- **原生分辨率支持**：不需要把所有图像 resize 到固定尺寸，保留原始宽高比信息
    

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/2oicuz5vRaStFBXGk24DFUo9iboVKkSFyNyia9Mu5kXK2Ss8NXqTGgOTtXwHFLsOKsWHdRS10YwXLwRwiaibZLza84XDoHuHwBtAxDUc7B0ykd9I/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=1)

这使得视觉token与文字token具有了**等价地位**——同样是离散 ID，同样可以做next-token prediction。

同时，多级RVQ意味着每个图像位置有多个 token（每级一个）。如果逐级逐位置自回归预测，序列长度会爆炸。

作者设计了一套高效的推理方案：

- **加法编码（Additive Encoding）**：多层token的嵌入直接相加融合，而非拼接。这保持了序列长度不变——不管用几级RVQ，LLM看到的序列长度和单级一样。
    
- **DepthTransformer**：一个轻量的辅助网络，专门负责在**一个自回归步骤内**解码多层表征。一次前向传播就能处理所有级别的token。
    

这使得视觉token在LLM内部的处理效率与文字token基本一致——不会因为多级量化而导致推理变慢。

## 音频处理

同样的设计理念，作者延伸到了音频模态。逻辑完全一致：找一个好的编码器 + 用RVQ做离散化。

**编码器：和视觉SAE的逻辑一样——Whisper也是通过大规模音频-文本联合训练获得了语义完备性（理解语音内容）和语言亲和性（与LLM兼容）的编码器。此外，Whisper还能捕捉副语言特征（语调、情感、语速等），这对高质量语音合成至关重要。**

**量化：同样用RVQ做离散化，压缩率为12.5Hz——即每秒音频只需要12.5个 token。这个压缩率足以保留语音的语义和韵律信息。**

**解码器：解码端用配对的decoder做初步重建，再用flow matching精炼网络做高保真音频还原。**

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/2oicuz5vRaStqXpLuIahYO5POkFFh53nbhHmFcpgkodOjc0H2PHLrg0RIXvXWqh5oG2oBX0gIQTJTWCJPG5z9gicJ21nkMwtsibMXNU1zqbRHc/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=2)

**创新设计——随机延迟统一训练**

文本引导的语音生成有两种常见模式：

- **并行模式**：文本和语音同时生成（低延迟，适合实时对话）
    
- **串行模式**：先完成文本再生成语音（高质量，适合朗读）
    

传统方法只能二选一。作者通过在训练时**随机引入文本-语音的延迟偏移**，让模型学会在两种模式间自由切换——推理时根据场景需求选择即可。这一个设计同时覆盖了实时对话和高质量合成两个场景。

## DiNA范式:统一到一个LLM中

所有模态的tokenizer准备好后——视觉有dNaViT，音频有Whisper+RVQ，文本有BPE——作者将它们整合到**LongCat-Flash MoE backbone**上，构成了**DiNA（Discrete Native Autoregression）**范式：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/2oicuz5vRaSvp38NvP1UmIF4p49sibhc578iaYOJql3XIoZOE7UicicKVibVFCvibStePur40o8ibY0wc4MT2RzrTIHvWlD7wC1ODRa0pdA0A9gL45M/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=3)

DiNA 的设计哲学极度简洁：

- **Backbone 本身不感知模态**：LLM 只看到一串离散tokenID，不知道也不需要知道哪些是文字、哪些是图像、哪些是音频
    
- **Tokenizer/Detokenizer 负责信号转换**：各模态的编解码器各司其职，把连续信号变成离散 ID（或反过来）
    
- **一个模型，一种loss，所有模态**：所有token统一用交叉熵loss做next-token prediction
    

最终，一个 DiNA 模型同时具备五种能力：**看图理解、生成图像、听懂语音、合成语音、处理文本**——全部在同一个自回归框架内完成。

## 几个反直觉的发现

论文中的实验验证了几个此前业界普遍持怀疑态度的结论。

**1. 离散表征没有性能天花板**

这是对离散方案最致命的质疑：VQ的信息损失是否会造成不可逾越的性能上限？

实验表明：**通过增加训练数据，离散表征可以无限逼近连续表征的效果**。在充分训练后，两者的loss差距可缩小到约1%。换句话说，所谓的「信息瓶颈」不是一个固定的天花板，而是一个可以通过scaling不断推高的弹性上限。

这个发现直接回应了业界选择连续方案的核心理由——「离散化必然有损」是对的，但「有损就必然有天花板」是错的。

**2. 理解和生成冲突较小**

另一个常见的担忧是：把理解和生成塞进同一个模型，会不会互相拖后腿？实验给出了一个意外的答案：

- **生成任务对理解能力的损害较小**——加入生成训练后，理解损失仅比纯理解模型高0.006
    
- **理解任务反而提升生成质量**——加入理解训练后，生成loss降低了0.02
    

这说明两种任务在共享的离散表征空间中冲突**较小**——理解学到的语义知识帮助了生成，生成学到的重建知识对理解负向影响有限。

**3. MoE专家自动按模态分化**

虽然LongCat-Flash的MoE架构是**模态无关的**（没有人为指定哪些专家处理哪种模态），但训练完成后，部分专家**自然地**偏向处理视觉或音频token。

这是一个涌现行为——模型自己发现了「术业有专攻」的好处，在不需要硬路由的情况下实现了软分工。对比OneCAT的「文本/理解/生成三路硬路由」设计，LongCat-Next的方案更优雅：让模型自己决定怎么分工，而非人为规定。

**4. 离散token空间天然对齐语言空间**

通过t-SNE可视化，作者发现LongCat-Next的视觉token和文本token在嵌入空间中是**交织分布的**——不是像很多模型那样形成分离的聚类（视觉的归视觉，文本的归文本），而是真正混合在一起。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/2oicuz5vRaSsV3mcuu4OnXnuK6aictNauTKdyd0HWEbPj4XcVvzn9DxbBfiaAgkjXJcGRyfUgTZiav9YjDiaS0ln0HVlRRVyVZr2da1d9libwN0g4/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=4)

这印证了近期备受关注的**「柏拉图表征假说」（Platonic Representation Hypothesis）**：不同模态在充分训练后，会收敛到同一个底层表征空间。LongCat-Next的离散统一设计，可能恰好为这种收敛提供了最自然的条件——所有模态共享同一种表征形式（离散 ID）、同一个训练目标（交叉熵），天然鼓励跨模态对齐。

## 天然兼容强化学习

离散表征还带来了一个出乎意料的额外优势——**与强化学习的天然兼容性**。

在连续方案（Diffusion/Flow Matching）中，图像生成是一个确定性的 ODE 求解过程。如果要做RL优化生成质量，需要把确定性的ODE采样转换为随机的SDE采样，才能定义策略梯度——这个转换非常不自然，也增加了工程复杂度。这也是为什么X-Omni等模型需要专门设计RL修补管线。

但在离散方案中，**视觉latent space本身就是action space**。每一步预测下一个离散tokenID，天然就是一个序列决策问题——可以无缝对接GRPO等LLM强化学习方法，和优化文本生成**完全一样**。

作者设计了**多维度奖励模型**，综合评估四个维度：

- 综合能力
    
- OCR准确性（文字渲染质量）
    
- 语义对齐度（图文一致性）
    
- 图像质量（美学、清晰度）
    

并解决了一个实际问题：RL训练理解任务时会出现**熵爆炸**（模型的输出熵急剧增大或减小，导致生成质量崩溃）。作者通过**序列级过滤机制**——过滤掉奖励异常的训练序列——有效控制了这个问题。

这意味着LongCat-Next不仅可以通过SFT提升性能，还可以通过RL进一步精调——而这对连续方案来说代价大得多。

---

回到最开始的问题：**图像有没有可能找到一套类似文本的「理解-生成统一表示」？**

LongCat-Next的答案是：**有。**

参考资料

[1] 

Unified Multimodal Understanding and Generation Models: _https://arxiv.org/abs/2505.02567_

[2] 

Unimodel: A visual-only framework for unified multimodal understanding and generation: _https://arxiv.org/abs/2511.16917_

[3] 

FUDOKI: Discrete Flow-based Unified Understanding and Generation via Kinetic-Optimal Velocities: _https://arxiv.org/abs/2505.20147_

[4] 

Lumina-mGPT: Illuminate Flexible Photorealistic Text-to-Image Generation with Multimodal Generative Pretraining: _https://arxiv.org/abs/2408.02657_

[5] 

Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scalin: _https://arxiv.org/abs/2501.17811_

[6] 

OneCAT: Decoder-Only Auto-Regressive Model for Unified Understanding and Generation: _https://arxiv.org/abs/2509.03498_

[7] 

BLIP3-o: A Family of Fully Open Unified Multimodal Models-Architecture, Training and Dataset: _https://arxiv.org/abs/2505.09568_

[8] 

OmniGen2: Exploration to Advanced Multimodal Generation: _https://arxiv.org/abs/2506.18871_

[9] 

UniWorld-V1: High-Resolution Semantic Encoders for Unified Visual Understanding and Generation: _https://arxiv.org/abs/2506.03147_

[10] 

Mogao: An Omni Foundation Model for Interleaved Multi-Modal Generation: _https://arxiv.org/abs/2505.05472_

[11] 

Emerging Properties in Unified Multimodal Pretraining: _https://arxiv.org/abs/2505.14683_

[12] 

Show-o2: Improved Native Unified Multimodal Models: _https://arxiv.org/abs/2506.15564_

[13] 

InternVL-U: Democratizing Unified Multimodal Models for Understanding, Reasoning, Generation and Editing: _https://arxiv.org/abs/2603.09877_

[14] 

LongCat-Next: _https://github.com/meituan-longcat/LongCat-Next/blob/main/tech_report.pdf_

## 主要收获


## 参考资料

[# 一个多模态模型，理解也要，生成也要](https://mp.weixin.qq.com/s/8ymAKU7jCWjmM4m6K_HaZw)

[# 重构原生多模态！美团发布纯离散基座，真正实现万物皆Token](https://mp.weixin.qq.com/s/OA-t-JlqINZAzhOxdvvtGg)

