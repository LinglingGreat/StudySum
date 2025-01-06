---
title: LLaMA3
created: 2024-07-24
tags:
  - 大模型
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - MetaAI
---

## 论文基本信息

标题：

作者：

链接：

代码：

框架图：

![](img/Pasted%20image%2020240724144456.png)

![](img/Pasted%20image%2020240724144934.png)

LLaMA-3特点
- supports multilinguality, coding, reasoning, and tool usage。它支持至少八种语言（而Qwen 2能够处理20种语言）。
- 最大的模型是405B的dense模型，支持128K

训练高质量基座模型的三个关键点：data, scale, and managing complexity
- pre-training and post-training阶段的数据质量和数量都提升了。Llama3用了15T tokens训练，而Llama2用了1.8T tokens
- 训练了一个405B的dense模型，用了15.6T tokens。we also train our smaller models for much longer than is compute-optimal. 结果是更好
- 选择dense架构而不是MOE，训练更稳定；采用SFT，RS(拒绝采样), DPO而不是难以训练和扩展的PPO

[沐神](https://zhida.zhihu.com/search?content_id=246751270&content_type=Article&match_order=2&q=%E6%B2%90%E7%A5%9E&zhida_source=entity)：15个T可能是目前在公有的网络上面，能够抓到的文本数据的一个大概的上限，这个"上限"的意思是指，与其再找一些增量的数据，不如去调整现有的数据的质量。

多模态部分
- 多模态预训练包括用于图像和语音的单独编码器 
- 图像编码器在图像文本对上预训练，而语音编码器以自监督方式进行预训练，通过离散令牌表示重建屏蔽输入 
- 两个预训练编码器（图像和语音）分别通过视觉和语音适配器连接到预训练的 LM。

MOE结构会让模型效果更好吗？答案是否定的。这个在很久以前ChatGPT火之前就有研究结论，从对模型效果的影响来说，MOE结构相对Dense模型本身并不会带来额外优势，甚至是有劣势的。MOE的主要优势是减少训练和推理成本，付出的代价是训练不够稳定以及推理时额外付出大内存来存储膨胀的参数量。但当用户量大请求多的时候，推理成本占比会更高，此时使用MOE对于推理会更友好，这是为何当模型大到一定程度模型结构就会从Dense转向MOE的主要原因，是出于成本、效率而非效果角度考虑。我之前看到有些介绍说MOE结构效果更好，这种观点是没有事实依据的。

Llama3 405B 之所以没有采用MOE，技术报告指出主要是考虑到Dense模型训练更稳定，所以选择了Dense结构。相比GPT 4的1.8T的MOE模型结构，405B的Dense模型效果与之相当甚至要更好一些（当然，不排除GTP 4目前已经是一个蒸馏小模型的可能）。

## 预训练

- 405B 参数
- 15.6T 标记，Llama-2的将近9倍（15.6T vs 1.8T）
- 知识截止时间为 2023 年底

- 8K 标记的上下文窗口
- 在持续预训练阶段，上下文窗口增加到 128K

数据清洗
- personally identifiable information（PII）and safety filtering. 首先就是要清洗掉和个人信息相关，以及包含成人内容的数据。
- 自定义 HTML 解析器，从web数据中解析多样性的数据。保持数学和代码内容的结构。对于数学相关的页面，特意保留了图片，因为很多公式都被渲染成了图片。我们发现与纯文本相比，Markdown 对主要在 Web 数据上训练的模型的性能有害，因此我们删除了所有 Markdown 标记。
- 去重
	- URL-level：对于同一个页面，只保留最新的版本。
	
	- Document-level：用MinHash做了文档级别的近似去重。
	
	- Line-level：和ccNet的做法相似，对于一个包含30M文档的bucket，如果某行数据重复出现超过6次就会被删除。人工检查发现这样做能够删掉一些如网页导航、cookie warnings这样的没太大价值的数据，但是也会删掉一些高频的高质量数据，不过从结果上来看总体的正收益是比较大的。

数据质量清洗：
- 参考《Scaling language models: Methods, analysis & insights from training gopher》，用n-gram coverage ratio过滤掉包含大量重复信息的内容（比如logging和error messages）；这些内容在大量重复的同时又不完全相同，所以可能在去重中会被漏掉。
- 参考《Exploring the limits of transfer learning with a unified text-to-text transformer》，用dirty word counting过滤成人内容。
- 通过token分布的KL散度过滤掉与训练语料库分布相比包含过量outlier token的内容。

Model-based quality filtering: 用Llama-2对数据质量做分类，然后用fasttext和DistilRoberta学习Llama-2给出的数据，用于对数据是否符合质量要求进行分类。

Code and reasoning data：在代码和推理数据上，使用类似DeepSeek-Coder-V2的做法。针对包含数学推理、STEM领域推理以及与自然语言交织的代码网页，调整了HTML的提取规则、质量分类的prompt等。

Multilingual data：对于多语言数据，在移除可能包含PII和成人内容的数据之后：

- 用fasttext把数据进行176种语言的分类。
    
- 进行document-level和line-level的去重。
    
- 用每种语言各自的质量分类器过滤低质量数据。
    
并通过实验确定最终各种语言的占比，平衡英文和多语言的应答质量。

**数据比例**

不同来源和领域的数据配比会极大影响各个下游任务效果。这里主要用到knowledge classification和scaling law experiments来决定数据配比。

- Knowledge classification：给数据进行领域的分类，并减少训练数据中某些种类的数据，比如arts和entertainment数据。
    
- Scaling laws for data mix：通过在规模较小的模型对不同的data mix分别跑scaling law的实验，来获取最佳的data mix。
    
- Data mix summary：最终的数据中，约50%属于general knowledge，25%属于数学和推理，17%的代码以及8%的多语言数据。

**数据退火**

在learning rate的退火阶段使用高质量的代码和数学数据可以提升在关键benchmark上的效果。参考《Datacomp-lm: In search of the next generation of training sets for language models》的做法，在退火阶段对高质量数据进行了upsampled。

按这个做法，在GSM8k和MATH数据集上检测了8B模型，发现都有比较大的提升，但是405B模型的提升则不大，猜测可能是因为405B模型的in-context learning能力和推理能力本身已经比较强了，因此即使不在退火阶段使用相关高质量数据集，也已经效果比较好。

另外，既然annealing加入对应数据可以提升下游任务的效果，那么就可以用annealing来检测数据质量了。通过在退火阶段加入不同的数据，观察对下游任务的影响，来判断所加数据是否是高质量数据，这和《Does your data spark joy?performance gains from domain upsampling at the end of training》的思路类似。


三阶段训练法：(1) 初始训练 initial pre-training, (2) 长文训练 long-context pre-training, and (3) 退火训练 annealing

初始训练：

- 余弦调度 8 × 10−5 , 8,000 steps热身, 然后在1,200,000步上降到 8 × 10−7
    
- 上下文长度和BS缓慢增加，配4M的bs用4,096长度, 训练 252M 个 token 后，再配8M的bs扩展序列长度为 8,192，这个阶段大约是 252M tokens。训练 2.87T 个 token 后，最终16M的BS
- 发现这种训练方法非常稳定：我们观察到很少有损失峰值，并且不需要干预来纠正模型训练偏差。
- 调整数据混合。我们在训练期间对预训练数据组合进行了一些调整，以提高模型在特定下游任务上的性能。特别是，我们在预训练期间增加了非英语数据的百分比，以提高 Llama 3 的多语言性能。我们还对数学数据进行上采样以提高模型的数学推理性能，我们在预训练的后期添加了更多最新的网络数据- 训练以推进模型的知识截止，并且我们对预训练数据的子集进行了下采样，这些数据后来被确定为质量较低。
    

长上下文训练

- 仅当模型在短上下文评估上的性能完全恢复，并且模型可以完美解决该长度内的“大海捞针”任务时，上下文长度才会增加。
- 上下文长度共分为六个阶段，逐步增加，从l 8K 逐步增加到 128K。
- 使用 800B 个训练 token 完成长上下文预训练。

退火训练

- 最后40M token，用128K长度，逐渐线性缩减学习率到0
- 在这一退火阶段，调整了数据混合配比，以增加高质量数据比如数学、代码、逻辑内容的影响。最后，将若干退火期间模型Check Point的平均值，作为最终的预训练模型。

## 模型结构

![](img/Pasted%20image%2020240725172304.png)

模型架构与 llama 和 llama-2 相同，但有一些修改（再次证明质量数据仍然是王道！）

- 具有 8 KV 头的 GQA，降低推理时KV cache的需求。

- RoPE 频率增加到 500,000，按《Effective long-context scaling of foundation models》的结果，这个数值足够支持32,768长度的窗口了。

- 注意力掩码可防止同一序列内不同文档之间的自我注意力。样本间穿越在预训练阶段影响不大，以前大家也不在乎，但作者说在扩长序列时候影响很大。

- 128K 词汇大小（100K 来自 tiktoken，提高了英语的压缩率。28K 额外标记用于英语以外的语言，可以提高压缩率和下游性能，并且对英语分词没有影响。）

- 126 层（层数多2层，是训练阶段方便流水线并行切分的技巧）、128 个注意力头和 16,384 嵌入大小

## scalinglaw

作者说了现有的scalinglaw通常只预测loss，而不是特定的benchmark上的表现；(2)scalinglaw可能因基于小资源进行的预训练运行而变得没那么可靠。

对此，作者搞了一个两步走方法。
1.先建立计算最优模型在**下游任务**上的负对数似然与训练FLOPs之间的相关性。
2.利用scalinglaw模型和使用更高计算FLOPs训练的旧模型，将**下游任务上的负对数似然**与**benchmark的准确率**指标关联上。这里显然跟用什么模型无关，因此作者在这步用了LLama2系列的模型。

具体来说，对从40M到16B的模型进行不同FLOPs的训练，得到各个compute预算下的最佳规模。如下图所示

![](img/Pasted%20image%2020240725172825.png)

这里训练的时候根据模型大小使用了不同的lr，同时在不同的compute budget下使用了从250k到4M不等的batch size。

基于这些实验结果，对给定compute budget C下的optimal number of training token  N*(C)进行拟合：

![](img/Pasted%20image%2020240729134744.png)

得到$(\alpha, A)=(0.53,0.29)$，从这里推算出3.8 x 10^25  FLOPs的计compute budget对应的最佳规模和数据量是402B和16.55T token。

从这些实验结果还另外得到一个发现：随着compute budget的增加，IsoFLOPs的曲线逐渐变得平缓，这说明大规模的模型对规模和训练数据量的少量波动会更加robust，少量的波动不会对最终结果造成很大影响。

在这个基础上，先拟合“各个compute budget下最佳模型在下游benchmark的正确答案上的Normalized NLL per Char”和FLOPs之间的线性关系，再拟合Normalized NLL per Char和下游任务accuracy的sigmoid关系。这样就建立了FLOPs和下游benchmark上accuracy的关系。在ARC Challenge任务上的拟合情况如下

![](img/Pasted%20image%2020240725172842.png)

从结果上看，这个方法预测的405B效果基本准确，偏差很小。

注意：
- 每个任务需要单独拟合各自的曲线
- 右边的图，用了Llama2的4个模型（传闻有个比70B更大的Evil模型没放出来），再加上用小模型训练的7个点，预测出曲线。
## Infra

- Llama 3 405B 在多达 16K H100 GPU 上进行训练，每个 GPU 以 700W TDP 运行，配备 80GB HBM3，使用 Meta 的 Grand Teton AI 服务器平台

- 专用集群，不是之前的Meta’s AI Research SuperCluster，全新的Meta’s production clusters。Tectonic（Meta 的内部）分布式文件系统用于存储，240 PB  SSD，7500 台机器，2TB-7TB/s 吞吐，如此高吞吐的存储集群是为了最小化ckpt的IO耗时。

- 基于 RoCE 的 AI 集群由 24K GPU 组成，通过三层网络连接, RoCE，单口400Gb/s
- 最底层1个ToR下2机器配16卡
- 1 Pod 配192 ToR
- 3072 张GPU 1:1 收敛比
- 往上8 个Pod 1:7 的收敛比
- 由于跨pod 带宽降低，所以模型并行编排和资源调度均需考虑网络架构

- 并行性和针对硬件拓扑优化的调度程序

- 增强的 ECMP 路由和深度缓冲区交换机用于拥塞控制

负载均衡
- 两个GPU 间使用16个流，而不是1个，来降低单流流量，以更好地负载均衡
- 在网络包的头部增加了特殊区域，通过hash 来使得流的选路更加均衡
拥塞控制
- 在spine 上使用deep-buffer 交换机以适应集合通信导致的短时拥塞，并能够降低慢节点引发持久的拥塞和反压

- 4D 并行性：四种不同类型的并行性方法的组合，包括张量并行性、管道并行性、上下文并行性和数据并行性，用于对模型进行分片

- 在上下文并行性中，分区跨越序列维度。基于 all-gather，它们会收集所有键和值，然后计算本地查询张量块的注意力输出

- 优化并行顺序，以获得更好的网络带宽和延迟：TP、CP、PP、DP
- 使用了FSDP，但是model weight 只拉取一次，以减少反向梯度计算时的weight allgather 通信
PP并行策略改进
- Batch Size 限制：当前的流水线并行策略 会限制 micro batch 个数为流水线stage 的整数倍，会导致global batch size 和 流水线 stage 相互制约
- 显存不均衡：第1个stage 会多出很多显存占用，之后逐stage 降低
- 计算不均衡：最后一层会额外计算 output layer 和 loss，增加了计算量和延时，首尾stage 做了padding
CP并行策略的改进
- 和Ring CP 一样的在序列维度切分，切分为2\*CP 份以负载均衡，但没有用环状通信来overlap 计算和通信，而是基于allgather 的通信。
网络架构感知的并行策略
- [TP, CP, PP, DP] 的并行策略是针对网络通信优化做专门设计的
- 开发了显存预估和性能探查工具，用以平衡显存开销和通信性能
数值稳定性
- BF16 MFU 为 38%-43%
- 使用 FP32 进行梯度累积。对于在多个地方使用的即时张量，如视觉编码器输出，梯度在 FP32 中累积
集合通信
- 基于NCCL 开发了 NCCLX，在高延迟网络下显著提升性能
- [TP, CP, PP, DP]  并行策略可能导致PP 和DP 通信跨Pod：原 allgather 和 reducescatter 实现依赖数据chunk 和拷贝，需要大量的小的控制信息的通信，进行额外的拷贝操作，且占用GPU 资源来做通信。对此，llama团队 优化chunk 策略，提升小的控制包的通信优先级。
可用性
- 54 天快照预训练，中断 467 次。GPU 问题占总问题的 58%（这就是 TPU 更胜一筹的原因）,剩余是网络问题。
- 白天，由于温度较高，GPU 的吞吐量会变化 1-2%


## Post-training

![](img/Pasted%20image%2020240725173402.png)

**总体流程**

首先用人工标注数据训练RM模型，用来评价一个<Prompt,answer>数据的质量，然后用RM参与拒绝采样（Rejection Sampling），就是说对于一个人工Prompt，用模型生成若干个回答，RM给予质量打分，选择得分最高的保留作为SFT数据，其它抛掉。这样得到的SFT数据再加上专门增强代码、数学、逻辑能力的SFT数据一起，用来调整模型得到SFT模型。之后用人工标注数据来使用DPO模型调整LLM参数，DPO本质上是个二分类，就是从人工标注的<Prompt，Good Answer，Bad Answer>三元数据里学习，调整模型参数鼓励模型输出Good Answer，不输出Bad Answer。这算完成了一个迭代轮次的Post-Training。

上述过程会反复迭代几次，每次的流程相同，不同的地方在于拒绝采样阶段用来对给定Prompt产生回答的LLM模型，会从上一轮流程最后产生的若干不同DPO模型（不同超参等）里选择最好的那个在下一轮拒绝采样阶段给Prompt生成答案。很明显，随着迭代的增加DPO模型越来越好，所以拒绝采样里能选出的最佳答案质量越来越高，SFT模型就越好，如此形成正反馈循环。

可以看出，尽管RLHF 和DPO两种模式都包含RM，但是用的地方不一样，RLHF是把RM打分用在PPO强化学习阶段，而LLaMA 3则用RM来筛选高质量SFT数据。而且因为拒绝采样的回答是由LLM产生的，可知这里大量采用了合成数据来训练SFT模型。


**特殊token**：使用新的多消息聊天协议的工具使用，该协议使用各种特殊标头和终止令牌。


**奖励建模**
- 和Llama-2相比，这次RM的一个变化是移除了训练时加入的margin term（用于把chosen和rejected response区分得更开），因为随着模型规模的增大，加入margin term收益越来越小了。
- 同Llama-2一样，preference data中只有区分度比较大的数据对用于训练RM。
- 过滤掉具有相似响应的样本后的偏好数据
- 数据上，除了常规的chosen和rejected response之外，还引入了第三种 -- “edited response”，即在chosen的基础上通过（人工）编辑，进一步提升这条response的质量。这样每条ranking sample就可能有3条response（edited > chosen > rejected）。
- 训练的时候，prompt和对应的多条随机打乱的response拼接在一起训练（prompt + resp_1 + resp_2 + resp_3），这和通常的做法，即每个response都拼接prompt有些不同（prompt + resp_1, prompt + resp_2, prompt + resp_3）。从结果上来看，都拼接到一起在accuracy上没有什么损失，而训练效率更高。


**监督微调**
- 训练好的RM模型会用于rejection sampling，对human annotation prompt的不同生成结果进行过滤。与真实数据和合成数据相结合得到SFT数据
- masking loss on prompt tokens
- 使用 1e-5 的 lr 训练 8.5K 到 9K 步。实践上这样的参数设置在多轮的post-training中都能保持较好的效果。

直接偏好优化
- 会用在上一轮post-training得到的最佳模型收集偏好数据对，这样能使得偏好数据的分布和强化学习时的policy model更一致。
- 除了DPO以外，Meta也尝试了一些on-policy的方案，如PPO。但是相对来说，DPO消耗更少的计算资源，并且效果也更好，特别是在instruction following的能力上，所以还是选择在post-training使用DPO。
- lr=1e-5 和 β=0.1
- Masking out formatting tokens in DPO loss。把特殊token比如header和termination token屏蔽，不用于计算训练loss。因为使用这些token计算loss会使得模型在生成时，出现如复读机或者在不合适的地方截断的情况。这可能就是因为chosen repsponse和rejected response同时包含的这些特殊token，让模型在训练时要同时增大和较小它们的likelihood，导致冲突。
- Regularization with NLL loss。除了DPO的常规loss，Meta额外加入了NLL损失项，这和《Iterative reasoning preference optimization》的做法类似，类似正负样本的精细化调权手段。这也有点像PPO里加入next token prediction loss，能使训练更加稳定，并能保持SFT学到的生成格式，并保持chosen response的log probability不下降（《Smaug: Fixing failure modes of preference optimisation with dpo-positive》）。其缩放系数为 0.2。

模型融合
- 参考《Averaging weights leads to wider optima and better generalization》《Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time》和《Branch-train-merge: Embarrassingly parallel training of expert language models》，在RM、SFT和DPO阶段，分别把“用不同版本的数据和超参训练得到模型”进行平均，以获得最终模型。

迭代式训练6次
- 同LLama2，最新的模型采样最新的偏好数据，武当总云梯，左脚踩右脚

**偏好数据**

首先，在每轮训练完后部署一批“在不同数据、超参、训练策略上训练”得到的模型，这些模型有各自的特点，比如有些擅长写代码，有些擅长数学推理。

对于每个user prompt，从这些模型里采样两个response。之后标注人员给每对chosen和rejected response分成4类：

- significantly better
    
- better
    
- slightly better
    
- marginally better
    

过程中标注人员也可以对chosen response进一步编辑，获得更好的response。

下表给出了偏好数据的的统计：

![](img/Pasted%20image%2020240727114410.png)

相比Llama-2的数据，Llama-3所用的prompt和response的长度都有所增加，这说明Llama-3的任务复杂度提升了。

在每一轮的post-training之后，都会分析当前版本模型效果不好的领域，并针对这些领域提升prompt的复杂度。

每轮post-training中，训练RM的时候，会使用所有来自不同轮所收集到的偏好数据。而DPO训练则只会用到最新的偏好数据。

对于RM和DPO，都只使用分类为significantly better 和 better的数据进行训练，而另外两类质量相近的偏好数据对则被丢弃。


**SFT Data**

SFT数据主要有这几个来源：

- 人工收集的prompt，以及对应的通过拒绝采样得到的response
    
- 特定领域的合成数据（后面capacities部分会讲到）
    
- 少量人类真实数据
    

1、拒绝采样（RS）

在RS阶段，每个prompt会从“最新的/领域最佳的chat模型”采样K个回复（一般10~30个），然后用RM选出最佳回复（《Constitutional AI: harmlessness from AI feedback》）。

在靠后轮次的post-training里，RS引入了控制风格、格式、语气等特性的system prompt以更精细地控制数据质量。不同的领域（如代码、推理、工具使用等）可能会采用不同的prompt。

PagedAttention 用于实现高效的拒绝采样

2、数据组成

下表给出了helpful数据中每个大类别的数据统计：

![](img/Pasted%20image%2020240727114525.png)

数据清洗：在post-training的前几轮中，研究人员发现数据中混入了一些包含过量emoji或者感叹号之类的数据，因此用专门的规则对发现的低质量pattern进行了清洗。此外有些数据还有overly-apologetic（比如模型经常回复“我很抱歉”）的问题，也会有规则识别如“I‘m sorry”这样的内容，并降低这类数据的比例。

数据裁剪：我们还应用一系列基于模型的技术来删除低质量的训练样本并提高整体模型性能
- 主题分类。微调llama3 8B。包括粗粒度和细粒度。
- 质量打分。用reward模型和 “model-as-judge”的方法得到分数。取 RM 分数前四分之一的数据是高质量的。“model-as-judge”对每个样本进行三分制评分（使用特定的prompt（不同领域prompt可能不同））：一般英语数据（准确性、指令遵循和语气/表达）；编码数据（错误识别和表达）采用两分制评分用户意图），并将获得最高分数的样本视为高质量。 RM 和基于 Llama 的分数具有很高的分歧率，我们发现结合这些信号可以在我们的内部测试集上产生最佳的召回率。最终，我们选择由 RM 或基于 Llama 的过滤器标记为高质量的示例。
- 难度打分。因为我们还对模型更复杂的示例的优先级感兴趣，所以我们使用两种难度度量对数据进行评分：Instag（Lu 等人，2023）和基于 Llama 的评分。对于 Instag，我们提示 Llama 3 70B 对 SFT 提示执行意图标记，其中更多意图意味着更多复杂性。我们还提示 Llama 3 以三分制衡量对话的难度（Liu et al., 2024c）。
- 语义去重：我们首先使用 RoBERTa 对完整的对话进行聚类（Liu et al., 2019b），并在每个聚类中按质量分数 × 难度分数对它们进行排序。然后，我们通过迭代所有排序的示例来进行贪婪选择，并且仅保留最大余弦相似度小于集群中迄今为止看到的示例的阈值的示例。

## 能力提升

### 代码

代码上，要提升的目标语言包括：Python, Java, Javascript, C/C++, Typescript, Rust, PHP, HTML/CSS, SQL, bash/shell。

代码能力提升的方法包括训练code expert、生成数据用于SFT训练、通过system prompt调整格式以及使用quality filter过滤低质量数据。

#### Expert training

首先，在主预训练模型的基础上，增加1T的代码继续预训练，其中>85%的样本是代码数据。然后采用和CodeLlama类似的方法训练code expert。

在训练的最后几千个step，会加入repo-level的长代码数据，以提升code expert的长窗口能力（ 16K）。

继续预训练之后会采用前面提到的方法进行post-training，只是所用数据主要是代码数据。

得到的code expert用于：

- 在主模型的post-training中获取高质量的代码数据
    
- code prompt的rejection sampling
    

#### 合成数据

生成的代码会存在一些问题，包括难以遵循指令、语法错误、生成错误代码和难以修复错误等。

虽然人工标注理论上可以解决这些问题，但合成数据的成本更低、更方便扩展到更大规模，因此还是使用Llama 3和code expert生成大量SFT合成数据。

#### 代码生成的方法

基于以下三个方法，一共生成了超过2.7M的代码SFT数据。

1、执行反馈

Llama-3的8B和70B模型在用更大的模型（比如405B）所生成的数据训练时，获得了明显的收益。但是405B模型在用自己生成的数据训练之后（毕竟这个规模下很难有更大的模型了），不仅没有提升，甚至还有些退化。

为了解决这个问题，Meta引入了execution feedback，来对代码进行正确性校验，并让模型从错误中学习。

具体来说，用以下的过程获得了1M左右的训练数据：

（1）生成问题描述

这一步生成大量涵盖广泛主题的编程问题描述。为了增加多样性，从不同的来源随机抽取代码片段，然后根据代码片对生成对应的问题描述。（《Magicoder: Empowering code generation with oss-instruct》）

（2）Solution生成

这一步用Llama-3生成代码问题的答案。

这个过程中，会在prompt里加入优质代码的general rule，并要求模型在注释里给出思路。这两个做法能有效促进代码质量的提升。

（3）正确性分析

检查生成的solution正确性包括两个方面。

一是静态分析，即通过parser和linter保证基础的语法正确性。

另一个是动态检查，通过让模型给代码生成单元测试并执行来判断代码的正确性。

（4）错误反馈 & 迭代修正

对于有问题的代码，并不是直接舍弃，而是让模型修改优化。

通过prompt把错误信息给到模型，不断迭代修改，直到代码通过所有单元测试。

原数据里大概有20%的样本通过这样的修改才通过测试，说明如果不对正确性进行校验的话，会在训练数据里引入大量的错误信息。

（5）微调 & 迭代优化

微调迭代了多个round，每个round产生的模型都用来生成新的数据给下一次迭代训练。

2、programming language translation

不同语言的代码数据量有不平衡的情况，因此Meta基于Llama-3把高频语言的代码“翻译”成低频语言的数据，并通过syntax parsing, compilation, execution等来保证翻译数据的质量。（类似《Breaking language barriers in multilingual mathematical reasoning: Insights and observations》的思路）

3、backtranslation

在代码相关的能力如documentation、debugging和explanation上，执行+反馈的做法并不适用。

因而采用一个多步方法backtranslation，从代码片段开始：

- Generate：让模型先生成，比如文档，或者代码功能解释
    
- Backtranslate：再要求用生成的文档或者功能说明生成代码
    
- Filter：如果第二步生成的代码和原代码一致性够高，则说明生成的文档/代码解释好用，可作为训练数据
    

通过backtranslation，大约获得了1.2M的documentation、debugging和explanation等数据。

#### 其他

1、system prompt

使用代码专用的system prompt可以提高生成数据的质量，下图是一个样例，右边多了comment，变量名更为合理，还更省空间。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/Aj0FZbibW464iactcJwYsUrZcBMI7BgI5ZPcgBlGP4CyfD2ghlBrzBibr4gN47MickYydicpeAatcLhgsEBlWJeBpuQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

2、Filtering training data with execution and model-as-judge signals

rejection sampling的过程会遇到有问题的代码，但是检验这些代码并不是想象中的那么straightforward，比如生成的内容可能包含了不能执行的内容（如伪代码），或者用户要求生成的是完整代码的一个小片段（无法单独执行），这些都无法直接通过单元测试来检验。

因此使用“model-as-judge”的方法，即通过Llama-3对生成内容做正确性和风格好坏的二分类，只有当二者都被分为好，对应的代码数据才会被使用。

但是这种方法会倾向于保留简单任务（因为复杂的任务更容易出现问题），导致模型在复杂问题上的能力受损。因此研究人员还专门人为地修改了困难任务上的response，直到这些response符合Llama-3的要求。


### 多语言

Llama-3支持8种语言：German, French, Italian, Portuguese, Hindi, Spanish, Thai。

#### Expert training

用包含超过90%的多语言（即除英语以外的语言）的data mix，对主预训练模型做继续预训练，之后再进行同code expert类似的post-training。得到的多语言expert model用于收集高质量的非英文数据。

#### 多语言数据收集

多语言的SFT数据中，包含：

- 2.4%的人类数据
    
- 44.2%的NLP task数据
    
- 18.8%来自rejection sampling
    
- 34.6%来自translated reasoning data
    

1、人类数据

这部分都是从native speaker收集的，大部分包含开放的多轮对话，代表了真实世界的数据。

2、NLP task

- 把常规NLP任务改写成对话格式。
    
- 为了提升语言的alignment，使用了来自《Parallel global voices: a collection of multilingual corpora with citizen media stories》和Wikimedia的parallel text。
    
- 用LID based filtering和Blaser2.0 （《Seamlessm4t—massively multilingual & multimodal machine translation》）清洗掉低质量数据。
    

3、拒绝采样数据

相比英文数据，多语言数据的RS做了几点改动：

- Generation：在post-training的前几轮中，使用0.2~1.0的随机温度来生成回复，以提升多样性。而在最后一轮中，则使用0.6的温度，以保持生成结果中创新性和流畅性的平衡。
    
- Selection：在RM模型之前，对prompt和response做了语言检查，保证语言的匹配性（比如不会出现一种语言问，另一种语言回答，除非明确要求）。
    

4、翻译数据

大部分数据都没有做翻译，以避免引入翻译腔等问题，除了一个例外：synthetic quantitative reasoning data。

这类数据的语言描述通常比较简单，所以翻译之后没有什么质量问题，而推理数据可以帮助改善多语言的定量推理能力。


### 数学和推理

reasoning被定义为“执行多步计算并得出最终正确答案”的能力。

reasoning能力的训练有几个挑战：

- 缺少prompt：这种高难度的任务数据相对较少
    
- 缺少正确的CoT：reasoning任务一般有多步，包含这些多步CoT的正确答案的数据也不多
    
- 错误的中间步骤：基于模型生成的CoT很容易有错误的中间步骤
    
- 使用外部工具：教会模型使用外部工具能极大提升效果，但这并不容易
    
- 训练与推理的差异：推理的时候可能需要在中间和用户进行交互获取反馈，这可能和训练数据不完全一致
    

针对这些问题，Meta给出以下解决方案。

1、解决缺少prompt的问题

为了解决缺少prompt的问题，研究人员从数学相关的context抽取数据片段并转换为对话形式，用于SFT。

对于模型表现不好的数学领域，专门收集了人类的prompt。为此构建了数学相关的分类体系（《Metacognitive capabilities of llms: An exploration in mathematical problem solving》），并让人类专家提供相应的prompt和问题。

2、Augmenting training data with step-wise reasoning traces

就是用Llama-3为一系列的prompt生成step-by-step的解决方案。

对于每个prompt，模型会生成不同数量的结果。这些生成结果随后根据正确答案进行筛选（《Common 7b language models already possess strong math capabilities》）。

此外还进行了自我验证，即使用Llama-3来验证给定的步骤解决方案对于特定问题是否有效。

3、Filtering incorrect reasoning trace

训练outcome RM和stepwise RM来把中间过程错误的数据清洗掉（《Let’s verify step by step》，《Math-shepherd:Verify and reinforce llms step-by-step without human annotations》）。

对于更难的prompt，使用Monte Carlo Tree Search (MCTS)和stepwise RM来生成有效的推理轨迹（《Monte carlo tree search boosts reasoning via iterative preference learning》）。

4、Interleaving code and text reasoning

在文本推理之外，加上python code的执行反馈来对结果正确性做进一步确认（《Tora: A tool-integrated reasoning agent for mathematical problem solving》）。

5、Learning from feedback and mistakes

为了模仿人类的反馈，使用包含错误的生成结果，并要求模型给出修正（《Learning from mistakes makes llm better reasoner》，《Generating sequences by learning to self-correct》，《Self-refine: Iterative refinement with self-feedback》）。

### 长文本

在预训练的最后阶段，训练窗口从8k扩展到128k。

而和预训练相似，在post-training阶段也需要仔细平衡模型的短文本能力和长文本能力。

1、SFT

如果直接把常规的、较短的SFT数据应用在预训练模型上做SFT，会使得预训练阶段得到的长文本能力退化，因此SFT阶段必须加上长数据。

由于让人类来给出超长（128k）的SFT数据，难度太大耗时太长，并不现实，所以主要还是依赖合成数据。

用早期的Llama-3版本来生成长文本关键场景的数据，比如多轮问答、长文本摘要和代码仓库级别的reasoning。

（1）Question answering

从预训练数据里筛选一些长文档，并把它们切分为8k的片段，之后让（短窗口）模型对随机选择的片段生成QA数据。长文本训练时则是把完整的文档和相关的QA作为输入。

（2）Summarization

摘要采用层次化的方式，即先用8k的模型对长文档的每个8k片段进行摘要，多个片段摘要合在一起再进行二次摘要，获得最终结果。

此外，Meta还基于文档摘要生成QA对，要求模型回答那些需要对文档做全面理解的问题。

（3）Long context code reasoning

首先解析Python文件，识别导入语句并确定它们的依赖关系。

接下来，对那些被至少五个其他文件使用的文件，随机删除一个，训练时要求模型识别哪些文件依赖于被删除的文件，并生成所需的缺失代码。

以上这些数据都被分成16K, 32K, 64K和128K的长度，方便进行细粒度的微调。

另外，消融实验发现，在原SFT数据中混入0.1%的这些合成的长文本，对模型的短文本和长文本能力都有提升。

2、DPO

实验发现DPO阶段仅使用短文本并不会对模型长文本能力造成明显影响，可能是因为DPO的更新步数比较少，因此DPO没有特意增加长文本数据。


### 工具使用

使用工具的能力可以拓展模型的能力边界，让模型从单纯的聊天机器人变成有用的智能助手。Llama-3被训练使用以下core tools：

- 搜索引擎：Brave Search
    
- Python interpreter：用于执行生成的代码
    
- Mathematical computational engine：Wolfram Alpha API
    

当用户的query需要用到多个工具时，Llama-3可以给出plan，对工具进行串行调用，并在每次调用之后进行推理整合。

除了core tool之外，Llama-3还有zero-shot的工具调用能力，能根据query调用此前没见过的用户定义的工具。

1、Implementation

Meta将core tools实现为具有不同方法的Python对象。

而zero-shot tool可以作为带有描述、文档（使用示例）的Python函数来实现，模型只需要函数的签名和文档字符串作为上下文来生成适当的调用。

函数的定义和调用都转换为json格式，例如用于Web API调用。

所有工具调用都由Python解释器执行，且需要在Llama-3的system prompt中启用（即告诉模型可以使用哪些工具能力）。core tool可以在system prompt中单独启用或禁用。

2、Data collection

与ToolFormer不同，Llama-3主要依赖人类的标注数据和偏好数据来训练。

人类标注员对模型给出的多个message进行排序，如果两个都不好，就手动编辑一个好的，并让对话继续。

工具使用的训练没有使用rejection sampling，因为实践上来看这样做没有效果。

为了减少标注的人力投入，会先进行基本的finetune让模型具备基本的工具使用能力，并且会先从单轮对话开始，慢慢迭代到多轮对话。

3、Tool datasets

通过以下方法来获取数据。

（1）Single-step tool use

先用few-shot prompt让模型生成core tools的调用，之后要求模型基于用户query和调用结果回答问题。

顺序如下：system prompt, user prompt, tool call, tool output, final answer。

生成的数据里有30%的数据有诸如无法执行，或者有格式问题，就被清除掉了。

（2）Multi-step tool use

先让Llama-3生成至少需要调用2次core tool（可以相同也可以不同）的prompt，然后再用few shot prompt让Llama-3生成一个由交错推理步骤和工具调用组成的解决方案，和ReAct类似。下图是一个多步工具调用的例子：

![](img/Pasted%20image%2020240729120624.png)


（3）File uploads

使用这些格式的文件：.txt, .docx, .pdf, .pptx, .xlsx, .csv, .tsv, .py, .json, .jsonl, .html, .xml。

基于上传的文件，要求模型进行摘要生成、查找并修复错误、优化代码片段、执行数据分析和可视化等任务。下图是一个示例

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/Aj0FZbibW464iactcJwYsUrZcBMI7BgI5ZvhxWbSnyiaEovWWGKs44Kekum6zMdnMRhXA5NtA6eUHdDhpibPm5Vd3g/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

在使用这些合成数据进行了微调之后，Meta进一步收集了多样化且具有挑战性的任务数据，包括多轮交互、三个以上步骤的工具使用，以及工具调用未能得到满意答案的case。

为了让模型避免对简单的query调用工具，使用了简单数学或问答数据集的query，及其不使用工具的response，但在system prompt中激活了工具。这样模型就能学到，即使工具时available的，但是对于简单问题可以不调用工具，避免了工具滥用。

4、Zero-shot tool use data

通过在一个大型的多样化（合成）数据集上微调，提高了Llama-3的zero-shot工具使用能力（函数调用）。

数据包括函数定义、用户query和相应的调用。然后另一批从未见过的工具上进行评测。

（1）Single, nested, and parallel function calling

函数的调用情况有多重，可以是简单的单次调用，也可以是嵌套的（即将一个函数调用作为另一个函数的参数），或者是并行的（即模型返回一个独立的函数调用列表）。

要生成多样化的工具调用数据并不容易（《Toolverifier: Generalization to new tools via self-verification》），因此通过在Stack里（《The stack: 3 tb of permissively licensed source code》）进行挖掘，确保函数调用和定义是真实的。即从里面提取出真实的函数调用和定义，过滤掉如文档有问题或者无法执行的函数，之后用Llama-3生成函数调用的query。

（2）Multi-turn function calling

参照《Api-bank: A comprehensive benchmark for tool-augmented llms》的做法，为带有函数调用的多轮对话生成了合成数据。

通过使用不同的prompt，让Llama-3扮演不同的agent，分别用于生成domains, APIs, user queries, API calls, 和 responses。

### 事实性

Hallucination依然是大模型的一个问题。即使在模型不怎么了解的领域，模型也会给出很自信的回答，这就会给大模型的使用带来风险。

Meta遵循的原则是，post-training应该使模型 “know what it knows” ，而不是增加知识（《Does fine-tuning llms on new knowledge encourage hallucinations?》，《Linguistic calibration through metacognition: aligning dialogue agent responses with expected correctness》）。

主要方法是生成数据 -- 生成与预训练数据中存在的实际数据保持一致的微调数据。

为了实现这一点，Meta开发了一种基于Llama-3的in-context能力的knowledge probing技术。

这个数据生成过程包括以下步骤：

- 从预训练数据抽取一个片段
    
- 用Llama-3对这个片段生成一个事实性问题
    
- 用Llama-3采样这个问题的答案
    
- 用原片段的context对生成答案的正确性进行打分
    
- 对生成结果的informativeness进行打分
    
- 用Llama-3生成对“信息丰富但错误的response”的refusal
    

Meta使用knowledge probing生成的数据，来鼓励模型只回答它有知识的问题，并拒绝回答它不确定的问题。

此外，预训练数据并不总是一致或正确的。因此还专门收集了一个数据集，处理那些事实矛盾或不正确陈述普遍存在的敏感话题。
### 可控性

可操控性是指引导模型的行为和结果以满足开发者和用户需求的能力。

由于Llama-3是一个通用的基础模型，它应该具备在不同使用场景下的可操控性。

Meta主要通过system prompt来增强Llama-3的可操控性，特别是在response长度、格式、语气等方面。

数据收集上，首先要求annotator为Llama-3设计不同的system prompt，然后，annotator与模型进行对话，评估模型在对话过程中遵循system prompt中定义指令的一致性，并收集偏好数据。

以下是一个增强可操控性的system prompt例子：

You are a helpful and cheerful AI Chatbot that acts as a meal plan assistant for busy families. The family consists of 2 adults, 3 teenagers, and 2 preschoolers. Plan two or three days at a time and use leftovers or extra ingredients for the second day’s plan. The user will let you know if they want two or three days. If they don’t, assume three days. Each plan should include breakfast, lunch, snack, and dinner. Ask the user if they approve of the plan or need adjustments. After they approve provide a grocery list with family size in mind. Always keep family preferences in mind and if there’s something that they don’t like provide a substitution. If the user is not feeling inspired then ask them what’s the one place they wish they could visit on vacation this week and then suggest meals based on that location’s culture. Weekend meals can be more complex. Weekday meals should be quick and easy. For breakfast and lunch, easy food like cereal, English muffins with pre-cooked bacon, and other quick easy foods are preferred. The family is busy. Be sure to ask if they have essentials and favorites on hand like coffee or energy drinks so they don’t forget to buy it. Remember to be budget-conscious unless it’s a special occasion.

## 视觉

1. 数据
- 图像编码器的图像文本和视频文本对

- 通过使用 n-gram 将图像标题对重新采样为约 350M 个较小样本量而创建的退火数据集。

- 使用视觉基础、屏幕截图解析、问答对、合成字幕、通过 LaTex 或 markdown 表示的图表、表格、方程式等合成生成的图像收集的额外 150M 个样本

- 视频时长从 16-21 秒不等，分辨率从 320p 到 4K 不等

2. 模型
- 三个组件：图像编码器、图像适配器和视频适配器

图像编码器
- ViT/H-14 变体，630M 个参数，在 2.5B 图像文本对上训练了五个时期。图像大小为 224x224，分成 16x16 个块

- 从前几层进行多层特征提取，并注入到最后一层以保留细粒度的定位信息

- 40 个变压器块，带有 8 个门控注意层

图像适配器
- GQA 注意
- 仅交叉注意层就有近 100B 个参数（wt..😱🫤😵‍💫🫨）
- 分两个阶段训练

视频适配器
- 从视频中均匀采样 64 帧，每个帧都由图像编码器处理
- 使用时间聚合器（感知器重采样器）的时间信息，以及一些额外的交叉注意层

3. 预训练
- 对于图像，从预训练的文本模型和视觉编码器权重开始。

- 视觉编码器解冻，文本冻结，并使用 6B 图像-文本对进行训练，批处理大小为 16,384，余弦计划，lr 10e-4，权重衰减为 0.01

- 对于视频，从预训练和退火权重的图像开始。视频聚合器从头开始训练，而其他一切都冻结

4. 后训练
- 数据程序与文本的情况大致相同
- 学术数据集、人工注释和合成数据
- 对于质量调整，整理了一个小而高度选择性的极高质量数据。对这些数据进行 DPO 以提高响应质量，帮助改进人工评估

## Speech

1. 数据
- 1500 万小时的多语言语音数据用于预训练
- ASR 训练数据包含 23 万小时的手动转录
- 涵盖 34 种语言的语音记录。AST 训练数据包含 9 万小时的双向翻译（33 种语言 -> 英语和英语 ->33 种语言）
- 2.5 万小时的合成数据

2. 模型
- 语音编码器和语音适配器
- 语音编码器是具有 1B 个参数的 Conformer 模型。

- 模型的输入是 80 维梅尔频谱图，由 4 步堆叠层处理，然后进行线性投影，然后传递给 conformer 编码器

- 每个 Conformer 层的潜在维度为 1536，
由两个维度为 4096 的 Macron-net 样式前馈网络、一个内核大小为 7 的卷积模块和一个旋转注意模块组成

- 另一方面，语音适配器包含大约 1 亿个参数。它由卷积层、旋转 Transformer 层和线性层组成。

3. 训练
- 预训练利用未标记数据来训练语音编码器

- 使用自监督 BEST-RQ 算法预训练编码器

- 将 32 帧长度的掩码（概率为 2.5%）应用于输入梅尔频谱图

- 通过堆叠 4 个连续帧、将 320 维向量投影到 16 维空间并在 8,192 个向量的码本内使用余弦相似度度量执行 NN 搜索来量化梅尔频谱图特征

- 16 个不同的码本
- 出于效率原因，仅在掩码帧上使用多 softmax 损失。

- 编码器训练了 500K 步，全局批量大小为 2,048 条话语。

- 在第二阶段，完成 SFT，其中适配器和预训练编码器与语言模型集成并与其联合训练，而 LLM 保持冻结。

## 总结

- Llama-3不仅仅是一个模型，而且是一个巨大的工程
    
- 大量的工作仍然是在数据上，而且post-training的权重提高了许多
    
- 对各个领域数据的细致整理，也提醒开发者们，目前阶段的“通用能力”说到底还是多任务训练，而多任务，就需要一个领域一个领域踏实优化

不断提升小模型效果的三个关键因素：

**第一个武器是预训练阶段增加训练数据数量和质量**。

**第二个武器是模型蒸馏**。

原先小模型预训练目标是根据上文context信息正确预测Next Token，而蒸馏则改成Teacher把自己做相同上下文做Next Token预测的时候，把Token词典里每个Token的生成概率都输出来，形成Next Token的概率分布，这就是Teacher交给Student的额外附加信息，小模型从原先的预测Next Token改为预测Next Token的概率分布，要求和Teacher输出的分布尽量一致，这样就学到了Teacher的内部信息。

**第三个武器是Annealing Data**。核心思想就是在预训练的最后阶段，对高质量数据比如数学、逻辑、代码数据进行上采样，增加其影响。LLama 3技术报告说这招对405B模型不怎么起作用，但是对8B小模型在逻辑代码能力方面有明显提升。

其实从ChatGPT火了以后看各种大模型的技术报告，包括LLama系列模型在内，可以看出大模型之所以能力仍在快速提升，主要驱动力有三个：**首先就是不断扩大模型和数据规模（Scaling Law）**。除此外，在数据方面有两个发展趋势：**一个是越来越强调数据质量的作用**，各种数据筛选方法和工具越来越多，保证质量是第一位的（这个早在Google T5时代就能推出这个结论，目前只是进一步验证并延续这个思路而已）。**第二个是不断增加数学、逻辑、代码这种能够提升大模型理性能力的数据配比比例**，包括在预训练阶段（增加预训练数据此类数据比例，且在预训练后面阶段来上采样此类数据，就是说同样数据多执行几遍，以增加其对模型参数影响的权重）和Post-Training阶段（增加此类数据占比，Llama3的经过instruct的模型比仅做预训练模型相比，各种尺寸的效果提升都很大）皆是如此。

目前看，在通用数据快被用完情况下，**第三个因素会成为之后大模型进步的主导力量**，包括使用数学、逻辑、代码合成数据在Post-Training阶段的应用，目前技术也越来越成熟，其质量和数量会是决定未来大模型效果差异的最关键因素。


## 参考资料

[ LLama 405B 技术报告解读](https://mp.weixin.qq.com/s/8RYqgfuYga0YU8H8XqNNOA)

[Meta Llama 3.1-405B AI 模型多项跑分超越 GPT-4o，如何评价该款模型？-张俊林的回答](https://www.zhihu.com/question/662354435/answer/3572364267)

[Llama3.1--post-training要点一览](https://mp.weixin.qq.com/s/wSVi2csJ9weL57iB_2XgCg)



