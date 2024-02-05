## CPM-Ant

### 预训练目标

CPM-Ant 利用文本生成和空白填充作为其预训练目标。如下图所示，文本生成和空白填充都是自回归的。为了构建两个目标的自监督数据，我们分别采用两种掩蔽策略：一种是掩蔽输入的尾部以进行文本生成，另一种是随机掩蔽输入以进行空白填充。掩蔽率服从均匀分布 U(0,1)。对于每个样本，我们将选择随机掩码策略以 50% 的概率进行文本填充，或者选择另一种 50% 概率的尾部掩码策略进行文本生成。

### 预先训练的软提示

在 CPM-Ant 中，我们引入了预训练的软提示来切换生成模式。对于文本生成和空白填充，我们分别设置了特定于目标的软提示。这些软提示由几个可学习的嵌入组成。在预训练过程中，这些软提示被添加到输入中，并激发特定于目标的知识来处理输入。在为下游任务调整 CPM-Ant 时，仅使用与任务相关的软提示来调整 CPM-Ant。

这部分可以参考[PPT](../../../Prompt/PPT/PPT.md)

### 模型结构

![](img/Pasted%20image%2020230210180241.png)

-   **基于提示模板的多段式框架设计（prompt-based multi-segment framework）**：
	
	提示模板（prompt）用以实现模型在理解、生成、摘要等功能之间的快速切换，也易于添加新的功能（学习新的prompt）。
    
    文本段（segment）提供基础的文本编码能力，通过段表示（segment embedding）来影响模型编码模式，复杂的文本编码模式可以拆解成若干基础段的组合，例如编码-解码框架可以拆解成一个编码段+一个解码段的组合。对于每一个基础段，段内采用相对位置编码。
    
    基于提示模板和文本段的组合、拼接，结构简单且易于实现增加、修改模块，进行持续学习和功能更新。
    
-   **共享embedding**：CPM-Ant输入embedding及输出embedding会共享参数，这点与BERT、GPT、T5一致，与T5-1.1、mT5不一致。我们的实验表明共享输入输出的embedding参数会极大增强训练稳定程度，而不共享embedding参数易于导致训练过程出现NaN。
    
-   **无bias**：我们的模型中，各类线性变换及layer norm均不设置bias。一方面源于不设置bias的模型训练稳定性会更强，另一方面也是不设置bias的模型在计算速度及显存消耗上要更占优。
    
-   **动态词表**：对于词表，初始阶段我们将提供大小为30000的中文词表，在后续训练过程中会结合新数据情况进行动态变动。

由于我们希望我们的 CPM-Ant 对各种下游任务足够通用，我们使用统一的编码器架构来同时编码上下文和生成令牌，通过修改注意掩码来控制生成过程，而不是应用 Transformer 的原始编码器-解码器架构:

![](img/Pasted%20image%2020230210182444.png)

其中M是注意力掩码，⊙ 是 Hadamard 乘积。

这部分可以参考[UniLM](../../../UniLM/UniLM.md)和[GLM](../../GLM/GLM.md)

为了进一步保证稳定训练，我们采用Pre-LN Residual结构为：

![](img/Pasted%20image%2020230210182946.png)

### 多段机制&相对位置偏差

我们将 CPM-Ant 的输入分成几个段，每个段用来携带特定的信息。具体来说，我们设计段分别承载软提示、空白填充数据和文本生成数据。更具体地说，对于第 i 个标记，我们分配额外的位置 id Pi和段 id Si。使用位置 id 和段 id，我们计算相对位置偏差如下，

![](img/Pasted%20image%2020230210183128.png)

其中B是注意力层中使用的偏置矩阵，f si,sj (·)是将token之间的相对距离映射到偏置值。直观上，多段相对位置偏差可以充分考虑段相关性来编码相对距离。在 CPM-Ant 中，为了简单起见，如果两个 token 不属于同一段，无论它们的相对距离如何，我们都会分配一个统一的偏置值 b si,sj。

这份部分可以参考相对位置编码[Self-Attention with Relative Position Representations](../../../PositionEncoding/经典式相对位置编码/Self-Attention%20with%20Relative%20Position%20Representations.md)和T5的位置编码： [T5式](../../../PositionEncoding/位置编码.md#T5式)


### NLGtune

输入数据格式
```
{"input": "", "target": ""}
```

一行一个样本，长度超过最大值的时候，会优先截断input，其次截断target（从后往前）

文本最大值=prompt_length+input_length+target_length

task_id默认是1，表示LM任务，0表示MLM任务

模型数据输入的组成：
1. input
- prompt_length * task_id + range(0, prompt_length), 比如prompt_length=32，那么这段输入就是`[32, 33, 34, 35, 36, ...]`
- bos_id+input编码后的id
- target编码后的id+eos_id, target是预测目标
2. length: input的长度
3. position: `[0, 1, ..., length-1]`
4. span: `[0, 0, 0, ...]`, 长度为length
5. context: True和False组成的长度为length的列表，target部分都为False,否则为True
6. segment: prompt部分为0，后面的部分为2
7. target: target部分是原始值，剩余部分是-100的列表，再往前移一位（最后一位是-100）（用来计算损失的）

全部数据组织起来得到的维度是(batch, seqlen)

prompt_length

### NLUtune

输入数据格式
```
{"input": "", "target": 1, "options":[]}
```

target必须是int类型

### forward

input的prompt部分经过prompt_embedding层，剩余部分经过input_embedding，两者拼起来得到input的完整embedding；segment经过segment_embedding，和前面的完整embedding加起来得到hidden_states

attention_mask矩阵：下三角都为1且对于每一行，target之外的部分都为1。（UniLM中的seq-to-seq结构）

`position_bias = self.position_bias(position, position, segment, segment)`
- 输出维度：(batch, num_heads, seq_len, seq_len)
- b_si, sj的矩阵relative_position_bucket=query_segment * self.num_segments + key_segment+ self.num_buckets: (batch, seq_len, seq_len)
- query_segment：(batch, seq_len, 1), segment延伸得到
- key_segment：(batch, 1, seq_len), segment延伸得到
- f_si,sj(pi-pj)的矩阵absolute_position_bucket，参考t5的相对位置编码 [T5式](../../../PositionEncoding/位置编码.md#T5式)


`hidden_states = self.encoder(hidden_states, attention_mask, position_bias)`
- hidden_states: (batch, seq_enc, dim_model)
- attention_mask: (batch, seq_enc, seq_enc)
- position_bias: (num_heads, seq_enc, seq_enc)
- 经过transformer层和layer_norm，得到维度(batch, seq_len, dim_model)

`logits = self.input_embedding.projection(hidden_states)`
- (batch, seq_len, vocab_size)

logits和targets计算损失


## 参考资料

https://www.openbmb.org/en/community/blogs/blogpage?id=98afef2ce45f4fe9a4bc15a66d7ccb92

