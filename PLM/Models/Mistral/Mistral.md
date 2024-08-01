---
title: Mistral
created: 2024-01-16
tags:
  - 大模型
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2023
institution:
  - mistral
---

# Mistral 7B

官方博客：https://mistral.ai/news/announcing-mistral-7b/  

mistral 7B 论文：https://arxiv.org/abs/2310.06825

![](img/Pasted%20image%2020240316142536.png)

Mistral7b instruct v0.2相比0.1，长度扩充到32k，sliding window从4k改成了null(也就是使用了full attention)，rope_theta改成了100w，并且效果还不错。0.2是在0.1基础上训练的。


## 核心亮点
### Sliding Window Attention

Mistral 采用的 window size 为 4096，而后一共有 32 层layer，那么采用 SWA 之后，理论上在进行 attention 的时候，理论上可以收集到约 131K tokens 的信息。(虽然论文里提到的 window size 是 4096，但 官方提供的 huggingface 上的权重中 max_position_embeddings 为 32768，且在新一点的版本中，比如 mistral-7b-instruct-v0.2 ，都不采用 sliding window 了)

![](img/Pasted%20image%2020240316142514.png)

由于代用了固定的 attention 窗口大小，因此我们只需要一个大小为 W=window size 的 cache ，在计算第 i 个 token 的 cache 的时候，只需要覆盖 cache 中 i mod M 位置上的 hidden state 即可。

参考 huggingface 的 mistral 实现，Sliding window attention 通过 attention_mask 来控制：

```python
# huggignface mistral attn mask 实现  
def _update_causal_mask(  
self,  
    attention_mask: torch.Tensor,  
    input_tensor: torch.Tensor,  
    cache_position: torch.Tensor,  
    past_key_values:Cache,  
):  
# ... 省略部分无关代码  
    past_seen_tokens = cache_position[0]if past_key_values isnotNoneelse0  
    using_static_cache = isinstance(past_key_values,StaticCache)  
    using_sliding_window_cache = isinstance(past_key_values,SlidingWindowCache)  
  
    dtype, device = input_tensor.dtype, input_tensor.device  
    min_dtype = torch.finfo(dtype).min  
    sequence_length = input_tensor.shape[1]  
# SlidingWindowCache  
if using_sliding_window_cache:  
        target_length = max(sequence_length,self.config.sliding_window)  
# StaticCache  
elif using_static_cache:  
        target_length = past_key_values.get_max_length()  
# DynamicCache or no cache  
else:  
        target_length =(  
            attention_mask.shape[-1]  
if isinstance(attention_mask, torch.Tensor)  
else past_seen_tokens + sequence_length +1  
)  
  
if attention_mask isnotNoneand attention_mask.dim()==4:  
# in this case we assume that the mask comes already in inverted form and requires no inversion or slicing  
if attention_mask.max()!=0:  
raiseValueError('Custom 4D attention mask should be passed in inverted form with max==0`')  
        causal_mask = attention_mask  
else:  
        causal_mask = torch.full(  
(sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device  
)  
        exclude_mask = torch.arange(target_length, device=device)> cache_position.reshape(-1,1)  
ifself.config.sliding_window isnotNone:  
ifnot using_sliding_window_cache or sequence_length >self.config.sliding_window:  
                exclude_mask.bitwise_or_(  
                    torch.arange(target_length, device=device)  
<=(cache_position.reshape(-1,1)-self.config.sliding_window)  
)  
        causal_mask *= exclude_mask  
        causal_mask = causal_mask[None,None,:,:].expand(input_tensor.shape[0],1,-1,-1)  
if attention_mask isnotNone:  
            causal_mask = causal_mask.clone()# copy to contiguous memory for in-place edit  
if attention_mask.dim()==2:  
                mask_length = attention_mask.shape[-1]  
                padding_mask = causal_mask[:,:,:,:mask_length]+ attention_mask[:,None,None,:]  
                padding_mask = padding_mask ==0  
                causal_mask[:,:,:,:mask_length]= causal_mask[:,:,:,:mask_length].masked_fill(  
                    padding_mask, min_dtype  
)  
  
return causal_mask
```

![](img/Pasted%20image%2020240316142610.png)

![](img/Pasted%20image%2020240316142626.png)

### GQA (Grouped Query Attention)

[MQA&GQA](../../BaseStruct/GQA/MQA&GQA.md)

# Mixtral 8\*7B

论文：https://arxiv.org/abs/2401.04088  
huggingface 模型权重：https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1  
官方博客：https://mistral.ai/news/mixtral-of-experts/  
huggingface 模型代码：https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py  
混合专家模型基础（推荐）：https://huggingface.co/blog/zh/moe

官方给出的评分来看，mixtral 8\*7 和 GPT3.5 有的一比。

- • 发布时间：23年12月
    
- • 模型大小：8 个 expert MLP 层，一共45B 大小。
    
- • 训练：除了预训练外，Mixtral MOE 后续还开源了一个经过 SFT + DPO 微调的版本。
    
- • 模型效果：

![](img/Pasted%20image%2020240801114308.png)

架构：Mixtral 的 MOE 架构类似于，在 MoE 模型中，只有 FFN 层被视为独立的专家，而模型的其他参数是共享的。大致参数为：

![](img/Pasted%20image%2020240801114323.png)

参考 huggingface 中的 mixtral 和 mistral 实现对比，差异在于 mixtral 中将传统 transformer decoder layer 中的 FFN 替换为了 block_sparse_moe。

![](img/Pasted%20image%2020240801114340.png)

mixtral 论文中提到专家分配在不同主题（如ArXiv论文、生物学和哲学文档）中没有明显的模式，只有在DM数学中显示出边际上的差异，这可能是由于其数据集的合成性质和有限的自然语言覆盖范围所致。router 在某些句法结构上表现出一定的结构化行为（比如 python 的 self 等），同时连续标记通常被分配给相同的专家。

# Mixtral 8\*22B

官方博客：https://mistral.ai/news/mixtral-8x22b/  
huggingface 开源模型：https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1

- • 架构：架构与 mixtral 8\*7B 架构一样，在 huggingface 中使用的都是MixtralForCausalLM ，但 22B 的各方面参数大一点，比较特别的是 context window 从 32k 升级到了 65k， vocab_size 也更大一些。
    
- • 支持 function calling，不过好像没有透露具体的 function calling 训练细节。
    
- • 数学和 coding 能力明显超越 llama2 70B。
    
- • 似乎对中文的支持不是很好。

![](img/Pasted%20image%2020240801114739.png)

![](img/Pasted%20image%2020240801114746.png)

Mistral 团队开源的模型，都比较注重 coding 和 math 的能力，Mixtral 系列的模型在这方便表现也是比较好

# Mistral Nemo

官方博客：https://mistral.ai/news/mistral-nemo/  
huggingface 模型权重：https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407

Mistral Nemo 使用的也是 MistralForCausalLM 架构，与 mistral 7B 的差别为：Mistral Nemo 的 hidden_size 从 4096 变为 5120；max_position_embeddings 变为 1024000，num_hidden_layers 增加到 40， vocab_size 增加到 131072，不用 sliding window。

此外，Mistral Nemo 支持 function calling，采用了 Tekken 作为 tokenizer，比 SentencePiece 更高效（压缩率更高，官方描述是~30% more efficient at compressing，不确定是哪个方面的 efficient）

NVIDIA 在[这个博客](https://blogs.nvidia.com/blog/mistral-nvidia-ai-model/)中提到：Mistral Nemo 采用这样的设计，是为了能够适配单个NVIDIA L40S、NVIDIA GeForce RTX 4090或NVIDIA RTX 4500 GPU。模型采用 Megatron-LM训练，用了 3,072 个 H100 80GB 。

但光采用 FP16 加载整个 Mistral Nemo 就需要花 23 GB 显存，要是要跑满整个 context window size，除了量化外，还是得需要采用 offload 或者其他方法来推理

![](img/Pasted%20image%2020240801114913.png)

# Mistral Large 2

官方博客：https://mistral.ai/news/mistral-large-2407/  
huggingface 模型权重：https://huggingface.co/mistralai/Mistral-Large-Instruct-2407

Mistral Large 2，参数量 123B，主打多语言以及 coding 能力。采用与 mistral 7B 一样的架构，huggingface 中同样使用 MistralForCausalLM；比较值得注意的是 context window size 为 131072，不用 sliding window。同样支持 function call。

Llama 3.1 刚出不久，就拿 Mistral Large 2 和别人来对比：

![](img/Pasted%20image%2020240801114957.png)

在代码能力上，Mistral large 2 比 llama 3.1 平均效果更好。

![](img/Pasted%20image%2020240801115009.png)

除了 coding 和数学外，在MT Bench 的评分也比 llama 3.1 高，平均生成的回复长度比 llama 3.1 要短

![](img/Pasted%20image%2020240801115023.png)

同时，中文能力相对上一代 mistral large 有大步幅提升：

![](img/Pasted%20image%2020240801115037.png)

## 参考资料
[Mistral instruct v0.2的效果讨论](https://www.reddit.com/r/LocalLLaMA/comments/18g2xs1/mistral7binstructv02/)

[Mistral instruct v0.2太疯狂了](https://www.reddit.com/r/LocalLLaMA/comments/18ghmw2/mistral_7b_v02_is_nuts/)

[从Mistral Nemo到Large2 核心技术详解](https://mp.weixin.qq.com/s/aiPejeckk6w5WDfnZ-jg1Q)

[reddit.com/r/LocalLLaMA/comments/18jslmf/tokens\_per\_second\_mistral\_8x7b\_performance/?rdt=57036](https://www.reddit.com/r/LocalLLaMA/comments/18jslmf/tokens_per_second_mistral_8x7b_performance/?rdt=57036)

