## 2024.4

## 模型

以中文为核心而不是英文为核心依旧可以训一个"现在是不错，以后可能是很好"的LLM！
	我们开源了主要通过清洗CC和OCR的中文预训练语料800B MAP-CC，这是当前中文NLP开源社区最大的经过深度清洗的中文开源预训练数据集之一！
	我们开源了数据清理的全流程pipeline！
	我们开源了经过SFT与DPO后的CT-LLM！
	我们开源了一个较小的类MT-Bench中文效果衡量的Benchmark，CHC-Bench！
	我们开源了训练全流程的intermediate ckpts供大家分析！
	最重要的是，我们证明了中文为核心也可以训一个很不错的LLM，也可以在英文(其他语言上)涌现能力！
	CT-LLM is now available:
	paper: https://arxiv.org/pdf/2404.04167.pdf
	twitter: https://twitter.com/GeZhang86038849/status/1777163413183193296
	huggingface collection: https://huggingface.co/collections/m-a-p/chinese-tiny-llm-660d0133dff6856f94ce0fc6


### MOE相关的模型

[A21 Labs宣布开源520亿参数的全新混合专家大模型（Mixture of Experts，MoE）Jamba：单个GPU的上下文长度是Mixtral 8x7B的三倍](https://www.datalearner.com/blog/1051711641710005)

[开源大模型再上台阶：Databricks开源1320亿参数的混合专家大模型DBRX-16*12B，评测超Mixtral-MoE！](https://mp.weixin.qq.com/s/dkx0UU2PgR_CpaVa88KcZQ)

[重磅！阿里开源自家首个MoE技术大模型：Qwen1.5-MoE-A2.7B，性能约等于70亿参数规模的大模型Mistral-7B](https://mp.weixin.qq.com/s/XHFjybR3GIg4RIpBlndVGg)

[马斯克旗下xAI发布Grok-1.5，相比较开源的Grok-1，各项性能大幅提升，接近GPT-4！](https://www.datalearner.com/blog/1051711675314896#google_vignette)

[MoE架构模型大爆发！元象科技XVERSE开源256亿参数模型XVERSE-MoE-A4.2B，评测结果接近Llama1-65B](https://mp.weixin.qq.com/s/g1le9yGBSGwe6WqeeaVSEw)

### 长文本

[超长文本无损能力压测！中文大模型“大海捞针”首批结果公布](https://mp.weixin.qq.com/s/QgoRf2LB-7vc3vTFOHJkpw)

## 2024.4.22

1. #数据 开源了15T的高质量网络数据FineWeb，对2013-2014期间的cc进行过滤和去重：https://twitter.com/gui_penedo/status/1781953413938557276 

### 2024.4.23
1. [好样本，事半功倍：使用样本设计工程 (SDE) 来构造更好的大模型下游微调样本](https://mp.weixin.qq.com/s/QbiTwDvXLJ_Bbsi3xFOgkQ)

