论文：Retrieval Augmentation Reduces Hallucination in Conversation

项目：https://parl.ai/projects/hallucination/

In this work, we study the various components of retrieval-augmented neural architectures for dialogue – retrievers, rankers and encoder-decoders and propose several new variants, while analyzing which methods work well and in which situations they do so.

In particular, we improve downstream performance by employing **Poly-encoder Transformers** (Humeau et al., 2020) for fifiner-grained context-candidate scoring of documents, by proposing an iterative retrieval scheme where the retrieval improves through repetition, by employing end-to-end-trained retrievers in the Fusion-in-Decoder(Izacard and Grave, 2020) technique, and by building a dialogue turn-based retrieval mechanism that avoids the problem of standard retrievers that ignore much of the dialogue context.

## 模型

**RAG和FiD**
- RAG Sequence和RAG Token，参照[RAGRelated](../RAGRelated/RAGRelated.md)

- FiD：DPR或者BM25 are still considered independently within the encoder of the generator model.对于FiD，the decoder can attend to all of the joint document/context representations at the same time when generating a response。训练的时候retriever是不变的（因为不考虑文档概率）。

**Seq2Seq Models**

- BART
- T5
- BlenderBot

**Improving Retrieval**

- Greater context-candidate interaction：context和condidate document之间有更多的interaction
	- DPR是计算文档的embedding和context的embedding之间的点乘相似度，交互性少
	- Full cross-attention效果最好，但是耗费计算量（鉴于候选文档非常的多）
	- Poly-encoders：将context降维成一系列m维度的codes，然后使用attention和候选文档交互，得到一个context representation，再去计算概率
		- DPR-Poly：code re-ranking，Poly-endoer score和DPR score的加权和
		- Joint DPR-Poly：用DPR模型的权重去初始化Poly-encoder
		- PplyFAISS：end-to-end re-ranking：we apply a reduction to the standard Polyencoder context representation to query a FAISS index, where the d(zj ) representations are computed offlfline with the Poly-encoder’s candidate encoder;we subsequently re-rank the retrieved documents with the full Poly-encoder scoring mechanism. We pre-train the Poly-encoder to vary its scoring mechanism between a standard dot-product and a Polyencoder score, so that the reduction is appropriate for FAISS
	- ColBERT：maxsim机制，context encoder的输出和condidate encoder的所有输出进行对比，最后的分数是每个context output的maximum similarity scores之和。
- Iterative Retrieval：两个轮次。第二轮根据第一轮生成的输出来retrieve（ReGReT）（retrieve, generate, retrieve, tune）
- Retriever-less Retrieval：直接将BART和T5的encoder去encodercontext和document。, allowing the full RAG model to propagate error from the token losses to the encoder seen as a retriever and as a generator
	- BREAD (BART-Retriever-Encoder-And-Decoder) for BART-based models, and TREAD for T5-based models.

**Improving Augmented Generation**

- Conditioning on Dialogue Turns：相比问答，对话的context是多轮的，是否需要基于整个对话历史来retrieve document是不确定的，而且大量的信息很可能导致模型confused
	- RAG-Turn：includes a marginalization step within turns of the dialogue prior to marginalization over the whole context. 
	- 允许对话轮次中的每轮都有相关的文档。
	- RAG-Turn Doc-Then-Turn：first marginalize over the documents within a turn, and then marginalize over documents across turns
	- RAG-Turn Doc-Only：We can alternatively consider each turn independently while considering documents within a turn jointly
	- RAG-Turn Token & Sequence：concat所有的turn，考虑所有文档的集合，然后就像原始的RAG一样。分为RAG-Turn Token, RAG-Turn Sequence.

- Improving FiD：加入retriver训练 in a RAG setup。FiD-RAG：models with a DPR-based retriver trained with RAG, and then used with FiD。

## 实验

retrieval是有用的

retrieval solely on the last turn of dialogue is strictly worse than retrieval over the whole context

RAG-Sequence model is good at incorporating knowledge but poor at retaining conversational ability

The RAG-Turn models bridge this gap and offer a balanced trade-off

The RAG-Turn Doc-Then-Turn method yields F1 scores higher than the RAG-Sequence model, and higher Knowledge F1 scores than the RAGToken model; the Doc-Only RAG-Turn method achieves the highest F1 on both the seen/unseen splits, and improves on Knowledge F1 scores of the RAG-Token model.

FiD is suboptimal outof-the-box for knowledge-grounded dialogue, and incorporating retrievers trained via RAG improves performance considerably.



![](img/Pasted%20image%2020220317232840.png)

![](img/Pasted%20image%2020220317232914.png)


![](img/Pasted%20image%2020220317233202.png)

![](img/Pasted%20image%2020220317233226.png)


![](img/Pasted%20image%2020220317233426.png)

![](img/Pasted%20image%2020220317233438.png)

![](img/Pasted%20image%2020220317233540.png)

![](img/Pasted%20image%2020220317233619.png)







