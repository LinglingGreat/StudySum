论文：https://arxiv.org/abs/1907.11692

代码：https://github.com/pytorch/fairseq

RoBERTa 模型是BERT 的改进版(从其名字来看，A Robustly Optimized BERT，即简单粗暴称为强力优化的BERT方法)。 在模型规模、算力和数据上，与BERT相比主要有以下几点改进：

- 更大的模型参数量（论文提供的训练时间来看，模型使用 1024 块 V100 GPU 训练了 1 天的时间）
- 更大bacth size。RoBERTa 在训练过程中使用了更大的bacth size。尝试过从 256 到 8000 不等的bacth size。
- 更多的训练数据（包括：CC-NEWS 等在内的 160GB 纯文本。而最初的BERT使用16GB BookCorpus数据集和英语维基百科进行训练）

另外，RoBERTa在训练方法上有以下改进：

- 去掉下一句预测(NSP)任务

- 动态掩码。BERT 依赖随机掩码和预测 token。原版的 BERT 实现在数据预处理期间执行一次掩码，得到一个静态掩码。 而 RoBERTa 使用了动态掩码：每次向模型输入一个序列时都会生成新的掩码模式。这样，在大量数据不断输入的过程中，模型会逐渐适应不同的掩码策略，学习不同的语言表征。
- 文本编码。Byte-Pair Encoding（BPE）是字符级和词级别表征的混合，支持处理自然语言语料库中的众多常见词汇。原版的 BERT 实现使用字符级别的 BPE 词汇，大小为 30K，是在利用启发式分词规则对输入进行预处理之后学得的。Facebook 研究者没有采用这种方式，而是考虑用更大的 byte 级别 BPE 词汇表来训练 BERT，这一词汇表包含 50K 的 subword 单元，且没有对输入作任何额外的预处理或分词。
  

## **Static vs Dynamic Masking**

原始静态mask：

- BERT中是准备训练数据时，每个样本只会进行一次随机mask（因此每个epoch都是重复），后续的每个训练步都采用相同的mask，这是原始静态mask，即单个静态mask，这是原始 BERT 的做法。

修改版静态mask：

- 在预处理的时候将数据集拷贝 10 次，每次拷贝采用不同的 mask（总共40 epochs，所以每一个mask对应的数据被训练4个epoch）。这等价于原始的数据集采用10种静态 mask 来训练 40个 epoch。

动态mask：

- 并没有在预处理的时候执行 mask，而是在每次向模型提供输入时动态生成 mask，所以是时刻变化的。
  

从Table 1中可以看出，修改版的静态mask与BERT原始静态mask效果相当；动态mask又与静态mask效果差不多，或者说略好了静态mask。

基于上述结果的判断，及其动态mask在效率上的优势，论文后续的实验统一采用动态mask。

## Model Input Format and NSP

SEGMENT-PAIR + NSP：

输入包含两部分，每个部分是来自同一文档或者不同文档的 segment （segment 是连续的多个句子），这两个segment 的token总数少于 512 。预训练包含 MLM 任务和 NSP 任务。这是原始 BERT 的做法。

SENTENCE-PAIR + NSP：

输入也是包含两部分，每个部分是来自同一个文档或者不同文档的单个句子，这两个句子的token 总数少于 512 。由于这些输入明显少于512 个tokens，因此增加batch size的大小，以使 tokens 总数保持与SEGMENT-PAIR + NSP 相似。预训练包含 MLM 任务和 NSP 任务。

FULL-SENTENCES：

输入只有一部分（而不是两部分），来自同一个文档或者不同文档的连续多个句子，token 总数不超过 512 。输入可能跨越文档边界，如果跨文档，则在上一个文档末尾添加文档边界token 。预训练不包含 NSP 任务。

DOC-SENTENCES：

输入只有一部分（而不是两部分），输入的构造类似于FULL-SENTENCES，只是不需要跨越文档边界，其输入来自同一个文档的连续句子，token 总数不超过 512 。在文档末尾附近采样的输入可以短于 512个tokens， 因此在这些情况下动态增加batch size大小以达到与 FULL-SENTENCES 相同的tokens总数。预训练不包含 NSP 任务。

BERT采用的是SEGMENT-PAIR（可包含多句话）的输入格式，从实验结果来看，如果在采用NSP loss的情况下，SEGMENT-PAIR 是优于SENTENCE-PAIR(两句话)的。发现单个句子会损害下游任务的性能，可能是如此模型无法学习远程依赖。接下来对比的是，将无NSP损失的训练与来自单个文档(doc-sentence)的文本块的训练进行比较。我们发现，与Devlin等人(2019)相比，该设置的性能优于最初发布的BERT-base结果：消除NSP损失在下游任务的性能上能够与原始BERT持平或略微升高。可能的原因：原始 BERT 实现采用仅仅是去掉NSP的损失项，但是仍然保持 SEGMENT-PARI的输入形式。

最后，实验还发现将序列限制为来自单个文档(doc-sentence)的性能略好于序列来自多个文档(FULL-SENTENCES)。但是 DOC-SENTENCES 策略中，位于文档末尾的样本可能小于 512 个 token。为了保证每个 batch 的 token 总数维持在一个较高水平，需要动态调整 batch-size 。出于处理方便，后面采用DOC-SENTENCES输入格式。

RoBERTa去除了NSP，而是每次输入连续的多个句子，直到最大长度512（可以跨文章）。这种训练方式叫做（FULL - SENTENCES），而原来的Bert每次只输入两个句子。

## Training with large batches

以往的神经机器翻译研究表明，采用非常大的mini-batches进行训练时候，搭配适当提高学习率既可以提高优化速度，又可以提高最终任务性能。最近的研究表明，BERT也可以接受 large batch训练。Devlin等人(2019)最初训练BERT-base只有100万步，batch size为256个序列。通过梯度累积，训练batch size=2K序列的125K步，或batch size=8K的31K步，这两者在计算成本上大约是是等价的。

large batches训练提高了masked language modeling 目标的困惑度，以及最终任务的准确性。large batches也更容易分布式数据并行训练， 在后续实验中，文本使用bacth size=8K进行并行训练。

另外，You et al. (2019)在训练BERT时候，甚至将batch size增大到32k。至于batch size值的极限探索，留待后续研究。

## Text Encoding

字节对编码(BPE)(Sennrich et al.,2016)是字符级和单词级表示的混合，该编码方案可以处理自然语言语料库中常见的大量词汇。BPE不依赖于完整的单词，而是依赖于子词(sub-word)单元，这些子词单元是通过对训练语料库进行统计分析而提取的，其词表大小通常在 1万到 10万之间。当对海量多样语料建模时，unicode characters占据了该词表的大部分。Radford et al.(2019)的工作中介绍了一个简单但高效的BPE， 该BPE使用字节对而非unicode characters作为子词单元。

总结下两种BPE实现方式：

基于 char-level ：原始 BERT 的方式，它通过对输入文本进行启发式的词干化之后处理得到。

基于 bytes-level：与 char-level 的区别在于bytes-level 使用 bytes 而不是 unicode 字符作为 sub-word 的基本单位，因此可以编码任何输入文本而不会引入 UNKOWN 标记。

当采用 bytes-level 的 BPE 之后，词表大小从3万（原始 BERT 的 char-level ）增加到5万。这分别为 BERT-base和 BERT-large增加了1500万和2000万额外的参数。

之前有研究表明，这样的做法在有些下游任务上会导致轻微的性能下降。但是本文作者相信：这种统一编码的优势会超过性能的轻微下降。且作者在未来工作中将进一步对比不同的encoding方案。


## RoBERTa

总结一下，RoBERTa使用dynamic masking，FULL-SENTENCES without NSP loss，larger mini-batches和larger byte-level BPE（这个文本编码方法GPT-2也用过，BERT之前用的是character粒度的）进行训练。除此之外还包括一些细节，包括：更大的预训练数据、更多的训练步数。

## 参考资料

[文献阅读笔记:RoBERTa：A Robustly Optimized BERT Pretraining Approach](https://blog.csdn.net/ljp1919/article/details/100666563)

