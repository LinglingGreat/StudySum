

**MHA (Multi-Head Attention)** 是 Transformer 的基石，而 **MLA (Multi-Head Latent Attention)** 则是 DeepSeek 团队在 DeepSeek-V2 中提出的一项重要创新，旨在解决大模型推理时的显存瓶颈问题。

---

### 一、 MHA：多头注意力机制 (Multi-Head Attention)

MHA 是经典 Transformer（如 BERT, GPT-3, Llama 1）使用的标准注意力机制。

#### 1. 核心概念

想象你在阅读一段长文本。你不可能只盯着一个词看，你需要同时关注语法结构、指代关系、语义关联等多个方面。

- **Head (头)**：可以理解为一个“独立的视角”。
    
- **Multi-Head (多头)**：模型分裂出多个子空间，并行地捕捉输入序列中不同位置的信息。
    

#### 2. 工作流程

对于输入向量 $x$，MHA 将其分别投影为三个矩阵：**Query ($Q$)**、**Key ($K$)**、**Value ($V$)**。

1. **切分**：将 $Q, K, V$ 切分成 $h$ 个头。例如，模型维度是 4096，有 32 个头，那么每个头的维度是 128。
    
2. 独立计算：每个头独立计算注意力分数：
    
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
    
3. **拼接与融合**：将所有头的输出拼接（Concat）在一起，最后经过一个线性变换（Linear Output）恢复到原始维度。
    

#### 3. MHA 的痛点：KV Cache 显存爆炸

在推理阶段（生成文本时），模型需要缓存之前所有 token 的 $K$ 和 $V$ 矩阵，以免重复计算。这就是 **KV Cache**。

- **问题**：对于 MHA，每个头都有自己独立的 $K$ 和 $V$。随着上下文长度（Context Length）增加，KV Cache 占用的显存呈线性增长，非常庞大。这限制了模型的最大并发数（Batch Size）和最大上下文长度。
    

---

### 二、 过渡方案：MQA 与 GQA

为了解决 MHA 的显存问题，业界提出了过渡方案：

1. **MQA (Multi-Query Attention)**：所有头**共享**同一个 $K$ 和同一个 $V$。显存占用极低，但模型性能（智力）会有所下降。
    
2. **GQA (Grouped-Query Attention)**：Llama 2/3 采用的方案。将头分组，每组共享一个 $K$ 和 $V$。这是性能与显存的折衷。
    

#### 1. MQA (Multi-Query Attention)

MQA 是由 Google 在 2019 年提出的（_Fast Transformer Decoding_）。它的极端之处在于：**让所有的 Query 头共享同一组 Key 和 Value**。

数学表达：

假设模型有 $H$ 个查询头（Query Heads），输入为 $x$：

- **Query**: $\mathbf{q}_i = x W_{Q,i}$ （对于 $i = 1, \dots, H$，每个头都有自己的权重）
    
- **Key**: $\mathbf{k} = x W_K$ （**只有一个头**，不带下标 $i$）
    
- **Value**: $\mathbf{v} = x W_V$ （**只有一个头**，不带下标 $i$）
    

计算分数时：

每一个头 $i$ 都在拿自己的 $\mathbf{q}_i$ 去跟同一个 $\mathbf{k}$ 做内积：

$$\text{Attention}_i = \text{softmax}\left(\frac{\mathbf{q}_i \mathbf{k}^T}{\sqrt{d_k}}\right) \mathbf{v}$$

- **优点**：KV Cache 减小到了原来的 $1/H$。
    
- **缺点**：性能损耗较大，因为所有头只能被迫关注相同的位置和内容。
    

---

#### 2. GQA (Grouped-Query Attention)

GQA 是在 Llama 2、Llama 3 和 Mistral 等模型中被广泛采用的平衡方案。它认为 MQA 太“抠门”了，于是决定**分组共享**。

数学表达：

假设模型有 $H$ 个 Query 头，我们将它们分为 $G$ 个组（Groups）。每一组 Query 共享一对 $K$ 和 $V$。

- **Query**: $\mathbf{q}_i = x W_{Q,i}$ （依然是 $H$ 个独立的头）
    
- **Key**: $\mathbf{k}_g = x W_{K,g}$ （其中 $g$ 是组索引，$1 \le g \le G$）
    
- **Value**: $\mathbf{v}_g = x W_{V,g}$ （其中 $g$ 是组索引，$1 \le g \le G$）
    

计算规则：

如果第 $i$ 个 Query 头属于第 $g$ 组，那么：

$$\text{Attention}_i = \text{softmax}\left(\frac{\mathbf{q}_i \mathbf{k}_g^T}{\sqrt{d_k}}\right) \mathbf{v}_g$$

- **比例关系**：通常 $H$ 是 $G$ 的整数倍（例如 32 个 Q 头，分为 8 组，每组 4 个 Q 头共享 1 个 KV 头）。
    
- **优点**：在显存占用和模型表现（智力）之间取得了完美的平衡。
    


### 三、 MLA：多头潜在注意力 (Multi-Head Latent Attention)

**MLA** 是 DeepSeek-V2 模型的核心创新。它的目标是：**拥有 MHA 的强大性能，同时只占用 MQA 级别的极低显存。**

#### 1. 核心思想：低秩键值联合压缩 (Low-Rank Key-Value Joint Compression)

MLA 认为，MHA 中的 $K$ 和 $V$ 矩阵虽然维度很高，但在实际运作中存在大量冗余信息（即由“低秩”矩阵主导）。

- **传统 MHA**：直接存储巨大的 $K$ 和 $V$。
    
- **MLA 做法**：
    
    1. **压缩（Down-projection）**：将输入的隐藏层向量投影到一个维度很小的**潜在向量（Latent Vector）**，记为 $c_{KV}$。
        
    2. **解压（Up-projection）**：在计算注意力时，利用投影矩阵将这个小的 $c_{KV}$ 恢复成用于计算的 $K$ 和 $V$。
        

**关键点**：在推理时的 KV Cache 中，我们**只需要存储这个压缩后的潜在向量 $c_{KV}$**，而不需要存储展开后的巨大的 $K$ 和 $V$。

#### 2. 巧妙处理位置编码 (Decoupled RoPE)

在大模型中，旋转位置编码（RoPE）对位置感知至关重要。但是 RoPE 很难直接作用于压缩后的向量。

MLA 采用了一种**解耦（Decoupled）** 策略：

- **内容部分**：使用上述的压缩向量 $c_{KV}$。
    
- **位置部分**：单独使用一个携带 RoPE 信息的向量 $k_{R}$。
    
- **计算时**：将这两部分拼接起来参与注意力计算。
    

#### 3. MLA 的优势

- **显存占用极低**：因为 Cache 中只存压缩向量，其大小主要取决于压缩维度，通常只有 MHA 的 1/8 甚至更少（接近 MQA）。
    
- **性能无损**：不同于 MQA 强行让所有头共享参数，MLA 实际上保留了生成多个不同 $K$ 和 $V$ 的能力（通过解压矩阵），因此它能维持类似 MHA 的高表达能力。
    

---

### 核心结论

- **MHA** 是“富二代”，性能最强但极度消耗资源，不适合超长上下文推理。
    
- **MLA** 是“技术流”，通过数学上的**低秩压缩**技巧，实现了“既要马儿跑（MHA的高性能），又要马儿不吃草（MQA的低显存）”。这就是为什么 DeepSeek-V2 能够在保持高性能的同时，推理成本极其低廉的原因。
    

---

要真正理解 MLA (Multi-Head Latent Attention)，我们需要将其拆解为两个核心数学技巧：**矩阵吸收（Matrix Absorption）** 和 **解耦 RoPE（Decoupled RoPE）**。

这是 DeepSeek-V2 论文（_DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model_）中最精彩的部分。

---

### 第一部分：低秩键值联合压缩 (Low-Rank Key-Value Joint Compression)

这一部分解决了显存占用的问题。

#### 1. 传统 MHA 的计算（回顾）

在标准 Multi-Head Attention 中，对于输入隐藏层向量 $\mathbf{h}$，我们需要通过三个大的投影矩阵 $W_Q, W_K, W_V$ 生成 $q, k, v$。

假设 $d$ 是模型维度，$n_h$ 是头数，$d_h$ 是每个头的维度。

$$\mathbf{k} = \mathbf{h} W_K, \quad \mathbf{v} = \mathbf{h} W_V$$

- **痛点**：在推理时，我们需要把生成的 $\mathbf{k}$ 和 $\mathbf{v}$ 存入 KV Cache。这些向量非常大。
    

#### 2. MLA 的压缩策略

MLA 引入了一个**潜在向量 (Latent Vector)** $\mathbf{c}_{KV}$ 来替代直接生成 $k$ 和 $v$。

Step 1: 降维（Down-projection）

我们将输入 $\mathbf{h}$ 投影到一个低维空间，生成压缩向量 $\mathbf{c}_{KV}$：

$$\mathbf{c}_{KV} = \mathbf{h} W_{DKV}$$

- 注意：$\mathbf{c}_{KV}$ 的维度远小于 $\mathbf{h}$，这就是省显存的根源。在 KV Cache 中，我们**只存储这个 $\mathbf{c}_{KV}$**。
    

Step 2: 升维（Up-projection）

在计算注意力时，理论上我们需要恢复出 $k$ 和 $v$：

$$\mathbf{k} = \mathbf{c}_{KV} W_{UK}$$

$$\mathbf{v} = \mathbf{c}_{KV} W_{UV}$$

其中 $W_{UK}$ 和 $W_{UV}$ 是升维矩阵。

#### 3. 核心魔法：矩阵吸收 (Matrix Absorption)

如果仅仅是“先压再解”，计算量并没有减少，反而增加了矩阵乘法。DeepSeek 利用矩阵乘法的**结合律**，消除了推理时的计算开销。

注意力分数的计算公式核心是 $q^T k$。我们将 MLA 的 $k$ 代入：

$$\text{Score} = \mathbf{q}^T \mathbf{k} = \mathbf{q}^T (\mathbf{c}_{KV} W_{UK})$$

利用结合律，我们可以先计算括号左边：

$$\text{Score} = (\mathbf{q}^T W_{UK}) \mathbf{c}_{KV}$$

**这意味着什么？**

- **对于 Key ($k$)**：我们不需要在推理时显式地把 $\mathbf{c}_{KV}$ 恢复成巨大的 $\mathbf{k}$。我们可以把升维矩阵 $W_{UK}$ **吸收**进 Query ($q$) 的投影矩阵中。
    
- **对于 Value ($v$)**：同样的逻辑，注意力权重 $A$ 与 $v$ 相乘：
    
    $$ \text{Output} = A \cdot \mathbf{v} = A \cdot (\mathbf{c}_{KV} W_{UV}) = (A \cdot \mathbf{c}_{KV}) W_{UV}$$
    
    我们可以先对小的 $\mathbf{c}_{KV}$ 加权求和，最后再统一乘一次 $W_{UV}$ 恢复维度。
    

**结论**：通过数学变换，MLA 在显存中只存了极小的 $\mathbf{c}_{KV}$，却在数学等价性上模拟了多头 $K$ 和 $V$ 的效果。

---

### 第二部分：解耦 RoPE (Decoupled RoPE)

这一部分解决了“压缩后的位置信息丢失”问题。

#### 0. MHA中的RoPE是什么样的？

简单一句话概括：**RoPE 将位置信息转化为旋转角度，分别“扭转”了 Query 和 Key 向量，使得它们在做内积（计算注意力）时，自然而然地携带了相对位置信息。**

以下是 RoPE 融入 QKV 的详细拆解：

1. 准备阶段：先生成标准的 Q、K、V

在 RoPE 介入之前，Transformer 的前几步和以前一模一样。

假设输入向量是 $x_m$（第 $m$ 个 token），我们先通过线性层生成初始的 $q, k, v$：

$$\mathbf{q}_m = x_m W_Q, \quad \mathbf{k}_m = x_m W_K, \quad \mathbf{v}_m = x_m W_V$$

- **注意**：此时的 $\mathbf{q}_m, \mathbf{k}_m, \mathbf{v}_m$ **没有任何位置信息**（因为 Transformer 是并行计算的，没有时序概念）。
    


2. 注入阶段：只“旋转” Q 和 K

RoPE 的核心操作发生在计算 Attention Score 之前。它**只对 $q$ 和 $k$ 动手**，而完全**放过 $v$**。

(1) 两两分组 (Pairing)

RoPE 不会把 $q$ 当作一个整体去乘一个大矩阵，而是把向量的每两个维度分成一组。

假设 $q$ 的维度是 $d$，它会被切分为 $d/2$ 个子空间（2D平面）：

$$\mathbf{q} = [q_0, q_1, q_2, q_3, ..., q_{d-2}, q_{d-1}]$$

分组为：$(q_0, q_1), (q_2, q_3), ...$

(2) 旋转操作 (Rotation)

对于位置 $m$ 的向量，RoPE 会在每个 2D 子平面上逆时针旋转一个角度 $m\theta_i$。

具体数学变换如下（以第一组为例）：

$$\begin{pmatrix} q'_0 \\ q'_1 \end{pmatrix} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix}$$

- **$m$ (Position)**：当前 token 的绝对位置（比如第 5 个词）。
    
- **$\theta$ (Base Angle)**：每个子平面的旋转基数不同（越靠前的维度旋转越快，越靠后的维度旋转越慢，类似时钟的时针分针）。
    

(3) 为什么不旋转 V？

- **V (Value)** 承载的是**语义内容**，它最终会被加权求和输出。如果给 $V$ 加上了随位置变化的旋转，那么同一个词在不同位置的语义表示就会变来变去，这会破坏模型的语义稳定性。
    
- **Q 和 K** 的任务是**计算匹配度**。在匹配度计算中注入位置关系是最合适的。
    

3. 结果：相对位置的自动涌现

这是 RoPE 最神奇的地方。当我们计算 Attention Score（$Q$ 和 $K$ 的内积）时，奇迹发生了。

假设 Query 在位置 $m$，Key 在位置 $n$。

它们的内积 $Q_m \cdot K_n$ 展开后，经过三角函数变换（积化和差），最终会变成：

$$\langle \mathbf{q}'_m, \mathbf{k}'_n \rangle = \text{Magnitude} \times \cos((m - n)\theta)$$

- **注意看**：公式里只剩下了 **$(m - n)$**。
    
- **含义**：注意力的强弱，**不再取决于 $m$ 和 $n$ 的绝对数值，而只取决于它们的距离（相对位置）**。
    
- 这就是为什么 RoPE 被称为“用绝对位置编码实现了相对位置编码的效果”。
    

总结：RoPE 的 QKV 融合流

1. **生成**：$x \to Q, K, V$（纯语义，无位置）。
    
2. **融合**：
    
    - $Q \leftarrow \text{Rotate}(Q, \text{pos})$
        
    - $K \leftarrow \text{Rotate}(K, \text{pos})$
        
    - $V \leftarrow V$ （**V 保持不变！**）
        
3. **计算**：$\text{Score} = Q_{rot} \cdot K_{rot}^T$ （此时相对位置信息自动生效）。
    
4. **输出**：$\text{Output} = \text{Softmax}(\text{Score}) \cdot V$。
    

---

视频推荐

为了让你更直观地看到 RoPE 是如何在向量空间中进行“旋转”操作的，这个视频提供了非常棒的可视化解释。

[Rotary Positional Embeddings Explained](https://www.youtube.com/watch?v=V8r__fXx7tU)

**推荐理由：** 该视频非常清晰地展示了 RoPE 如何通过旋转 Q 和 K 向量来编码位置信息，特别是其中的动态演示能让你直观理解“两两分组旋转”的概念，这比纯看公式要容易理解得多。

---

#### 1. 为什么压缩会破坏 RoPE？

旋转位置编码 (RoPE) 对向量的**维度顺序**和**配对**极其敏感。如果我们对向量进行了低秩压缩（$W_{DKV}$），向量内部的几何结构会被打乱。如果在 $\mathbf{c}_{KV}$ 上直接加 RoPE，再经过 $W_{UK}$ 投影回去，由于矩阵乘法不具备旋转不变性，位置信息会变得一团糟。

#### 2. MLA 的解决方案：分而治之

MLA 决定不把位置信息塞进那个被压缩的 $\mathbf{c}_{KV}$ 里，而是**单独开辟一条通路**来处理位置。

它将 Query ($q$) 和 Key ($k$) 拆分为两部分：

1. **Content Part (内容部分)**：携带语义信息，使用上述的“低秩压缩”策略。
    
2. **RoPE Part (位置部分)**：携带位置信息，**不进行压缩**，直接携带 RoPE。
    

#### 3. 数学表达

在生成最终用于计算 Attention Score 的向量时，我们将这两部分**拼接 (Concat)** 起来。

**对于 Query ($q$)**:

$$\mathbf{q} = [\mathbf{q}_{content}, \mathbf{q}_{rope}]$$

- $\mathbf{q}_{content}$：经过矩阵吸收后的内容查询向量。
    
- $\mathbf{q}_{rope}$：单独生成的、应用了 RoPE 的查询向量。
    

**对于 Key ($k$)**:

$$\mathbf{k} = [\mathbf{k}_{content}, \mathbf{k}_{rope}]$$

- $\mathbf{k}_{content}$：即上文的 $\mathbf{c}_{KV}$ (直接作为 Key 的内容部分)。
    
- $\mathbf{k}_{rope}$：单独生成的、应用了 RoPE 的键向量 (需要在 Cache 中额外存储，但它很小)。
    

#### 4. 最终的注意力计算

注意力分数变为两项之和：

$$\text{Score} = \mathbf{q}_{content}^T \mathbf{k}_{content} + \mathbf{q}_{rope}^T \mathbf{k}_{rope}$$

- **第一项**：负责匹配语义（如：“苹果”匹配“好吃”），利用了低秩压缩技术。
    
- **第二项**：负责匹配位置（如：关注第 5 个词和第 6 个词的相对距离），利用了标准的 RoPE。
    

---

### 第三部分：全景总结

我们可以这样总结 MLA 的完整推理流程：

1. **输入**：来了一个 Token。
    
2. **压缩**：
    
    - 将 Token 的语义信息压缩成极小的 $\mathbf{c}_{KV}$。
        
    - **存入 Cache**：存 $\mathbf{c}_{KV}$。
        
3. **位置处理**：
    
    - 生成一个很小的位置向量 $\mathbf{k}_{rope}$ 并应用旋转。
        
    - **存入 Cache**：存 $\mathbf{k}_{rope}$。
        
    - _(注：Cache 总大小 = 小 $\mathbf{c}_{KV}$ + 小 $\mathbf{k}_{rope}$，远小于传统 MHA 的大 K + 大 V)_
        
4. **计算注意力**：
    
    - **Query 来访**：Query 自身也做了相应的变形（吸收了升维矩阵）。
        
    - **混合计算**：Query 的内容部分去找 Cache 里的 $\mathbf{c}_{KV}$（语义匹配），Query 的位置部分去找 Cache 里的 $\mathbf{k}_{rope}$（位置匹配）。
        
    - **得出结果**：两者分数相加，经过 Softmax，完成注意力机制。
        

### 为什么这很牛？

MLA 证明了：大模型中巨大的 KV Cache 其实充斥着冗余。我们不需要存储完整的矩阵来保持“多头”的独立性。通过**线性代数的低秩分解**加上**巧妙的 RoPE 旁路设计**，我们将显存占用压缩了 90% 以上，却几乎没有损失任何模型智力。


### 一、 为什么 Query (Q) 也要降维？

在传统的 Transformer 中，Q 通常是不缓存的（因为推理时只需要当前 Token 的 Q）。但 MLA 对 Q 也进行了低秩压缩，主要原因有两点：

1. 减少计算量（FLOPs）：
    
    DeepSeek-V2 是一个巨大的模型。如果直接从隐藏层 $d$（如 7168 维）生成多头 Q（如 128 头 $\times$ 128 维 = 16384 维），投影矩阵 $W_Q$ 会非常庞大。通过先降维到较小的 $d_c$，再升维到多头，可以显著减少参数量和推理时的计算开销。
    
2. 训练稳定性与对称性：
    
    KV 已经通过压缩降低了秩，如果 Q 保持全秩，可能会导致模型在特征匹配时出现某种不平衡。对 Q 进行压缩可以看作是一种正则化，强制模型在低维潜在空间中提取最核心的语义特征。
    

---

### 二、 MLA 的详细公式拆解

我们将 MLA 的计算分为 **“内容部分”** 和 **“位置部分”**。

#### 1. 降维阶段 (Compression)

对于当前位置的输入 $\mathbf{h}_t$，首先通过降维矩阵生成两个潜在向量：

- **KV 潜在向量**：$\mathbf{c}_t^{KV} = \text{LayerNorm}(\mathbf{h}_t W_{DKV})$ （存储在 Cache 中）
    
- **Q 潜在向量**：$\mathbf{c}_t^Q = \text{LayerNorm}(\mathbf{h}_t W_{DQ})$ （仅瞬时计算，不存 Cache）
    

#### 2. 生成内容部分 (Content - No RoPE)

利用升维矩阵恢复出用于注意力计算的向量（注意：这些向量**不带**位置信息，下标为 $C$）：

- **Query 内容**：$\mathbf{q}_{t,i}^C = \mathbf{c}_t^Q W_{UQ, i}$ （第 $i$ 个头）
    
- **Key 内容**：$\mathbf{k}_{t,i}^C = \mathbf{c}_t^{KV} W_{UK, i}$
    
- **Value 内容**：$\mathbf{v}_{t,i}^C = \mathbf{c}_t^{KV} W_{UV, i}$
    

#### 3. 生成位置部分 (Position - RoPE)

为了应用 RoPE 且不破坏上述的压缩结构，MLA 单独开辟了两个专门携带位置信息的向量：

- **Query 位置**：$\mathbf{q}_{t,i}^R = \text{RoPE}(\mathbf{h}_t W_{QR})$
    
- **Key 位置**：$\mathbf{k}_t^R = \text{RoPE}(\mathbf{h}_t W_{KR})$
    
    - _注：在 DeepSeek-V2 中，为了进一步省显存，所有头**共享**同一个位置 Key $\mathbf{k}_t^R$。_
        

#### 4. 最终拼接与注意力计算

最终进入 Softmax 计算的 $Q$ 和 $K$ 是由内容和位置拼接而成的：

$$\mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^C, \mathbf{q}_{t,i}^R]$$

$$\mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C, \mathbf{k}_t^R]$$

---

### 三、 总结：推理时 Cache 到底存了什么？

这是 MLA 最引以为傲的地方。传统 MHA 存的是巨大的 $K$ 和 $V$，而 MLA 的 KV Cache 只存两样东西：

|**存储项**|**符号**|**作用**|**维度大小**|
|---|---|---|---|
|**压缩语义向量**|$\mathbf{c}_t^{KV}$|包含该 token 的所有语义信息|$d_c$ (通常为 512)|
|**位置向量**|$\mathbf{k}_t^R$|包含该 token 的位置信息|$d_R$ (通常为 64)|

总显存消耗：只有 $d_c + d_R$。

相比之下，如果用 MHA，显存消耗是 $n_h \times d_h \times 2$（头数 $\times$ 每个头的维度 $\times$ K和V两个）。

结果：MLA 的 KV Cache 大小通常只有 MHA 的 1/10 甚至更低，这让 DeepSeek 能够以极低的成本处理超长文本。

---

### 四、 矩阵吸收 (Matrix Absorption) 的本质

为什么 Q 的压缩不影响速度？因为在计算 $\mathbf{q}^T \mathbf{k}$ 时：

$$(\mathbf{c}_t^Q W_{UQ})^T (\mathbf{c}_j^{KV} W_{UK}) = \mathbf{c}_t^Q (W_{UQ}^T W_{UK}) \mathbf{c}_j^{KV}$$

在推理前，我们可以直接把 $W_{UQ}^T W_{UK}$ 预先乘好，合并成一个中间矩阵。这样在推理时，我们直接用低维的 $\mathbf{c}_t^Q$ 去乘低维的 $\mathbf{c}_j^{KV}$ 即可，效率极高。

---

### 1. 为什么要用“拼接”？（数学解耦的需求）

在 MLA 中，内容向量和位置向量有着完全不同的命运：

- **内容部分（Content）**：为了省显存，经历了**低秩压缩**。计算时需要通过“矩阵吸收”来还原多头的效果。
    
- **位置部分（RoPE）**：为了保持位置敏感性，**不能被压缩**。它必须保持原始的旋转结构。
    

如果采用“相加”：

如果我们把 RoPE 后的向量直接加到压缩后的内容向量上，由于矩阵乘法的分配律：

$$(Q_{content} + Q_{rope})^T (K_{content} + K_{rope})$$

展开后会出现交叉项（例如 $Q_{content}^T K_{rope}$）。这些交叉项没有任何物理意义，会极大地干扰注意力机制的准确性，甚至导致模型无法训练。

采用“拼接”的好处：

根据向量内积的定义，拼接后的向量内积等于分量内积之和：

$$[q_C; q_R]^T [k_C; k_R] = (q_C^T k_C) + (q_R^T k_R)$$

- **左半部分 ($q_C^T k_C$)**：负责纯粹的语义匹配（比如“北京”和“首都”）。
    
- **右半部分 ($q_R^T k_R$)**：负责纯粹的位置关系（比如“前一个词”和“后一个词”）。
    

**两者互不干扰，完美解耦。**

---

### 2. 维度是多少？（以 DeepSeek-V2 为例）

我们可以通过具体的数值来看看拼接后的向量到底长什么样。

#### 对于每一个“头”（Head）来说：

在 DeepSeek-V2 的设置中：

- **内容维度 ($d_{C}$)**：每一个头的语义向量维度是 **128**。
    
- **位置维度 ($d_{R}$)**：每一个头的位置向量维度是 **64**。
    

所以，对于每一个头 $i$，最终参与计算的 $Q$ 和 $K$ 的维度是：

$$\text{Total Dimension} = 128 (\text{Content}) + 64 (\text{RoPE}) = \mathbf{192}$$

---

### 3. 一个极佳的省钱技巧：共享位置 Key

你可能会问：如果每个头都加 64 维位置向量，KV Cache 还是会变大吗？

DeepSeek 在这里又耍了一个“聪明”：

- **Query (Q)**：每个头都有自己独立的位置向量 $q_{i}^R$（为了捕捉不同视角的位置敏感度）。
    
- **Key (K)**：**所有头共享同一个位置向量 $k^R$**。
    

这意味着：

在推理时，KV Cache 只需要存：

1. **压缩后的潜在向量 $c_{KV}$** (512 维)
    
2. **唯一的一个位置 Key $k^R$** (64 维)
    

**总 KV Cache 维度 = 512 + 64 = 576。**

相比之下，如果使用传统的 MHA（假设 128 个头，每个头 128 维）：

总 KV Cache 维度 = 128 (头数) × 128 (维度) × 2 (K和V) = 32,768。

> **结论**：MLA 通过这种“拼接 + 共享位置 Key”的设计，将 KV Cache 的维度从 **32,768** 直接降到了 **576**，压缩率高达 **50 倍以上**！

---


### 四、 总结对比表

|**特性**|**MHA (标准)**|**GQA (Llama 3等)**|**MQA (极致省显存)**|**MLA (DeepSeek-V2)**|
|---|---|---|---|---|
|**K/V 头数量**|等于 Q 头数量 ($H$)|小于 Q 头数量 (分组)|只有 1 个|**虚拟**等于 Q 头数量|
|**KV Cache 大小**|巨大 (瓶颈)|中等|极小|**极小** (通过压缩)|
|**模型性能**|最好|好|较弱|**最好 (媲美 MHA)**|
|**实现复杂度**|低|低|低|**高** (需改动模型结构)|


三者维度与 KV Cache 总结对比

为了方便记忆，我们假设模型有 **32 个头**，每个头维度 **128**：

|**机制**|**Query 维度**|**Key / Value 维度**|**推理时 KV Cache 大小 (相对于 MHA)**|
|---|---|---|---|
|**MHA**|$32 \times 128$|$32 \times 128$|100% (全量存储)|
|**GQA**|$32 \times 128$|$8 \times 128$ (假设8组)|25%|
|**MQA**|$32 \times 128$|$1 \times 128$|3.12%|
|**MLA**|$128 \times 192$|**压缩后的 512 + 64**|**约 2% - 5% (但性能媲美 MHA)**|
