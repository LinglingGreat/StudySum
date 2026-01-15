「Speculative decoding」使用两个模型：一个是原始目标模型称为大模型，另一个是比原始模型小得多的近似模型称为小模型。主要思想是先让小模型提前解码多个 token 进行猜测，并将它们作为单个 batch 输入到一个大模型中进行审核修正，其效果和直接用大模型解码等价。如果小模型猜测的不准确，那么大型模型会放弃小模型预测的 token，继续使用大型模型进行解码。

SGLang支持： [Speculative Decoding — SGLang](https://docs.sglang.io/advanced_features/speculative_decoding.html)

VLLM支持：[Speculative Decoding - vLLM - vLLM 文档](https://docs.vllm.com.cn/en/latest/features/spec_decode/#speculating-using-eagle-based-draft-models)

训练推测解码的代码仓库：[GitHub - sgl-project/SpecForge: Train speculative decoding models effortlessly and port them smoothly to SGLang serving.](https://github.com/sgl-project/SpecForge)

- 2026.1：EAGLE3草稿模型目前只支持LLama模型，写死的

# SpecForge

### 调用链

```
train_eagle3.py
    ├── build_target_model()  ← 加载原模型
    │   └── get_eagle3_target_model(backend="custom")  ← 这里可选项包括sglang, hf, custom
    │       └── CustomEagle3TargetModel.from_pretrained()
    │           └── AutoDistributedTargetModel.from_pretrained()  ← 这里使用 _model_mapping
    │               └── 根据 config 类型选择模型类
    │                   ├── LlamaConfig → LlamaForCausalLM
    │                   ├── Qwen2Config → Qwen2ForCausalLM
    │                   └── ...
    │
    └── build_dataloaders()  ← 构建数据集，得到训练数据、验证集数据、词汇映射的路径
        └── draft_model.load_vocab_mapping(vocab_mapping_path)
    └── OnlineEagle3Model(draft_model)  ← 构建草稿模型的训练，后面就是训练相关的代码了
```

## 构建草稿模型的链路

```
build_draft_model(args)
    │
    ├── 1. 获取 draft model config
    │   ├── 如果没有指定 --draft-model-config:
    │   │   └── create_draft_config_from_target() 自动生成
    │   │       └── generate_draft_model_config()
    │   │           ├── 读取 target model 的 config
    │   │           ├── 使用模板 llama3-8B-eagle3.json
    │   │           └── 复制target model的参数到 draft config
    │   │
    │   └── 如果指定了 --draft-model-config:
    │       └── AutoDraftModelConfig.from_file(config_path)
    │
    ├── 2. 根据 config 中的 "architectures" 创建模型
    │   └── AutoEagle3DraftModel.from_config(config)
    │       └── 查找 _model_mapping[type(config)]
    │           └── LlamaConfig → LlamaForCausalLMEagle3  ← 目前只有这一个！
    │
    └── 3. 加载 embedding
        └── draft_model.load_embedding(target_model_path)  ← 默认的embedding_key是model.embed_tokens.weight
        └── draft_model.freeze_embedding()  ← 冻结潜入层参数，训练中不更新
```

1. **词汇表示一致性**：draft model 和 target model 使用完全相同的词嵌入，保证它们对同一个 token 有相同的向量表示
2. **节省训练资源**：嵌入层通常参数量很大，冻结它可以减少需要训练的参数数量和显存占用
3. **保持语义空间**：在推测解码（speculative decoding）中，draft model 需要与 target model 在同一个语义空间中工作，共享嵌入层是实现这一点的关键

如果自动生成草稿模型的设置的话，会有以下信息，注意这里默认把vocab_size设置成32000：

```python
# Special handling for some parameters
# Ensure num_hidden_layers is always 1 (EAGLE3 feature)
draft_config["num_hidden_layers"] = 1

# Keep some fixed draft model specific parameters
draft_config["tie_word_embeddings"] = False
draft_config["use_cache"] = True

# If template doesn't have draft_vocab_size, set default
if "draft_vocab_size" not in draft_config:
	draft_config["draft_vocab_size"] = 32000  # Default value
```

举个例子，draft_model的模型长这样：

```python
LlamaForCausalLMEagle3(
  (embed_tokens): Embedding(131072, 5120, padding_idx=0)
  (midlayer): LlamaDecoderLayer(
    (self_attn): LlamaUSPAttention(
      (q_proj): Linear(in_features=10240, out_features=4096, bias=False)
      (k_proj): Linear(in_features=10240, out_features=1024, bias=False)
      (v_proj): Linear(in_features=10240, out_features=1024, bias=False)
      (o_proj): Linear(in_features=4096, out_features=5120, bias=False)
      (rotary_emb): LlamaRotaryEmbedding()
    )
    (mlp): LlamaMLP(
      (gate_proj): Linear(in_features=5120, out_features=14336, bias=False)
      (up_proj): Linear(in_features=5120, out_features=14336, bias=False)
      (down_proj): Linear(in_features=14336, out_features=5120, bias=False)
      (act_fn): SiLUActivation()
    )
    (hidden_norm): LlamaRMSNorm()
    (input_layernorm): LlamaRMSNorm()
    (post_attention_layernorm): LlamaRMSNorm()
  )
  (fc): Linear(in_features=15360, out_features=5120, bias=False)
  (norm): LlamaRMSNorm()
  (lm_head): Linear(in_features=5120, out_features=32000, bias=False)
)
```


## 构建数据集

构建数据集，得到

```
results = {"input_ids": [], "loss_mask": [], "attention_mask": []}
```

其中loss_mask中的assistant部分会设置成1，其余部分是0.

会把处理好的到的数据集cache下来，存储成pkl文件（参数控制，需要传cache_dir和cache_key）。cache_dir有默认值，cache_key是根据参数生成的。

```
cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"  # Tokenizer may also different
    )
cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
```

这里也可以提前预处理好，存储起来，然后训练的时候直接加载就行了。

这里在替换的时候使用的方式是我之前没见过的，又学到了：

### `offset_mapping` 的含义

`offset_mapping` 是 tokenizer 返回的一个**字符级别的位置映射**，记录了每个 token 在原始文本中的起止位置。

结构

```python
offsets = [(0, 5), (5, 6), (6, 11), ...]
#          token0  token1  token2
```

每个元素是一个 `(start_char, end_char)` 元组，表示：
- `start_char`：该 token 在原始字符串中的**起始字符索引**
- `end_char`：该 token 在原始字符串中的**结束字符索引**（不包含）

示例

```python
text = "Hello world"
# 假设 tokenizer 把它分成 ["Hello", " ", "world"]

offset_mapping = [
    (0, 5),   # "Hello" 对应原文的第 0-5 个字符
    (5, 6),   # " " 对应原文的第 5-6 个字符  
    (6, 11),  # "world" 对应原文的第 6-11 个字符
]
```

在代码中的用途

在 parse.py 中，`offset_mapping` 被用来**精确定位哪些 token 属于 assistant 的回复内容**：

```python
# 用正则表达式找到 assistant 回复在原文中的字符范围
assistant_start_char = match.start(1)
assistant_end_char = match.end(1)

# 然后遍历每个 token 的字符范围
for idx, (token_start, token_end) in enumerate(offsets):
    # 如果这个 token 完全落在 assistant 回复的字符范围内
    if assistant_start_char <= token_start <= token_end <= assistant_end_char:
        loss_mask[idx] = 1  # 标记这个 token 需要计算 loss
```

为什么需要它？

因为 token 和字符不是一一对应的：
- 一个单词可能被拆成多个 subword token
- 一个 token 可能包含多个字符

`offset_mapping` 提供了 **token 索引 ↔ 字符索引** 的桥梁，让你可以：
1. 先用正则表达式在原始文本中找到目标区域（字符级别）
2. 再通过 offset_mapping 找到对应的 token（token 级别）

## 构建词汇映射

这个也是会cache下来的，cache_key跟数据集的一样，注意如果你改了草稿模型的配置，那么cache_key是不会变的（因为没用到草稿模型这个信息去生成）

`generate_vocab_mapping_file` 函数解释：这个函数用于生成**词汇表映射文件**，建立 target model 词汇表和 draft model 词汇表之间的映射关系。

### 背景

- **Target model**：原始大模型，词汇表较大（如 128K）
- **Draft model**：小型草稿模型，词汇表较小（如 32K）

由于词汇表大小不同，需要建立两者之间的映射。

### 核心流程

```
1. 统计数据集中每个 token 的出现频率
         ↓
2. 选出频率最高的 top N 个 token（N = draft_vocab_size）
         ↓
3. 生成双向映射：d2t 和 t2d
         ↓
4. 保存到缓存文件
```

### 关键变量解释

#### `d2t` (Draft to Target)
- **含义**：从 draft token id 映射到 target token id
- **结构**：`d2t[draft_id] = offset`，实际 target_id = draft_id + offset
- 例如：draft 词汇表选择了 target 中的 `[0, 2, 5, 8]` 四个 token
  ```python
  d2t = [0-0, 2-1, 5-2, 8-3] = [0, 1, 3, 5]
  # draft_id=1 对应 target_id = 1 + d2t[1] = 1 + 1 = 2
  ```

#### `t2d` (Target to Draft)  
- **含义**：标记 target 词汇表中的 token 是否被 draft 使用
- **结构**：`t2d[target_id] = True/False`
- 例如：
  ```python
  t2d = [True, False, True, False, False, True, False, False, True]
  #       0      1     2      3      4     5      6      7     8
  # 表示 target token 0, 2, 5, 8 被 draft 使用
  ```

### 为什么要统计频率？

选择**高频词**进入 draft 词汇表，可以最大化覆盖训练数据中实际出现的 token，提高 draft model 的预测准确率。

函数会打印覆盖率：
```python
print(f"top {draft_vocab_size} token frequency ratio: {top_N_ratio:.2%}")
# 例如：top 32000 token frequency ratio: 99.85%
```

### 缓存机制

```python
vocab_mapping_path = os.path.join(cache_dir, f"{cache_key}.pt")
if os.path.exists(vocab_mapping_path):
    return vocab_mapping_path  # 直接返回，避免重复计算
```

生成的映射会被保存，下次运行时直接加载，节省时间。

### 使用场景

在 train_eagle3.py 中：
```python
# 生成词汇映射
vocab_mapping_path = generate_vocab_mapping_file(...)

# 加载到 draft model
draft_model.load_vocab_mapping(vocab_mapping_path)
```

这样 draft model 在推理时就能正确地在两个词汇表之间转换。

### 相关疑问

[[Question] Why configs/qwen3-30B-A3B-eagle3.json draft\_vocab\_size=32000 · Issue #341 · sgl-project/SpecForge](https://github.com/sgl-project/SpecForge/issues/341)

[Vocab size Issue between target model and draft model (Maybe Tokenizer Issue?) · Issue #258 · SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE/issues/258)


## Eagle3模型训练

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练流程                                   │
├─────────────────────────────────────────────────────────────────┤
│  输入数据 (input_ids, attention_mask, loss_mask)                 │
│           ↓                                                      │
│  Target Model → hidden_states + target logits                   │
│           ↓                                                      │
│  Draft Model → 预测下一个 token 的分布                            │
│           ↓                                                      │
│  Loss = KL散度(draft_logits, target_logits)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

### 1. 数据准备阶段

在 `run_forward` 中，首先通过 target model 生成训练数据：

```python
# target model 前向传播，获取：
# - hidden_states: 中间层的隐藏状态 (用作 draft model 的输入)
# - target: target model 的 logits (作为训练目标)
eagle3_data = target_model.generate_eagle3_data(
    input_ids=data["input_ids"].cuda(),
    attention_mask=data["attention_mask"].cuda(),
    loss_mask=data["loss_mask"].cuda(),
)
```

`generate_eagle3_data` 详解：这个函数通过 Target Model 前向传播，生成训练 Draft Model 所需的数据。

输出结构

```python
@dataclass
class Eagle3TargetOutput:
    hidden_states: torch.Tensor      # 辅助层隐藏状态
    target: torch.Tensor             # Target model 的 logits（训练目标）
    loss_mask: torch.Tensor          # 损失掩码
    input_ids: torch.Tensor          # 输入 token ids
    attention_mask: torch.Tensor     # 注意力掩码
    last_hidden_states: Optional[torch.Tensor] = None  # 最后一层隐藏状态（可选）
```

---

### 各字段详解

#### 1. `hidden_states` - 辅助层隐藏状态
**形状**: `(batch, seq_len, 3 × hidden_size)`

```python
# 从 Target Model 的 3 个特定层提取隐藏状态
aux_hidden_states_layers = [
    1,                    # 浅层（第 1 层）
    num_layers // 2 - 1,  # 中层（如 32 层模型取第 15 层）
    num_layers - 4,       # 深层（如 32 层模型取第 28 层）
]

# 拼接成一个张量
hidden_states = torch.cat(
    (hidden_states0, hidden_states1, hidden_states2), dim=-1
)
# 例如: hidden_size=4096 → 拼接后为 4096×3 = 12288
```

**用途**：作为 Draft Model 的**额外输入**，提供 Target Model 的"知识提示"

```
┌─────────────────────────────────────────────────────┐
│              Target Model (32 层)                    │
├─────────────────────────────────────────────────────┤
│  Layer 0  ──────────────────────────────────────────│
│  Layer 1  ─────────────→ hidden_states0 ────────┐   │
│  ...                                            │   │
│  Layer 15 ─────────────→ hidden_states1 ────────┼─→ concat → hidden_states
│  ...                                            │   │
│  Layer 28 ─────────────→ hidden_states2 ────────┘   │
│  ...                                                │
│  Layer 31 ──────────────────────────────────────────│
│       ↓                                             │
│    logits ─────────────→ target                     │
└─────────────────────────────────────────────────────┘
```

---

#### 2. `target` - Target Model 的 logits
**形状**: `(batch, seq_len, vocab_size)`

```python
target = outputs.logits  # Target model 最后一层输出的 logits
target = padding(target, left=False)  # 右移一位（预测下一个 token）
```

**用途**：作为 Draft Model 的**训练目标**（soft label）

```
原始:     [A] [B] [C] [D] [E]
padding后: [B] [C] [D] [E] [PAD]  ← Draft 要预测的目标
```

---

#### 3. `loss_mask` - 损失掩码
**形状**: `(batch, seq_len, 1)`

```python
loss_mask = loss_mask[..., None]  # 扩展维度用于广播
```

**用途**：标记哪些位置需要计算 loss（只在 assistant 回复部分计算）

```
input:     [SYS] [USER] [QUERY] [ASST] [ANSWER1] [ANSWER2] [EOS]
loss_mask:   0     0       0      0        1         1       1
```

---

#### 4. `input_ids` - 输入 token IDs
**形状**: `(batch, seq_len)`

```python
input_ids = padding(input_ids, left=False)  # 右移一位
```

**用途**：Draft Model 的输入（与 target 对齐后的版本）

---

#### 5. `attention_mask` - 注意力掩码
**形状**: `(batch, seq_len)`

**用途**：标记有效 token 位置，用于 attention 计算

---

### 完整数据流示例

假设输入序列为 `"Hello world"` → tokens `[15496, 995]`

```
输入:
  input_ids = [15496, 995]
  
Target Model 前向传播后:
  hidden_states0 (layer 1):  [[h0_0, h0_1, ..., h0_4095], [h0_0, h0_1, ..., h0_4095]]
  hidden_states1 (layer 15): [[h1_0, h1_1, ..., h1_4095], [h1_0, h1_1, ..., h1_4095]]  
  hidden_states2 (layer 28): [[h2_0, h2_1, ..., h2_4095], [h2_0, h2_1, ..., h2_4095]]
  
  logits (layer 31): [[vocab_size 个概率], [vocab_size 个概率]]

输出:
  hidden_states: (1, 2, 12288)  ← 3 层拼接
  target:        (1, 2, vocab_size) ← 右移后的 logits
  input_ids:     (1, 2)  ← 右移后的 input_ids
  loss_mask:     (1, 2, 1)
```

---

### 为什么选择这 3 个层？

| 层 | 位置 | 捕获的信息 |
|----|------|-----------|
| Layer 1 | 浅层 | 词汇/语法级别的特征 |
| Layer N/2 | 中层 | 语义理解特征 |
| Layer N-4 | 深层 | 接近最终预测的高级特征 |

这种设计让 Draft Model 能从 Target Model 的**不同抽象层次**获取信息，实现更好的知识蒸馏效果。

### 2. OnlineEagle3Model 的前向传播

#### 输入

| 输入              | 形状                              | 含义                          |
| --------------- | ------------------------------- | --------------------------- |
| `input_ids`     | (batch, seq_len)                | Token ID 序列                 |
| `hidden_states` | (batch, seq_len, 3×hidden_size) | Target model 3个辅助层的隐藏状态拼接   |
| `target`        | (batch, seq_len, vocab_size)    | Target model 的 logits（训练目标） |
| `loss_mask`     | (batch, seq_len)                | 标记哪些位置计算 loss               |


```python
def forward(
    self,
    input_ids: torch.Tensor,        # (batch, seq_len) - token序列
    attention_mask: torch.Tensor,   # (batch, seq_len) - 有效位置掩码
    target: torch.Tensor,           # (batch, seq_len, vocab_size) - Target模型的logits
    loss_mask: torch.Tensor,        # (batch, seq_len, 1) - 计算loss的位置
    hidden_states: torch.Tensor,    # (batch, seq_len, 3*hidden_size) - Target模型的辅助层隐藏状态
    ...
)
```

---

#### Step 1: 处理词汇表映射和目标概率

```python
target_p_padded, position_mask = _compute_target_p_padded(
    target=target,
    t2d=self.draft_model.t2d,
    loss_mask=loss_mask,
    length=self.length,
)
```

这一步做了什么？看 `_compute_target_p` 函数：

```python
def _compute_target_p(target, t2d, loss_mask):
    target_head = target  # (batch, seq_len, vocab_size)
    
    # 1. 找出 target 预测的最大概率 token
    target_max_token = target_head.argmax(-1)  # (batch, seq_len)
    
    # 2. 检查这个 token 是否在 draft 词汇表中
    target_mask = t2d[target_max_token]  # t2d 是布尔数组，True表示在draft词汇表中
    target_mask = target_mask[..., None].int()  # (batch, seq_len, 1)
    
    # 3. 只有 loss_mask=1 且 token在draft词汇表中 的位置才计算loss
    position_mask = target_mask * loss_mask  # (batch, seq_len, 1)
    
    # 4. 压缩词汇表：只保留 draft 词汇表中的 logits
    target_head = target_head[..., t2d]  # (batch, seq_len, draft_vocab_size)
    
    # 5. 转换为概率分布
    target_p = nn.Softmax(dim=2)(target_head.float())  # soft label
    
    return target_p, position_mask
```

**然后进行 padding**：

```python
# 为 TTT 多轮展开预留空间
target_p_padded = F.pad(
    target_p,
    pad=(0, 0, 0, length),  # 在 seq_len 维度末尾填充 length 个位置
    mode="constant",
    value=1 / target_p.shape[-1],  # 填充均匀分布
)
# 形状: (batch, seq_len + 7, draft_vocab_size)
```

**示意图**：
```
原始 target_p:     [P0] [P1] [P2] [P3] [P4]
padding 后:        [P0] [P1] [P2] [P3] [P4] [均匀] [均匀] [均匀] [均匀] [均匀] [均匀] [均匀]
                                            ←───── 填充 7 个位置 ─────→
```

---

#### Step 2: 投影隐藏状态

```python
hidden_states = self.draft_model.project_hidden_states(hidden_states)
# 输入: (batch, seq_len, 3 * hidden_size) = (batch, seq_len, 12288)
# 输出: (batch, seq_len, hidden_size) = (batch, seq_len, 4096)
```

将 3 层拼接的隐藏状态投影回原始维度：

```
[layer1_hidden | layer15_hidden | layer28_hidden]  →  MLP  →  [projected_hidden]
     4096            4096            4096                          4096
```

---

#### Step 3: 准备位置编码

```python
if position_ids is None:
    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
# 结果: [0, 1, 2, 3, ..., seq_len-1]
```

---

#### Step 4: 准备注意力掩码

```python
if self.attention_backend in ("sdpa", "usp"):
    attention_mask = self.draft_model.prepare_decoder_attention_mask(
        attention_mask=attention_mask,
        hidden_states=hidden_states,
        batch_size=batch_size,
        seq_length=seq_length,
        past_key_values_length=past_key_values_length,
    )
```

生成 causal attention mask（下三角矩阵），确保每个位置只能看到之前的 token。

---

#### Step 5: TTT 循环（核心！）

```python
plosses = []  # 每轮的 loss
acces = []    # 每轮的准确率

# 初始化缓存
cache_hidden = [[], []]  # 用于存储中间隐藏状态
past_key_values = None   # KV cache

for idx in range(self.length):  # 默认循环 7 次
```

#### 5.1 获取当前轮的目标

```python
target_p = target_p_padded[:, idx : idx + seq_length, :]
```

**关键理解**：每轮使用不同的目标切片！

```
seq_len = 5, length = 7

轮次0: target_p_padded[:, 0:5, :]  → 预测位置 1,2,3,4,5
轮次1: target_p_padded[:, 1:6, :]  → 预测位置 2,3,4,5,6
轮次2: target_p_padded[:, 2:7, :]  → 预测位置 3,4,5,6,7
...
轮次6: target_p_padded[:, 6:11, :] → 预测位置 7,8,9,10,11
```

---

#### 5.2 词嵌入

```python
inputs_embeds = self.draft_model.embed_input_ids(input_ids)
inputs_embeds = inputs_embeds.to(hidden_states.dtype)
# (batch, seq_len) → (batch, seq_len, hidden_size)
```

---

#### 5.3 Draft Model Backbone 前向传播

```python
hidden_states_out = self.draft_model.backbone(
    input_embeds=inputs_embeds,      # 词嵌入
    hidden_states=hidden_states,      # 投影后的 target hidden states
    cache_hidden=cache_hidden,        # 历史隐藏状态缓存
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_values=past_key_values,  # KV cache
    use_cache=True,
)
```

**Draft Model 内部做了什么**：

```
┌─────────────────────────────────────────────────────────────┐
│                    Draft Model Backbone                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   inputs_embeds ─────┐                                       │
│   (词嵌入)            │                                       │
│                      ├──→ [Concat] ──→ (batch, seq, 2*hidden)│
│   hidden_states ─────┘                                       │
│   (target提示)              ↓                                 │
│                        [Transformer Layers]                  │
│                             ↓                                │
│                      hidden_states_out                       │
│                      (batch, seq, hidden)                    │
└─────────────────────────────────────────────────────────────┘
```

**更新隐藏状态**：
```python
hidden_states = hidden_states_out  # 用当前输出作为下一轮的输入
```

---

#### 5.4 计算 Logits

```python
logits = self.draft_model.compute_logits(hidden_states)
# (batch, seq_len, hidden_size) → (batch, seq_len, draft_vocab_size)

logits = gather_outputs_and_unpad(logits, gather_dim=1)  # 序列并行时收集
```

---

#### 5.5 计算准确率（不计算梯度）

```python
with torch.no_grad():
    acces.append(
        _compute_metric_acc(
            logits=logits,
            target_p=target_p,
            position_mask=position_mask,
            loss_mask=loss_mask,
        )
    )
```

准确率计算：
```python
def _compute_metric_acc(logits, target_p, position_mask, loss_mask):
    # Draft 预测的 token == Target 预测的 token
    correct = (logits.argmax(-1) == target_p.argmax(-1))
    # 只统计有效位置
    correct_masked = correct * position_mask.squeeze(-1)
    # 准确率 = 正确数 / 总有效位置数
    return correct_masked.sum() / loss_mask.sum().clamp_min(1e-6)
```

---

#### 5.6 计算 Loss

```python
loss = LogSoftmaxLoss.apply(logits, target_p, position_mask)
plosses.append(loss)
```

**Loss 计算**（soft label 交叉熵）：
```
Loss = -Σ target_p * log(softmax(logits))
```

这本质上是 **KL 散度**，让 Draft 的输出分布逼近 Target 的输出分布。

---

#### 5.7 为下一轮准备（左移/右移）

```python
if not is_last:
    global_input_ids = padding(global_input_ids, left=False)  # 右移
    position_mask = padding(position_mask, left=False)
    loss_mask = padding(loss_mask, left=False)
```

**padding 操作**：
```python
# padding(x, left=False) 实现右移
# 原始: [A, B, C, D, E]
# 右移: [B, C, D, E, PAD]
```

**为什么要右移？**

TTT 的核心是让 Draft Model 学会**预测越来越远的未来**：

```
原始输入:    [A] [B] [C] [D] [E]
             ↓   ↓   ↓   ↓   ↓
轮次0 输入:  [A] [B] [C] [D] [E]  → 预测 [B] [C] [D] [E] [F]
             
轮次1 输入:  [B] [C] [D] [E] [_]  → 预测 [C] [D] [E] [F] [G]
             
轮次2 输入:  [C] [D] [E] [_] [_]  → 预测 [D] [E] [F] [G] [H]
             
...以此类推...
```

每一轮，输入序列右移一位，目标也右移一位，这样：
- **轮次 0**：预测下一个 token（1 步提前）
- **轮次 1**：预测下下个 token（2 步提前）
- **轮次 6**：预测第 7 个未来 token（7 步提前）

---

#### 完整数据流图

```
输入数据
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                           Step 1: 目标处理                                │
│                                                                          │
│  target (vocab_size=128K) ──→ 词汇表压缩 ──→ target_p (draft_vocab=32K)   │
│                           ──→ padding ──→ target_p_padded                │
│                           ──→ position_mask (有效计算位置)                │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        Step 2: 隐藏状态投影                               │
│                                                                          │
│  hidden_states (3×4096=12288) ──→ Linear ──→ hidden_states (4096)        │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         Step 5: TTT 循环 (7轮)                            │
│                                                                          │
│  ┌─────────────────── 每轮循环 ───────────────────┐                      │
│  │                                                │                      │
│  │  input_ids ──→ embed ──┐                       │                      │
│  │                        ├──→ concat ──→ backbone ──→ logits            │
│  │  hidden_states ────────┘                       │        │             │
│  │                                                │        ▼             │
│  │  target_p[:, idx:idx+seq] ←────────────────────┼───→ Loss + Acc       │
│  │                                                │                      │
│  │  input_ids = padding(input_ids)  # 右移        │                      │
│  │  hidden_states = hidden_states_out  # 更新     │                      │
│  │                                                │                      │
│  └────────────────────────────────────────────────┘                      │
│                           │                                              │
│                           ▼                                              │
│                    重复 7 次                                              │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
输出: plosses (7个loss), acces (7个准确率)
```

---

### 3. Loss 权重衰减

上述前向传播得到：

```python
plosses, acces = run_forward(
                args, eagle3_model, data, target_model, is_online
            )
```

| 输出 | 含义 |
|------|------|
| `plosses` | 7 个位置的 loss 列表 |
| `acces` | 7 个位置的准确率列表（draft 预测与 target 一致的比例） |


```python
# 在 run_backward_and_update 中
ploss_weight = [0.8**i for i in range(len(plosses))]
# [1.0, 0.8, 0.64, 0.512, 0.41, 0.33, 0.26]

ploss = (
        sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        / args.draft_accumulation_steps
    )
ploss.backward()

if global_step % args.draft_accumulation_steps == 0:
	optimizer.step()
```

**越近的位置权重越大**，因为：
- 预测第1个 token 最重要（准确率最高）
- 越远的预测越难，给予较小权重

### 总结

训练目标: 让 Draft Model 学会模仿 Target Model 的输出分布

输入: 
  - input_ids: 原始 token 序列
  - hidden_states: Target 模型的中间层特征（提供"提示"信息）

输出:
  - Draft 预测的 logits

Loss:
  - Draft logits 与 Target logits 的 KL 散度
  - 7 个位置的 loss 加权求和

关键技术:
  1. 共享 embedding（draft 和 target 用同一个词嵌入）
  2. 使用 target hidden states 作为额外输入（知识蒸馏）
  3. TTT 多步展开训练（一次学会预测多个 token）
  4. 词汇表压缩（draft 只预测高频 token）


| 设计                  | 目的                          |
| ------------------- | --------------------------- |
| **3层隐藏状态拼接**        | 从 Target 获取多层次的知识提示         |
| **词汇表压缩 (t2d)**     | 减小 Draft 输出维度，加速推理          |
| **TTT 7轮展开**        | 训练 Draft 预测多个未来 token       |
| **Soft Label Loss** | 知识蒸馏，学习 Target 的概率分布而非硬标签   |
| **position_mask**   | 跳过目标 token 不在 draft 词汇表中的位置 |
| **隐藏状态复用**          | 每轮的输出作为下一轮输入，模拟自回归生成        |

这是因为不同的 attention 实现对 mask 格式的要求不同：

## 三种 Attention Backend 的区别

### 1. SDPA (Scaled Dot-Product Attention)
```python
# PyTorch 原生的 F.scaled_dot_product_attention
# 需要预先计算完整的 4D attention mask
attention_mask = self.draft_model.prepare_decoder_attention_mask(...)
# 输出形状: (batch, 1, seq_len, seq_len) - 完整的因果掩码矩阵
```

SDPA 要求传入一个**预计算好的 4D mask**，形如：
```
[[0, -inf, -inf, -inf],
 [0,  0,  -inf, -inf],
 [0,  0,   0,  -inf],
 [0,  0,   0,   0  ]]
```

### 2. USP (Unified Sequence Parallelism)
```python
# 序列并行场景，需要特殊处理
# 同样需要预计算 4D mask，但可能有额外的分片逻辑
```

### 3. Flex Attention (PyTorch 2.0+)
```python
# 不需要预计算 mask！
# flex_attention 使用 block_mask 或动态计算
```

Flex Attention 的设计理念不同：
- 它使用 **score_mod** 函数动态修改 attention scores
- 或使用 **block_mask** 在计算时按需生成
- **不需要预先构建完整的 NxN mask 矩阵**

---

## 为什么 Flex Attention 更高效？

```python
# SDPA: 需要 O(N²) 内存存储 mask
mask = torch.zeros(batch, 1, seq_len, seq_len)  # 预分配

# Flex Attention: 只需要定义规则，动态计算
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx  # 按需计算，不存储
```

---

## 代码中的体现

```python
# Step 4: handle attention mask
if self.attention_backend in ("sdpa", "usp"):
    # SDPA/USP 需要预计算完整 mask
    attention_mask = self.draft_model.prepare_decoder_attention_mask(
        attention_mask=attention_mask,
        hidden_states=hidden_states,
        batch_size=batch_size,
        seq_length=seq_length,
        past_key_values_length=past_key_values_length,
    )
# flex_attention 不进入这个分支，保持原始 2D mask 或 None
# 它会在 attention 计算内部处理因果关系
```

---

## 总结

| Backend | Mask 格式 | 内存占用 | 处理方式 |
|---------|----------|---------|---------|
| SDPA | 4D tensor `(B,1,N,N)` | O(N²) | 预计算 |
| USP | 4D tensor + 分片 | O(N²/P) | 预计算 |
| Flex Attention | 函数/规则 | O(1) | 动态计算 |

Flex Attention 是更现代的实现，**把 mask 逻辑融入到 kernel 中**，避免了显式构建大矩阵，所以不需要 `prepare_decoder_attention_mask`。







