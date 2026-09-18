---
title: Jev / System One Models（TypeSafe AI）
created: 2026-09-18
tags:
  - 结构化输出
  - 校准
  - 分类模型
type: 技术博客
papername:
conference:
year: 2026
institution:
  - TypeSafe AI
---

## 基本信息

标题：Introducing System One Models & Jev

作者：Diogo Almeida（TypeSafe AI 创始人，前 OpenAI，RLHF / InstructGPT / ChatGPT 研究的共同发明人之一）

博客地址：https://typesafe.ai/blog/introducing-system-one-models-and-jev

文档：https://docs.typesafe.ai/ ｜ Eval 站：https://evals.typesafe.ai/ ｜ 控制台：https://console.typesafe.ai/

开源适配器（把 LLM 包成同接口，主模型不开源）：https://github.com/typesafe-ai/system-one-adapter-python

发布：2026-09-15，早期访问（waitlist），融资 4000 万美元，DCVC 领投。潜伏两年。

命名来自 Kahneman《思考，快与慢》的 System 1（快、直觉）；Jev 取自 William Stanley Jevons（杰文斯悖论：效率提升反而拉高总需求）。

## 一句话总结

**把「生成」换成了「打分」**：不输出文本，只在调用方预先声明的 schema 上输出一组带校准概率的结构化决策。定位是「代码里的智能 if 语句」，不是聊天。

> 类比：**Jev 之于 LLM ≈ BERT 之于 GPT**——只是被推到了前沿智能水平，且加了概率校准和任意 schema 零样本。

## 研究动机

作者的立论（我认为这条站得住）：

> "If a model can do a task 95% of the time but doesn't say when it's in the 5%, it can't automate that task."

也就是说，阻碍自动化的不是智能不够，是**模型给不出可信的不确定度**。RLHF 训出来的模型就算你 prompt 它报置信度，也系统性过度自信、前后不一致。他们对 RLHF 的批评是本质性的：**偏好优化会 mode dropping**（把概率分布压尖到某个受偏好的风格上），顺手把校准毁了。

他们预期大规模自动化会是 99% 机器对机器、1% 人机交互，所以「机器接口」比「聊天接口」重要——他们叫 Machine Native Intelligence。

## 接口形态（理解机制的钥匙）

`POST /v1/systemone`，只有两个输入：

- `state`：非结构化上下文（文本 / JSON / 文本数组）。**目前只吃文本**，不支持图像音视频。
- `questions`：三种原语，可混着一次问。

| 原语 | 问题形态 | 返回字段 |
|---|---|---|
| `Choice` | 从 N 个选项里选一个 | `choice` + 各选项 `probabilities` + `confidence` |
| `Score` | 在有序分档上打分 | `score`（可插值，如 1.4）+ 各档 `probabilities` + `confidence` |
| `Noul` | 是 / 否 | `noul`：yes 的概率 0–1（**无 confidence**） |

文档原话（这句是机制的关键证据）：

> "Every question is evaluated **in parallel and in isolation** against the same state in one go. **Adding questions barely changes the response time.**"

`confidence` 不是独立预测的，是从 `probabilities` 算出来的统计量（分布尖锐度塌成一个 0–1 标量）。官方推荐三段式用法：高置信自动执行 / 中置信降级确认 / 低置信转人工，且**阈值按动作风险分级**而不是全局一个数。

## 方法要点

### 训练：RLCD

| | 优化目标 |
|---|---|
| RLHF | 人类评分员偏好的文本 |
| RLVR | 能被程序验证的输出 |
| **RLCD**（Reinforcement Learning for Calibrated Decisions） | **校准的决策**：说 0.8 的那批，实际就该有 80% 对 |

校准（calibration）= 模型报的概率和真实命中率对得上。注意这是**群体性质**，不保证单条答案对。

### 架构：官方无任何细节，以下是我从公开约束反推

> ⚠️ 下面这张图是**推断**，不是官方公布。官方只说了「new model architecture + parallel sampler」，一个字细节没给。

一次前向，靠 **block-diagonal attention mask**（块对角注意力掩码，让不同问题互相看不见）把问题隔开：

```
[  state tokens  ][ Q1: instr + opt_a opt_b opt_c ][ Q2: instr + lv0 lv1 lv2 ][ Q3: ... ]
   ↑ 所有 block 都能看到 state
                   └── 只看 state + 自己，看不到 Q2/Q3 ──┘
```

- state 编码一次（长，是延迟大头）
- 每个 question 是独立 block，attend 到 state + 自身
- logit 从**每个选项 token 所在位置**读出（一个共享的标量 readout head），再在该 question 的选项集合上做 softmax

**关键点：不是「每个类别一个头」。** 类别是调用时用户现定义的任意字符串，固定一个类别一个头 = 固定标签集 = 零样本能力没了。**头只有一个，"类别"住在输入里，不在权重里。**

### 这个推断能解释掉的全部公开约束

| 公开事实 | 被解释成 |
|---|---|
| 零样本任意 schema | 选项是**输入 token**，不是权重矩阵的行 |
| **cardinality 上限 255** | 读出张量的固定宽度（`2^8-1`，像个 uint8 槽位），不是标签表大小 |
| 加问题几乎不增延迟 | 问题 block 短且互相 mask 开，不产生跨问题二次开销；延迟由 state 长度主导 |
| 「类型错误数学上不可能」 | softmax 的支撑集**就是**那几个选项位置，没有第 N+1 个位置 |
| `Score` 返回 1.4 这种插值 | 对 K 个有序档位的分布取**期望**，不是 argmax |
| `Noul` 唯独**没有 confidence** | Bernoulli 的分布形状就是 `p` 本身，confidence 会是 \|p−0.5\| 的单调函数，纯冗余 |
| confidence 是"从 probabilities 算出来的" | 分布尖锐度塌成标量，不是另一个预测头 |

七条全对上。其中 `Noul` 没有 confidence 这条最有说服力——只有在「输出就是一个概率分布、confidence 是它的导出量」的假设下才讲得通。

**未定的分歧点：选项之间有没有交互？** A) 同 block 内互相可见（cross-encoder 式）；B) 各自独立打分再归一化（bi-encoder 式）。倾向 A——因为超 255 基数要回退成「先独立打分、再显式选」的两阶段；若是 B，高基数本来就 O(N) 可分批，根本不需要两阶段。

### 为什么快、为什么输出免费

自回归成本 ≈ `输出 token 数 × 一次前向`，推理模型还要额外烧几百到几千 CoT token。Jev 输出长度是 **1**——省掉的不是常数因子，是**整个输出序列维度**。

所以 "output tokens FREE (too cheap to meter)" 的真相是：**输出压根不是 token**，边际成本接近零，只按输入收费。

## 实验设置与主要结果

| 指标 | Jev | 前沿 LLM |
|---|---|---|
| 端到端延迟 | 70–500 ms | 3–329 s |
| 输入计价 | $0.042 / MTok | $0.20–10 / MTok |
| 输出计价 | 免费 | 约 5× 输入 |
| 类型错误率 | 0%（结构保证，非实测） | >0 |

首页口径 **193.6× 快 / 444.6× 便宜** 来自他们自建的 workflow eval。

**Workflow eval 的做法**：不设 ground truth，所有模型跑同一个 workflow，拿 **GPT-6 Astra + Fable 5.1 的平均**当参考概率。LLM 侧统一套他们的 system-one-adapter 做约束输出。

Demo：玩 Doom（10 次查询/秒 ≈ $7/小时，基于结构化文本游戏状态而非图像）、Wikiracing（高基数选择）。

## 优点

- **立论正确**：把「可用的不确定度」而不是「更高的智能」当作自动化的瓶颈，这个诊断我认为是对的。
- **约束烧进结构而非套在外面**：LLM 的 constrained decoding / grammar sampling 是在自回归外面套一层 mask，本质还是逐 token 生成；Jev 是把约束做进模型结构本身。
- **nuance 写得很诚实**：每个结论后面都主动列自己的偏置来源（eval 是自己团队造的、参考答案偏向 OpenAI/Anthropic、0% 不是实测、速度是从西海岸笔记本测的）。这个姿态比大多数发布博客好。

## 局限与风险

1. **「不会幻觉」是定义换的，不是能力换的。** 类型错误率 0% ≠ 决策正确率高。它照样会选错分类，只是那个错误不叫幻觉了。官方自己写："Our number is not empirical. Schema matching is guaranteed."
2. **不可证伪性偏高。** **主动拒绝公开 benchmark**，理由包装成方法论主张（"Put no weight on public benchmarks"）。自建 eval 的参考答案是「最强 LLM 的平均」——衡量的是**对齐程度**而非正确率，且 workflow 由自己的 model capabilities team 构造。「没法测」和「不给测」是两回事。
3. **缺的对照组才是关键。** 全篇拿它跟前沿 LLM 比快多少，但真正该比的是**一个 fine-tune 的 encoder 分类器**。它相对后者的唯一卖点是「零样本 + 任意 schema + 前沿级理解力」——不用为每个新任务标数据重训。这才是它值不值钱的地方，博客里一个字没提。
4. **「新架构」是空词。** 零细节。encoder + 并行 query head 这个形状本身一点不新（BERT 时代就有）。
5. 只有 waitlist、服务在美西、模型不开源、只吃文本、cardinality ≤255。

## 实践建议

### 对我们（judge / 审核 / 离线标注）的相关性

这东西的形状和我们做 **judge / 审核兜底 / nsfw 判定 / 离线逐条标注** 完全同构：state 进、`Score`/`Noul` 出、带校准概率。

**最有价值的不是快，是校准。** 我们现在靠多 judge 投票去近似不确定度（见 index_a100_eval 里的 judge 位置偏见、多 judge 投票），根子就是 LLM 给不出可信置信度。RLCD 是从第一性原理上对这个问题的正面解法。

**但现实路径是离线批量评测 judge，不是在线兜底**——waitlist + 美西服务，线上链路延迟不现实。

### 拿到 access 后先跑的三个证伪实验

这套架构推断是可实测的：

1. **固定 state，问题数 1 → 20**：延迟应近似平坦，且 **input token 计费基本不变**（state 只算一次）。若 token 数线性涨 → state 被重复编码，上面的图错了。
2. **固定问题，state 长度 ×10**：延迟应近似线性涨，验证「延迟由 state 主导」。
3. **固定 state 和问题，选项数 3 → 200**：延迟明显涨 → 选项进了序列（支持 A）；几乎不动 → 支持 B。

实验 1 顺带能算出**一次 state 编码挂多个判定维度的实际单价**——这直接决定它能不能替掉我们按条调 LLM 的离线标注成本模型。

### 不用等 access 就能做的

拉 `system-one-adapter-python` 看他们怎么把 LLM 约束成「带概率的结构化决策」。他们声称这是从 LLM 里取决策最准的方式，可以直接借鉴到我们现有 judge 上。

## 短评

**立论比产品扎实。** 「可用的不确定度是自动化瓶颈」+「RLHF 的 mode dropping 毁掉校准」这两条诊断是真东西，值得我们自己的 judge 体系借鉴，哪怕永远不用他们的 API。

**但证据链撑不住宣传口径。** 193.6× / 444.6× 这两个数字来自一个没有 ground truth、参考答案是竞品平均、题目由自己团队出的 eval。**我只当它证明了「在他们定义的任务形态上确实快得多」**，智能水平的可比性存疑。真创新大概率在数据不在架构——他们自己说 "TypeSafe is primarily a data research lab... We make all the data ourselves"。

**最该警惕的是话术层面的偷换**：把「类型错误率 0」包装成「不会幻觉」、把「输出不是 token」包装成「输出免费」。这两条技术上都成立，但读者接收到的意思和事实不是一回事。

## 参考资料

- 官方博客：https://typesafe.ai/blog/introducing-system-one-models-and-jev
- The Register 报道：https://www.theregister.com/ai-and-ml/2026/09/16/typesafe-ai-debuts-model-for-machines-that-plays-doom/5296711
- DataCamp 解读：https://www.datacamp.com/blog/system-one-models-jev
- 相关笔记：`PLM/Models/ChatGPT`、`PLM/Models/InstructGPT`（RLHF 谱系）、`PLM/Alignment/`（对齐方法）、`PLM/BERT/`（encoder 路线）
