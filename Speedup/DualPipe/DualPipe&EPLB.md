---
title: DualPipe
created: 2025-03-03
tags:
  - deepseek
---
[GitHub - deepseek-ai/DualPipe: A bidirectional pipeline parallelism algorithm for computation-communication overlap in V3/R1 training.](https://github.com/deepseek-ai/DualPipe)

DualPipe：一个双向流水线并行算法。

[GitHub - deepseek-ai/EPLB: Expert Parallelism Load Balancer](https://github.com/deepseek-ai/eplb)

EPLB：一种用于 MoE 的负载均衡算法。




都知道，大模型得靠并行撑着。

想象一条流水线：每个工位负责一个活儿，比如这边装引擎，那边安轮子，车子顺着流水线一步步成型，总比一个人从头造到尾快多了。

AI 训练里，管道并行也是这个道理，把模型分成几块，每块扔给一个 GPU，大家同时处理不同的数据，效率蹭蹭往上涨。

但问题来了，传统管道并行有个毛病：GPU 之间得来回传数据，有时候就像接力赛跑，交棒的时候没对上，前面的人跑完了后面的人还没动。这就造成了 管道气泡——GPU 闲着没事干，白白浪费时间。

所以，今天这俩工具，是通过**并行**，给大模型训练和推理提速。

## DualPipe：AI 训练的加速神器

这时候 DualPipe 登场了。它是个**双向管道并行算法**，意思是数据能双向流动——既能往前跑（处理输入），也能往后跑（反馈更新）。

这里有一个专业名字，叫**计算-通信高效重叠**，同时优化正向和反向传播阶段的计算与通信流程，两者重叠，这让 GPU 可以同时干两件事（计算和通信），几乎不闲着，气泡时间大幅减少。在 DeepSeek-V3 的技术报告里说明，这招能大大提升训练效率。

贴一张全面的对比图——

![图片](https://mmbiz.qpic.cn/mmbiz_png/5fknb41ib9qGsourSwR8shqqaZK84J8nYrkXdq4cW9SS4C42nSxMTrxmW68DlupglNSWjr4kgfSDOCdVulaoV3w/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

1F1B（one-forward-one-backward）：一种传统流水线并行策略，交替执行前向和后向传播计算，效率最低。

ZB1P（Zero-Bubble Pipeline）：也是一种消除流水线空闲时间（气泡）的并行策略，改进了 1F1B。

DualPipe：就是 ds 这次创新性的流水线并行算法。

**白色表示流水线气泡 Bubble，DualPipe 明显更少。**

来看个对比表，直观感受一下 DualPipe 的厉害：

![图片](https://mmbiz.qpic.cn/mmbiz_png/5fknb41ib9qGsourSwR8shqqaZK84J8nYcHGqJTd3BENRTlm9Xj231ZgvgdmibjicckB5Xn8ViaeU7OV52xQMGicCZQ/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

_（注：𝐹 是前向计算时间，𝐵 是反向计算时间，𝑊 是权重反向时间，𝐹&𝐵 是重叠的前向和反向时间。）_

DualPipe 把气泡时间砍了不少，但是代价是内存加倍。

因为实现双向并行，要维护两份模型参数。（两套生产设备）

本质是用空间换取时间。虽然参数用量翻倍，但在 AI 训练里，为了速度多花点力气绝对值得。

项目里还注明了是梁文峰亲自参与了 DualPipe。

## EPLB：GPU 的任务调度大师

接下来是 专家并行负载均衡器（EPLB），关于专家并行（EP）的科普，在第二天的文章[**DeepSeek开源第二天：拉爆MoE训练和推理**](https://mp.weixin.qq.com/s?__biz=MzIwNzc2NTk0NQ==&mid=2247604310&idx=1&sn=17a10c321d0733728c43b3a2eb045005&scene=21#wechat_redirect)有介绍。

在用 混合专家（MoE）结构里面，模型里有很多“专家”，每个专家擅长不同的活儿。当然这些专家被分到不同 GPU 上。  
有的专家忙得要死，有的却闲得发慌。EPLB 就是来解决这问题的。

它会把忙碌的专家复制一份，分到其他 GPU 上，让工作量平均摊开。就像交通管理员，看到哪条道堵了就多开几条，确保车流顺畅。这样一来，每个 GPU 都能干活，既不累死也不闲死，系统效率拉满。

EPLB 有两种玩法：

**层级负载均衡**：当服务器节点数能整齐划分专家组时用这招。先把专家组在节点间平衡好，再在每个节点里复制专家，最后分配到具体 GPU 上。这样能少折腾点跨节点的数据传输，速度更快。

**全局负载均衡**：其他情况下就直接全局复制专家，分到各个 GPU 上，不分组。这招在大规模推理的解码阶段特别好使。

举个例子，想象你在管一个厨房，有的厨师忙得锅都端不过来，有的却站着发呆。EPLB 就像请了几个帮手，把活儿重新分一分，大家节奏一致，菜出得又快又好。

![图片](https://mmbiz.qpic.cn/mmbiz_png/5fknb41ib9qGsourSwR8shqqaZK84J8nYjvO1aGMwK9uXibbfRqiaH5rFArRfeEcmf0nvt83yOfibl606xZfk4ddmg/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

这样能把专家任务均匀分到 GPU 上，谁也别想偷懒。

最后聊聊 DeepSeek Infra 。

这次顺便带了一个用于性能分析的库,它让我们能一窥 DualPipe 和 EPLB 在实战中的表现。

> https://github.com/deepseek-ai/profile-data

DeepSeek 用 PyTorch Profiler 记录了训练和推理的性能数据，还大方地分享了出来。

你可以下载这些数据，在浏览器里（输入 chrome://tracing 或 edge://tracing）可视化查看。

数据覆盖了三块：

**训练**：展示了 DualPipe 怎么在训练中叠加计算和通信。每个块有四个 MoE 层，跟 DeepSeek-V3 的预训练配置差不多，效率拉满。

![图片](https://mmbiz.qpic.cn/mmbiz_png/5fknb41ib9qGsourSwR8shqqaZK84J8nYqsYIBNPygqMekF1ed6R6ibzomOmviaOuSujVIMic7qibLyfLpHVQeFGVew/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

**预填充**：这是推理阶段处理初始输入的部分，用了 EP32 和 TP1 配置，两个微批次叠加计算和通信，工作量分配得刚刚好。

![图片](https://mmbiz.qpic.cn/mmbiz_png/5fknb41ib9qGsourSwR8shqqaZK84J8nYJctzeiaQvnZkaltFgkB4Ytn6fEYIgeFYSR0m5STlLgokxWXbUpbwFGA/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

**解码**：生成输出时用的是 EP128 和 TP1，通信通过 RDMA 异步处理，GPU 能专心算东西。

![图片](https://mmbiz.qpic.cn/mmbiz_png/5fknb41ib9qGsourSwR8shqqaZK84J8nY3icEwWkXIcIySX8vrrqjrdyyaV3t9hQlakD5AiazeqCLDQxU9kYKNISg/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

这些数据就像 AI 后台实况，让你清晰地看到每个动作是怎么完成的。