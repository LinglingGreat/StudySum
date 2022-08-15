---
title: ESC
created: 2022-08-15
tags: [对话/情感, 论文]
---

论文：Towards Emotional Support Dialog Systems

每个strategy都对应一个tokenid

`torch.tensor([f.labels[0] for f in features], dtype=torch.long) - len(toker) + 8`

toker是所有的token

模型没有变化，只是加了一些loss和指标

输入：`[c + [eos] for c in context]`

输出：

labels `[strat_id] + response + [eos]`

decoder_input_ids `[bos] + labels[:-1]`

预测：

先预测策略，保证第一个token一定是从策略的token里面选的

然后预测回复