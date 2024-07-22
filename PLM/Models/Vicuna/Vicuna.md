---
title: Vicuna
created: 2023-06-08
tags: LLM, SFT

---

最新的代码中为了保证每次训练验证集划分固定，加了随机种子`np.random.seed(0)`. 导致数据每次打乱结果都一样，训练过程中可能出现loss 曲线在每个epoch的时候有一个突降。

