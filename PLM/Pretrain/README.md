---
title: README
created: 2024-08-08
tags:
  - ContinualLearning
---

## 参考资料

[浅谈领域模型训练](https://mp.weixin.qq.com/s/-do7_ZLaaIxSyXOfSk1E1A) 主要讲post-pretrain
- Scaling law: D-CPT明确指出“domain能力“和”general 能力“是相互冲突的
1. 小学习率，domain 学得快，通用忘得慢；
    
2. 大学习率，domain 学得快，但到一定地步后就震荡，毕竟学习能力有限；
    
3. 不同 size 的模型适合不同的学习率。

- 数据：质量高；配比，英文很重要（有些 paper 认为模型的 general knowledge 基本来自于英文语料，中文更多的是对齐作用），代码数学也要加
- 学习率：大模型的学习率小；记得warmup
- 优化器
- 做 domain post-pretrain 需要看 channel loss，也就是每个领域的训练loss
5. 初始 loss 低：任务简单，或者模型已经训过这份数据。如果你使用的底座模型效果巨强，比如是 qwen2-72B，llama3-70B，你甚至可以断言这个数据的质量很高（能力差的小模型不能随便下定论）。当然，loss 低也有可能存在一种情况，那就是数据十分的脏，全都是重复 token 或者 固定 pattern；
    
2. 初始 loss 高：好现象，说明模型没有见过这个数据。但也有数据质量很差的风险，最好再清洗下这个数据源；
    
3. loss 持平或缓慢下降：好现象，没有比这更好的现象了，基本就是我们蒙对了底座模型 pretrain 阶段使用的数据配比才会有的现象；
    
4. loss 快速下降：说明这个数据很容易学习，有可能是 domain 数据的特点比较显著，也有可能是数据比较脏，都是固定 pattern 或者具有明显的格式（提一句，llama 说任何 markdown 数据都对模型性能有损失，所以有明显格式的数据要慎重使用）；
    
5. common channel loss 下降明显：你的 common 数据显然不够 common，它相对模型来说有可能更像是 domain 数据，说明当前数据配比和 pretrain 的配比偏离有点远；
    
6. domain channel loss 下降明显：好事，鼓掌欢呼；
    
7. domain channel loss 不下降：初始 loss 低说明模型大概率已经训过这份 domain 数据了，初始 loss 高还不下降，可能是数据不够干净，也可能是数据比较难学，再多训会吧；
    
8. loss 上升：和导师或领导汇报就说学习率设置的不合适，自己私下再顺带 check 一下训练代码；
