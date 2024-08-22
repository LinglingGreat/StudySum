---
title: D-CPTLaw
created: 2024-08-21
tags:
  - ScalingLaw
  - 预训练
  - 增量预训练
  - 数据配比
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - 阿里
  - 香港科技大学
  - Waterloo
---

## 论文基本信息

标题：D-CPT Law: Domain-specific Continual Pre-Training Scaling Law for Large Language Models

作者：

链接：

代码：

框架图：


## 背景
通过拟合 D-CPT 定律，可以在有限的实验中使用小规模训练成本轻松预测任意混合比例、模型大小和数据集大小的通用和下游性能。

![](img/Pasted%20image%2020240821192407.png)

![](img/Pasted%20image%2020240821200442.png)
## 相关研究



## 核心亮点

1. D-CPT Law

![](img/Pasted%20image%2020240821192238.png)

2. Cross-Domain D-CPT Law

对于每个domain，定义参数Domain-specific Learnable Coefficient (DLC) K，代表特定领域的可学习性，值越小越容易学习。

![](img/Pasted%20image%2020240821201745.png)

![](img/Pasted%20image%2020240821201937.png)



## 实验

数据集：
- 包括6个领域Code, Math, Law , Chemistry, Music and Medical
- 训练集是充分的，不存在训练不饱和的情况
- 构造了高质量的held-out验证集

模型：Qwen-1.5-0.5B, Qwen-1.5-1.8B, and Qwen-1.5-4B

训练设置
- 参考Chinchilla，固定模型大小，调整训练数据token数来获取数据点
- 总的训练步数是20K，每1000步跑一次验证集loss
- 9个general-domain mix ratios, {0:10, 1:9, 2:8, 3.3:6.7, 5:5, 6.7:3.3, 8:2, 9:1, 10:0}

评估指标
- 使用验证集loss作为性能指标。
- 为了比较各种参数，使用 R^2 和 Huber 损失作为评估指标。具体来说，首先，the coefficient of determination（即R2）表示拟合质量，通常范围为0到1，其中值越高意味着回归模型的解释力越好。其次，Huber 损失结合了均方误差和平均绝对误差的特性，这对于异常值的回归特别有用。同样，Huber 损失还评估不同参数化的拟合质量，其中较低的 Huber 损失表明更好的拟合性能。

D-CPT Law

![](img/Pasted%20image%2020240821202720.png)

![](img/Pasted%20image%2020240821202838.png)

Effectiveness: 如表 1 所示，我们展示了五种不同参数化的性能。在有效性设置中，我们使用整个数据点进行拟合，目的是验证各种参数化的有效性。在表 1 中，我们观察到虽然 L5 的拟合参数最少，但与其他参数相比，其性能明显较差。 L1和L4的拟合参数相对较少，但与L2和L3相比，性能仍然较差。而且，L1不能满足隐式趋势的要求，而L4不能满足显式趋势的要求。最后，L2和L3的结果具有可比性，但L2不满足一致性要求。因此，我们选择L3作为D-CPT法则。此外，图 3 显示了 L3 在不同数据集大小、混合比率、模型大小和领域中的稳健有效性。

![](img/Pasted%20image%2020240821202917.png)

Model Size Generalizability: 我们的主要实验集中在 3 个模型尺寸：0.5B、1.8B 和 4B。我们使用3倍交叉验证来评估D-CPT法则的模型大小泛化性，跨域的平均结果如表2所示。例如，我们用0.5B和1.8的数据点拟合D-CPT法则B 并评估 4B 的 Huber 损失和 R2。在表 2 中，我们观察到 D-CPT 定律可以很好地推广到模型大小，并且 L3 显示出最佳性能。此外，我们对未见过的7B大小（即Qwen-1.5 7B）进行了实验，观察到D-CPT法则可以准确预测图4中通用语料混合比为0.2的通用语料库验证损失。

Dataset Size Generalizability: 我们的主要实验涵盖了从 0.1B 到 26B token 的数据集大小，并且我们还利用了 3 折交叉验证方法。将数据点统一分为三段，其中2/3用于拟合模型，其余1/3用于测试。在表 3 中，我们报告了跨域的平均结果，并观察到 ​​L3 显示出显着增强的数据集大小通用性。

![](img/Pasted%20image%2020240821203216.png)

Mixture ratio Generalizability: 我们在各种参数化中应用 k 折交叉验证方法。具体来说，我们选择 9 个混合比例中的 7 个进行拟合，其余的进行测试，从而每个域进行 36 个实验。为简单起见，我们在表 4 中显示了跨域的平均结果，并观察到 ​​L3 在混合比泛化性方面仍然表现出明显更好的性能。此外，在图 1 中，我们观察到我们的 D-CPT 定律对于未见的混合比具有良好的泛化性。

Usage 1: Trade-off between general and domain-specific abilities

Usage 2: Optimal mixture on limited domain-specific data

Usage 3: Resource allocation



## 未来方向



## 主要收获


## 参考资料
