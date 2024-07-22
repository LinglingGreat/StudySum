---
title: ArenaLearning
created: 2024-07-20
tags:
  - 数据飞轮
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - 微软
  - 清华
---

## 论文基本信息

标题：Arena Learning : Build Data Flywheel for LLMs Post-training via Simulated Chatbot Arena

作者：

链接：

代码：https://github.com/nlpxucan/WizardLM

框架图：

![](img/Pasted%20image%2020240720170626.png)

将数据分成多个批次。
- 给定数据1，让模型v0和几个优秀的LLM PK，选出弱模型失败的那些数据，将强模型的输出作为groud truth去微调得到SFTv1。
- 给定数据2，让模型SFTv1和几个优秀的LLM PK，选出弱模型失败的那些数据pair对，DPO训练得到DPOv1
- 给定数据3，让模型DPOv1和几个优秀的LLM PK，选出弱模型失败的那些数据pair对，训练reward模型和PPOv1
- 再用PPOv1重复迭代上述过程
## 背景



## 相关研究




## 核心亮点

论文通过提出一个名为"Arena Learning"的创新框架来解决上述问题。以下是该框架的关键组成部分和解决策略：

1. **模拟聊天机器人竞技场（Simulated Chatbot Arena）**：使用AI驱动的注释来模拟在线聊天机器人竞技场比赛，从而减少人类标注的需求。
2. **WizardArena**：开发了一个精确的预测管道，用于预测不同模型的Elo排名，并通过精心设计的离线测试集来维持在线竞技场的一致性。
3. **数据飞轮（Data Flywheel）**：通过战斗结果不断更新训练数据，突出目标模型的弱点，并从多个不同模型的优势中学习。
4. **迭代战斗和模型进化（Iterative Battle and Model Evolving）**：通过迭代过程，目标模型（如WizardLM-β）在每次模拟竞技场战斗和训练数据生成后都会更新，并重新引入竞技场进行进一步的战斗。
5. **训练策略**：包括监督式微调（SFT）、直接偏好优化（DPO）和近端策略优化（PPO），这些策略使得模型能够从其他优秀模型中学习。
6. **自动化评估**：使用“法官模型”（judge model）(Llama3-70B-Chat) 自动模仿人类评估者的方式，评估两个模型的响应对，并提供排名、得分和解释。
7. **成本效益分析**：通过使用AI LLMs进行战斗，相比于传统的人类标注方法，实现了显著的成本节约和效率提升。
8. **实验验证**：通过实验结果展示了WizardArena与在线LMSys Chatbot Arena的高一致性，并证明了通过Arena Learning训练的模型在多个基准测试中的性能提升。

![](img/Pasted%20image%2020240720171315.png)

数据飞轮，迭代优化

![](img/Pasted%20image%2020240720171516.png)

![](img/Pasted%20image%2020240720171738.png)
## 实验

使用随机采样的ShareGPT数据训练初始模型，并从公开可用的数据集中收集指令，经过筛选、清洗、去重等步骤，构建了用于训练和评估的276K数据集。

1. **构建离线测试集**：通过K-Means聚类处理源数据，构建了包含多样性和难度的测试集，包括Offline-Diverse WizardArena和Offline-Hard WizardArena，并将它们合并为Offline-Mix WizardArena。
2. **LLM Battle**：在OfflineMix WizardArena上进行了模型之间的成对战斗，使用Llama3-70B-Instruct作为“法官”模型，根据得分高低判断胜负。
    
2. **评估一致性**：比较了WizardArena与在线的LMSYS ChatBot Arena的排名一致性，以及与MT-Bench的比较。
    
3. **数据飞轮迭代训练**：展示了使用Arena Learning方法在三个数据飞轮迭代中对WizardLM-β模型进行后训练的影响，包括在WizardArena-Mix ELO分数和MT-bench分数上的进步。
    
4. **迭代SFT, DPO, 和 PPO的影响**：探索了在SFT、DPO和PPO阶段，随着从更多Arena Learning战斗迭代中添加更多选定数据，模型性能如何逐步提高。
    
5. **消融研究**：
    
    - **数据选择策略**：比较了使用不同数据选择策略在SFT阶段的效果。
        
    - **数据大小与性能的关系**：探讨了数据大小和质量对模型性能的影响。
        
    - **法官模型的选择**：比较了使用Llama3-70B-Instruct和GPT-4作为法官模型的一致性。
        
    - **战斗模型的数量**：研究了参与战斗的不同模型数量对模型性能的影响。
        
    - **不同的战斗模式**：探索了使用不同战斗模式构建数据飞轮的必要性。
        
    - **在更多基准上的性能**：在多个基准上评估了经过多次迭代后的WizardLM-β模型的性能。
        
6. **性能影响分析**：分析了在不同阶段使用更先进的模型与WizardLM-β-7B-I0进行战斗的性能影响。


## 未来方向



## 主要收获


## 参考资料
