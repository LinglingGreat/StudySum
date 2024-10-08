---
title: SyntheticWithPersonahub
created: 2024-07-01
tags:
  - 数据
  - 数据合成
type: 论文
papername: Scaling Synthetic Data Creation with 1,000,000,000 Personas
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2024
institution:
  - 腾讯
---

## 论文基本信息

标题：Scaling Synthetic Data Creation with 1,000,000,000 Personas

作者：

链接：http://arxiv.org/abs/2406.20094

代码：https://github.com/tencent-ailab/persona-hub

框架图：

![](img/Pasted%20image%2020240701163405.png)


## 背景

这篇论文提出了一种新颖的、以人物角色（persona）为驱动的数据合成方法，旨在解决如何大规模创造多样化的合成数据的问题。具体来说，论文的主要贡献和解决的问题包括：

1. **合成数据的多样性和可扩展性**：传统的数据合成方法在扩展数据量时难以保持数据的多样性。论文提出的人物角色驱动的数据合成方法可以通过使用大量不同的人物角色来引导大型语言模型（LLM）创造具有不同视角的合成数据。
    
2. **合成数据的实用性**：通过人物角色驱动的方法，可以创建用于各种场景的合成数据，如数学和逻辑推理问题、用户提示、知识丰富的文本、游戏NPC对话和工具开发等。
    
3. **合成数据的自动化和易用性**：通过自动化地从网络数据中提取人物角色，并构建了一个包含10亿个多样化人物角色的集合——Persona Hub，使得合成数据的创造过程更加自动化和易于使用。


![](img/Pasted%20image%2020240701165553.png)

开源了一些数据

![](img/Pasted%20image%2020240701165810.png)

## 相关研究




## 核心亮点

论文通过以下几个关键步骤解决了如何大规模创造多样化合成数据的问题：

1. **提出人物角色驱动的数据合成方法**：这种方法利用大型语言模型（LLM）内的多种视角，通过添加人物角色到数据合成提示中，引导LLM从相应的视角创造独特的合成数据。
    
2. **构建Persona Hub**：这是一个包含10亿个多样化人物角色的集合，这些人物角色自动从网络数据中策划而来，作为世界知识的分布式载体，可以触及LLM内部封装的几乎每一个视角。
    
3. **使用Text-to-Persona和Persona-to-Persona方法**：这两种方法从大量网络数据（RedPajama v2）中派生出多样化的人物角色，
- Text-to-Persona通过特定文本推断可能阅读/写/喜欢/不喜欢该文本的角色（看后面的图）。
- Persona-to-Persona通过人物角色间的人际关系来衍生新的角色。我们对通过 Text-to-Persona 获得的每个角色进行六次角色关系扩展迭代（根据六度空间理论），从而进一步丰富我们的角色集合。（Text-to-Persona可以合成涵盖几乎每个方面的角色。然而，它仍然可能会错过一些在网络上知名度较低的角色，例如儿童、乞丐或电影的幕后工作人员。为了补充 Text-to-Persona 可能难以达到的角色，我们提出了 Persona-to-Persona，它从通过 Text-to-Persona 获得的角色中派生出具有人际关系的角色。）
    
4. **去重处理**：为了确保Persona Hub中人物角色的多样性，论文采用了基于MinHash（用角色描述的1-gram特征，signature size of 128，阈值0.9）和基于嵌入的去重方法（阈值0.9，OpenAI的text-embedding-3-small），移除相似度较高的角色。最终得到1,015,863,523 personas
    
5. **展示用例**：论文展示了使用Persona Hub在不同场景下合成数据的用例，包括数学和逻辑推理问题、用户提示、知识丰富的文本、游戏NPC和工具开发等。
    
6. **评估和验证**：通过使用不同的测试集（包括合成测试集和MATH基准测试集）来评估使用合成数据微调后的LLM的性能。
    
7. **讨论伦理和责任问题**：论文详细讨论了使用这种技术可能带来的伦理和责任问题，并强调了避免滥用和确保道德和负责任应用的重要性。

Text-to-Persona方法：

![](img/Pasted%20image%2020240701170539.png)

Persona-to-Persona方法：

![](img/Pasted%20image%2020240701171450.png)

角色驱动的数据合成：

![](img/Pasted%20image%2020240701205028.png)

举例说明（数学）

![](img/Pasted%20image%2020240701205456.png)

![](img/Pasted%20image%2020240701205552.png)




## 实验

论文中进行了一系列实验来验证所提出的人物角色驱动的数据合成方法的有效性。以下是主要的实验内容：

1. **数学问题合成**：使用从Persona Hub中选取的109 万个角色，通过0-shot prompting方法，利用GPT-4生成了109万个数学问题。然后，从这些问题中随机抽取了2万个作为合成测试集，剩余的 1.07M 用于训练。

- 分布内测试集：上述的20k测试集，又用gpt-4-turbo生成答案，保留两个模型答案一致的数据，剩余11.6K。
- MATH（分布外测试集）：最广泛认可的测试法学硕士数学推理能力的基准。其测试集包含 5,000 个竞争级别的数学问题以及参考答案。
    
2. **逻辑推理问题合成**：展示了使用人物角色驱动的方法合成典型逻辑推理问题的示例，并创建了中国特色的“Ruozhiba”风格的逻辑推理问题。

![](img/Pasted%20image%2020240701210932.png)
    
3. **指令合成**：通过零样本（0-shot）和人物角色增强的少样本（persona-enhanced few-shot）提示方法，合成了用于LLM的指令（即用户提示），并展示了如何使用这些指令生成模拟的用户-LLM对话（用LLM去合成）。

![](img/Pasted%20image%2020240701211050.png)
    
4. **知识丰富的文本合成**：利用从Persona Hub中抽样的人物角色，促使LLM撰写Quora文章，生成了信息丰富和知识密集的文本。

![](img/Pasted%20image%2020240701211245.png)
    
5. **游戏NPC合成**：展示了如何使用Persona Hub中的人物角色为“魔兽世界”和“天涯明月刀”等游戏创建多样化的非玩家角色（NPC）。
![](img/Pasted%20image%2020240704154814.png)

![](img/Pasted%20image%2020240704154854.png)

7. **工具（功能）开发**：使用人物角色来预测用户可能需要的工具，并展示了如何为这些人物角色开发工具的高级接口。

![](img/Pasted%20image%2020240704154910.png)

![](img/Pasted%20image%2020240704155018.png)



7. **评估和性能测试**：使用合成的数学问题对开源的7B LLM——Qwen2-7B进行微调，并在内部合成测试集和MATH基准测试集上评估其性能（表1和表2，图9）。
    
8. **质量检验**：对合成的数学问题进行了质量检验，包括由数学专家评估问题的有效性，以及分析不同人物角色间相似性对合成问题相似性的影响（图10）。
    
9. **性能对比**：将微调后的Qwen2-7B模型的性能与现有的开源LLMs和最先进的LLMs进行了对比。
    

这些实验不仅证明了人物角色驱动的数据合成方法能够创造出多样化和高质量的合成数据，还展示了这种方法在不同应用场景下的通用性和有效性。此外，通过性能评估，论文还展示了使用合成数据微调的LLM在数学问题解决能力上的显著提升。

评估结果：

![](img/Pasted%20image%2020240701210246.png)

![](img/Pasted%20image%2020240701210306.png)

![](img/Pasted%20image%2020240701210411.png)

此外，我们还专门研究了提示中人物角色差异对综合数学问题的影响。我们首先采样了 100 对语义相似度分别为 0.4、0.6 和 0.8 的角色。对于每对角色，我们使用它们通过贪婪解码（即温度=0）创建一对数学问题。然后，我们计算这些数学问题对的语义相似度，并在图 10 中显示结果。

![](img/Pasted%20image%2020240701210451.png)


## 未来方向

论文在最后部分提出了一些未来工作和进一步探索的点，主要包括：

1. **Persona Hub的改进**：尽管第一版的Persona Hub已经包含了10亿个人物角色，但这些角色的描述主要集中在主要方面，缺少细节。未来的工作计划将致力于细化这些人物角色的描述，使其更加独特，从而扩大Persona Hub的规模，并为合成数据创造提供更多机会。
    
2. **多模态LLM的数据合成**：当前的工作仅探索了基于文本的LLM的数据合成。未来的工作将考虑将这种方法应用于多模态LLM，以创造包括图像、视频和音频在内的多模态合成数据。
    
3. **使用超人物角色（super personas）指导LLM**：论文提出了使用特定的超人物角色来引导LLM探索现有知识范围之外的可能性，这可能为挖掘LLM的超级智能提供一种新方法。
    
4. **伦理和责任问题的深入研究**：考虑到合成数据可能带来的伦理和责任问题，如数据安全、错误信息的传播等，需要进一步研究如何在使用这种技术的同时确保伦理和负责任的应用。
    
5. **合成数据对LLM训练的影响**：研究合成数据对LLM训练过程的影响，包括数据的多样性、质量和规模如何影响模型的性能和泛化能力。
    
6. **合成数据在不同领域的应用**：探索合成数据在不同领域的应用，如教育、娱乐、医疗和金融等，以及如何根据不同领域的需求定制合成数据。
    
7. **合成数据的评估和验证**：开发更系统的方法来评估合成数据的质量，包括其准确性、可靠性和多样性，并研究如何验证合成数据在实际应用中的有效性。
    
8. **与其他AI技术的集成**：研究如何将合成数据技术与其他AI技术（如强化学习、知识图谱等）集成，以创造更复杂的应用场景。
    
9. **用户研究和反馈**：进行用户研究，收集用户对使用合成数据的LLM的反馈，以了解合成数据在实际用户体验中的表现和潜在的改进空间。
    

这些探索点表明，尽管论文提出了一种创新的数据合成方法，但仍有许多机会进一步发展和完善这项技术，以及更深入地理解其在实际应用中的潜力和影响。

## 主要收获


## 参考资料
