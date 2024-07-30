
## 数据梳理

英文：refined web，redpajama，pile，redpajamav2

中文：wudao，wanjuan（中英），BAAI-CCI，tigerbot（中英），skywork，MVBNC（中英）

- “中文互联网语料库”（Chinese Corpora Internet, 简称CCI）首期开放的数据（CCI v1.0.0）规模为 104GB。数据集总体的时间跨度为2001年1月至2023年11月。数据地址： https://data.baai.ac.cn/details/BAAI-CCI

多语言混合：mC4，OSCAR，CC100，BigScienceROOT
- CulturaX：对mC4和OSCAR进行清洗得到

【OpenWebMath：包含互联网上大部分高质量数学文本的数据集，从 Common Crawl 的超过 2000 亿 HTML 文件中过滤并提取出包含 147 亿 Token 的 630 万份文档，OpenWebMath 旨在用于预训练和微调大型语言模型】《open-web-math/open-web-math · Datasets at Hugging Face》

[m-a-p/Matrix · Datasets at Hugging Face](https://huggingface.co/datasets/m-a-p/Matrix) 包含 46900 亿个 token 的开源预训练数据集，这个包含中英文文本的双语数据集用于训练 Neo 模型。

## 数据清洗流程

第一步需要进行语言识别，以便适当地将数据分配给相应的语言。先前的研究表明，cld3比FastText差很多.

接下来的步骤是采用各种特定数据集的规则和启发式方法，根据特殊字符、短行、坏词等的比例过滤不受欢迎的内容。

数据还可以通过轻量级模型进行过滤，例如通过KenLM语言模型，以避免出现噪声文档。

最后，应进行数据去重，以去除相似或重复的信息。这方面的一个重要步骤是在文档层面进行模糊重复数据删除，例如通过MinHash来删除相似文档，从而减少记忆并提高LLM的泛化效果。

## 参考资料

[大语言模型（LLM）预训练数据集调研分析](https://mp.weixin.qq.com/s/CoZkPnxsB6Ay3RCJ8nl5BQ?forceh5=1)

【Galactic：用于处理大规模非结构化文本数据集的工具，提供清理和筛选功能，旨在筛选微调数据集、创建用于检索增强生成(RAG)的文档集合，甚至对LLM预训练Web规模数据集进行去重】'Galactic - data cleaning and curation for unstructured text' Taylor AI GitHub: github.com/taylorai/galactic

【Data-Juicer: 一站式数据处理系统，旨在为大语言模型 (LLM) 提供更高质量、更丰富、更易“消化”的数据】'Data-Juicer: A One-Stop Data Processing System for Large Language Models - A one-stop data processing system to make data higher-quality, juicier, and more digestible for LLMs!' Alibaba GitHub: github.com/alibaba/data-juicer

[再看大模型预训数据质量如何评估：困惑度、错误L2范数和记忆化三种度量方法的效果对比分析研究](https://mp.weixin.qq.com/s/d7fxiScyBIhyKYBi5wPgfw)

[GitHub - ZigeW/data\_management\_LLM: Collection of training data management explorations for large language models](https://github.com/ZigeW/data_management_LLM)

[Karpathy点赞，这份报告教你如何用 LLaMa 3创建高质量网络数据集](https://mp.weixin.qq.com/s/luZGMG1RRUT4X_ckt8hsCQ)







