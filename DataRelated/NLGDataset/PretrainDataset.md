
## 数据梳理

英文：refined web，redpajama，pile，redpajamav2

中文：wudao，wanjuan（中英），BAAI-CCI，tigerbot（中英），skywork，MVBNC（中英）

- “中文互联网语料库”（Chinese Corpora Internet, 简称CCI）首期开放的数据（CCI v1.0.0）规模为 104GB。数据集总体的时间跨度为2001年1月至2023年11月。数据地址： https://data.baai.ac.cn/details/BAAI-CCI

多语言混合：mC4，OSCAR，CC100，BigScienceROOT
- CulturaX：对mC4和OSCAR进行清洗得到

【OpenWebMath：包含互联网上大部分高质量数学文本的数据集，从 Common Crawl 的超过 2000 亿 HTML 文件中过滤并提取出包含 147 亿 Token 的 630 万份文档，OpenWebMath 旨在用于预训练和微调大型语言模型】《open-web-math/open-web-math · Datasets at Hugging Face》

[m-a-p/Matrix · Datasets at Hugging Face](https://huggingface.co/datasets/m-a-p/Matrix) 包含 46900 亿个 token 的开源预训练数据集，这个包含中英文文本的双语数据集用于训练 Neo 模型。

#### **1. CCI2-Data**

为了解决中文高质量安全数据集的稀缺问题，BAAI于2023年11月29日开源了CCI（Chinese Corpora Internet）数据集，并在此基础上进一步扩展数据来源，采用更严格的数据清洗方法，完成了CCI 2.0数据集的构建。CCI 2.0由来自可靠互联网来源的高质量数据组成，经过严格的清洗、去重和质量过滤处理。数据处理包括基于规则的关键词和垃圾信息过滤、基于模型的低质量内容筛选，以及数据集内部和之间的去重处理。最终发布的CCI 2.0语料库总容量为501GB，是一个高质量且可靠的中文安全数据集。

#### **2. SkyPile-150B**

SkyPile-150B是一个专为大规模语言模型预训练设计的综合性中文数据集，涵盖了来自广泛的公开中文互联网网页的数据。为了确保数据质量，SkyPile-150B经过了严格的过滤、广泛的去重以及全面的敏感数据过滤处理。还使用了先进的工具，如fastText和BERT，来过滤低质量数据。该数据集的公开部分包含约2.33亿个独特的网页，每个网页平均包含超过1000个汉字。整个数据集共计约1500亿个tokens，纯文本数据的总容量达620GB，全部为中文数据。

#### **3. IndustryCorpus**

IndustryCorpus 是一个由BAAI发布的多行业中文预训练数据集，旨在提升行业模型的性能。该数据集总量约为3.4TB，涵盖了包括医疗、教育、法律、金融等在内的18个行业的数据。IndustryCorpus的数据来自Wudao等多个大型数据集，并经过22个行业特定数据处理操作的精细清洗和过滤，最终生成了1TB的高质量中文数据和2.4TB的英文数据。由于其丰富的行业覆盖和严格的数据处理流程，该数据集特别适用于行业特定的语言模型训练，已经在医学领域的模型训练中展示了显著的性能提升。

#### **4. Tele-AI**

TeleChat-PTD是一个从电信星辰大模型TeleChat预训练语料中抽取出的综合性大规模中文数据集，数据集的原始大小约为1TB，经过压缩后为480GB，共分为189个文件。该数据集的数据主要来源于网页、书籍和官方媒体等多种渠道。采用了规则和模型结合的方式对数据进行了相关的过滤，并进行了相似性去重，但要训练好的模型还需要更高质量的处理。

#### **5. MAP-CC**

MAP-CC（Massive Appropriate Pretraining Chinese Corpus）是一个规模庞大的中文预训练数据集，专为训练中文大模型而设计。该数据集包含800亿个Token，由多个子集组成，每个子集都来自不同的数据源，如：博客、新闻文章、中文百科全书、中文学术论文、中文图书等。尽管MAP-CC进行了一系列的去重处理和低质量数据筛除，但以客观的眼光来看数据质量还是偏低，往往需要进一步筛选才能用于训练。

#### Chinese Fineweb Edu

[opencsg/chinese-fineweb-edu · Datasets at Hugging Face](https://huggingface.co/datasets/opencsg/chinese-fineweb-edu)

**Chinese Fineweb Edu**数据集是一个精心构建的高质量中文预训练语料数据集，专为教育领域的自然语言处理任务设计。该数据集通过严格的筛选和去重流程，利用少量数据训练打分模型进行评估，从海量的原始数据中提取出高价值的教育相关内容，确保数据的质量和多样性。最终，数据集包含约90M条高质量的中文文本数据，总大小约为300GB。

在数据筛选过程中，Chinese Fineweb Edu 数据集采用了与 Fineweb-Edu 类似的筛选策略，重点关注数据的教育价值和内容质量。具体筛选步骤如下：

1. **教育价值评估**：首先使用csg-wukong-enterprise打分模型对样本的教育价值进行评估，模型会根据样本内容的相关性和质量给出0-5的评分。在初步筛选阶段，我们选取了约100k条评分较高的数据。
    
2. **打分模型训练**：利用这100k条样本数据训练了一个BERT模型，用于对更大规模的预训练数据集进行文本打分。这一步确保了模型能够有效地识别出具有高教育价值的内容。
    
3. **数据筛选**：接下来，使用训练好的BERT模型对原始数据进行全面打分，仅保留得分大于4的数据。这一筛选过程极大地提高了数据集的质量和相关性，确保了其在教育领域的应用价值。
    
4. **MinHash去重**：为避免重复内容对模型训练的负面影响，数据集采用MinHash算法对所有数据进行了去重处理。这种方法确保了数据的独特性，同时保留了多样化的教育内容。

Chinese Fineweb Edu 数据集的原始数据来源广泛，涵盖了多个国内主流的中文预训练数据集。这些数据集虽然在规模和覆盖领域上各有不同，但通过精细筛选和处理，最终为Chinese Fineweb Edu 数据集提供了坚实的基础。主要数据来源包括：

- CCI2-Data：经过严格的清洗、去重和质量过滤处理，一个高质量且可靠的中文安全数据集。
    
- SkyPile-150B：一个来自中国互联网上的1500亿token大规模数据集，经过复杂的过滤和去重处理
    
- IndustryCorpus：一个涵盖多个行业的中文预训练数据集，包含1TB的中文数据，特别适合行业特定的模型训练
    
- Tele-AI：一个从电信星辰大模型TeleChat预训练语料中提取出的高质量大规模中文数据集，包含约2.7亿条经过严格过滤和去重处理的纯中文文本。
    
- MAP-CC：一个规模庞大的中文预训练语料库，结合了多种来源的高质量数据，特别针对中文语言模型的训练进行了优化



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







