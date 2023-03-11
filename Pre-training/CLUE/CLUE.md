---
title: CLUE
created: 2023-03-01
tags: Benchmark
type: 论文
papername: 
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 
institution: 
---

## 论文基本信息

标题：

作者：

链接： https://www.cluebenchmarks.com

包括小样本学习、基于数据的测评数据集、零样本学习、知识图谱问答、语义理解与匹配、搜索相关性、语义相似度、短文本分类、长文本分类、语言推理、代词消歧、关键词识别、阅读理解、NER

## FewCLUE小样本学习
FewCLUE：小样本学习测评基准-中文版，为Prompt Learning定制的学习基准，9大任务，5份训练验证集，公开测试，NLPCC2021测评任务。

下载地址：[https://github.com/CLUEbenchmark/FewCLUE](https://github.com/CLUEbenchmark/FewCLUE)   

文章：[https://arxiv.org/abs/2107.07498](https://arxiv.org/abs/2107.07498)

![](img/Pasted%20image%2020230301193022.png)

EPRSTMT:电商评论情感分析；CSLDCP：科学文献学科分类；TNEWS:新闻分类；IFLYTEK:APP应用描述主题分类；
OCNLI: 自然语言推理；BUSTM: 对话短文本匹配；CHID:成语阅读理解；CSL:摘要判断关键词判别；CLUEWSC: 代词消歧
EPRSTMT,CSLDCP,BUSTM 为新任务；其他任务（TNEWS,CHID,IFLYTEK,OCNLI,CSL,CLUEWSC）来自于CLUE benchmark，部分数据集做了新的标注。

## DataCLUE

下载地址：[https://github.com/CLUEbenchmark/DataCLUE](https://github.com/CLUEbenchmark/DataCLUE)  [中文文章](https://docs.qq.com/doc/p/51a1f46f64a72bc2ac0633427dc59d6e8d1b8d0a?dver=2.1.27277463)   [英文文章](https://arxiv.org/abs/2111.08647)

![](img/Pasted%20image%2020230301193139.png)

国内首个以数据为中心的AI测评。之前的测评一般是在固定的数据集下使用不同的模型或学习方式来提升效果，而DataCLUE是需要改进数据集。

系统化方式、通过迭代形式改进数据集：
#1.训练模型；
#2.错误分析：发现算法模型在哪些类型的数据上表现不佳（如：数据过短导致语义没有表达完全、一些类别间概念容易混淆导致标签可能不正确）
#3.改进数据：
  1）更多数据：数据增强、数据生成或搜集更多数据--->获得更多的输入数据。
  2）更一致的标签定义：当有些类别容易混淆的时候，改进标签的定义--->基于清晰的标签定义，纠正部分数据的标签。
#4.重复#1-#3的步骤。

参与测评者需要改进任务下的数据集来提升任务的最终效果；将使用固定的模型和程序代码（公开）来训练数据集，并得到任务效果的数据。 可以对训练集、验证集进行修改或者移动训练集和验证集建的数据，也可以通过非爬虫类手段新增数据来完善数据集。可以通过算法或程序或者结合人工的方式来改进数据集。


## ZeroCLUE 
下载地址：[https://github.com/CLUEbenchmark/ZeroCLUE](https://github.com/CLUEbenchmark/ZeroCLUE)   

文章：[https://arxiv.org/abs/2107.07498](https://arxiv.org/abs/2107.07498)

ZeroCLUE：零样本学习测评基准-中文版，9大任务，5份训练验证集，公开测试，采用与小样本学习一致的任务进行评估；为Prompt Learning定制的学习基准。  

零样本学习（Zero-shot Learning）是在通用的预训练模型基础上在模型没有调优情况下的机器学习问题。GPT系列模型展示了只依靠提示或示例即可以得到较好学习效果。结合预训练语言模型通用和强大的泛化能力基础上，探索零样本学习最佳模型和中文上的实践，是本课题的目标。

![](img/Pasted%20image%2020230301194110.png)

## KgCLUE 大规模知识图谱的问答

下载地址：[https://github.com/CLUEbenchmark/KgCLUE](https://github.com/CLUEbenchmark/KgCLUE)    [在线demo](http://www.cluebenchmarks.com:5000/)

KBQA（Knowledge Base Question Answering），是给定自然语言问题情况下通过对问题进行语义理解和解析，进而利用知识库进行查询、推理得出答案。

## SimCLUE 大规模语义理解与匹配数据集

[SimCLUE](https://github.com/CLUEbenchmark/SimCLUE)：大规模语义匹配数据集

提供一个大规模语义数据集；可用于语义理解、语义相似度、召回与排序等检索场景等。  
作为通用语义数据集，用于训练中文领域基础语义模型。 可用于无监督对比学习、半监督学习、Prompt Learning等构建中文领域效果最好的预训练模型。整合了中文领域绝大多数可用的开源的语义相似度和自然语言推理的数据集，并重新做了数据拆分和整理。

## QBQTC QQ浏览器搜索相关性

下载地址：[https://github.com/CLUEbenchmark/QBQTC](https://github.com/CLUEbenchmark/QBQTC)   

QQ浏览器搜索相关性数据集（QQ Browser Query Title Corpus），是QQ浏览器搜索引擎目前针对大搜场景构建的一个融合了相关性、权威性、内容质量、 时效性等维度标注的学习排序（LTR）数据集，广泛应用在搜索引擎业务场景中。  

相关性的含义：0，相关程度差；1，有一定相关性；2，非常相关。数字越大相关性越高。

## AFQMC 蚂蚁金融语义相似度
下载地址：[https://github.com/CLUEbenchmark/CLUE](https://github.com/CLUEbenchmark/CLUE)   [论文](https://aclanthology.org/2020.coling-main.419/)   [一种实现思路](https://kexue.fm/archives/8739)

每一条数据有三个属性，从前往后分别是 句子1，句子2，句子相似度标签。其中label标签，1 表示sentence1和sentence2的含义类似，0表示两个句子的含义不同。

## TNEWS' 今日头条中文新闻（短文）分类

下载地址：[https://github.com/CLUEbenchmark/CLUE](https://github.com/CLUEbenchmark/CLUE)   [论文](https://aclanthology.org/2020.coling-main.419/)   [一种实现思路](https://kexue.fm/archives/8739)

每一条数据有三个属性，从前往后分别是 分类ID，分类名称，新闻字符串（仅含标题）。

## IFLYTEK' 长文本分类

下载地址：[https://github.com/CLUEbenchmark/CLUE](https://github.com/CLUEbenchmark/CLUE)   [论文](https://aclanthology.org/2020.coling-main.419/)   [一种实现思路](https://kexue.fm/archives/8739)

该数据集共有1.7万多条关于app应用描述的长文本标注数据，包含和日常生活相关的各类应用主题，共119个类别："打车":0,"地图导航":1,"免费WIFI":2,"租车":3,….,"女性":115,"经营":116,"收款":117,"其他":118(分别用0-118表示)。每一条数据有三个属性，从前往后分别是 类别ID，类别名称，文本内容。

## CMNLI 语言推理任务
下载地址：[https://github.com/CLUEbenchmark/CLUE](https://github.com/CLUEbenchmark/CLUE)   [论文](https://aclanthology.org/2020.coling-main.419/)   [一种实现思路](https://kexue.fm/archives/8739)

CMNLI数据由两部分组成：XNLI和MNLI。数据来自于fiction，telephone，travel，government，slate等，对原始MNLI数据和XNLI数据进行了中英文转化，保留原始训练集，合并XNLI中的dev和MNLI中的matched作为CMNLI的dev，合并XNLI中的test和MNLI中的mismatched作为CMNLI的test，并打乱顺序。该数据集可用于判断给定的两个句子之间属于蕴涵、中立、矛盾关系。每一条数据有三个属性，从前往后分别是 句子1，句子2，蕴含关系标签。其中label标签有三种：neutral，entailment，contradiction。

## WSC Winograd模式挑战中文版-代词消歧
下载地址：[https://github.com/CLUEbenchmark/CLUE](https://github.com/CLUEbenchmark/CLUE)   [论文](https://aclanthology.org/2020.coling-main.419/)   [一种实现思路](https://kexue.fm/archives/8739)

威诺格拉德模式挑战赛是图灵测试的一个变种，旨在判定AI系统的常识推理能力。参与挑战的计算机程序需要回答一种特殊但简易的常识问题：代词消歧问题，即对给定的名词和代词判断是否指代一致。其中label标签，true表示指代一致，false表示指代不一致。

数据量：训练集(532)，验证集(104)，测试集(143)

例子：

{"target": {"span2_index": 28, "span1_index": 0, "span1_text": "马克", "span2_text": "他"}, "idx": 0, "label": "false", "text": "马克告诉皮特许多关于他自己的谎言，皮特也把这些谎言写进了他的书里。他应该多怀疑。"}
## CSL 论文关键词识别

下载地址：[https://github.com/CLUEbenchmark/CLUE](https://github.com/CLUEbenchmark/CLUE)   [论文](https://aclanthology.org/2020.coling-main.419/)   [一种实现思路](https://kexue.fm/archives/8739)

中文科技文献数据集包含中文核心论文摘要及其关键词。 用tf-idf生成伪造关键词与论文真实关键词混合，生成摘要-关键词对，关键词中包含伪造的则标签为0。每一条数据有四个属性，从前往后分别是 数据ID，论文摘要，关键词，真假标签。

数据量：训练集(532)，验证集(104)，测试集(143)

例子：

{"id": 1, "abst": "为解决传统均匀FFT波束形成算法引起的3维声呐成像分辨率降低的问题,该文提出分区域FFT波束形成算法.远场条件下,以保证成像分辨率为约束条件,以划分数量最少为目标,采用遗传算法作为优化手段将成像区域划分为多个区域.在每个区域内选取一个波束方向,获得每一个接收阵元收到该方向回波时的解调输出,以此为原始数据在该区域内进行传统均匀FFT波束形成.对FFT计算过程进行优化,降低新算法的计算量,使其满足3维成像声呐实时性的要求.仿真与实验结果表明,采用分区域FFT波束形成算法的成像分辨率较传统均匀FFT波束形成算法有显著提高,且满足实时性要求.", "keyword": ["水声学", "FFT", "波束形成", "3维成像声呐"], "label": "1"}

## CMRC2018 简体中文阅读理解

下载地址：[https://github.com/CLUEbenchmark/CLUE](https://github.com/CLUEbenchmark/CLUE)   [论文](https://aclanthology.org/2020.coling-main.419/)   [一种实现思路](https://kexue.fm/archives/8739)

第二届“讯飞杯”中文机器阅读理解评测 (CMRC 2018)

数据量：训练集(短文数2,403，问题数10,142)，试验集(短文数256，问题数1,002)，开发集(短文数848，问题数3,219)

例子：

`{ "version": "1.0", "data": [ { "title": "傻钱策略", "context_id": "TRIAL_0", "context_text": "工商协进会报告，12月消费者信心上升到78.1，明显高于11月的72。另据《华尔街日报》报道，2013年是1995年以来美国股市表现最好的一年。这一年里，投资美国股市的明智做法是追着“傻钱”跑。所谓的“傻钱”策略，其实就是买入并持有美国股票这样的普通组合。这个策略要比对冲基金和其它专业投资者使用的更为复杂的投资方法效果好得多。", "qas":[ { "query_id": "TRIAL_0_QUERY_0", "query_text": "什么是傻钱策略？", "answers": [ "所谓的“傻钱”策略，其实就是买入并持有美国股票这样的普通组合", "其实就是买入并持有美国股票这样的普通组合", "买入并持有美国股票这样的普通组合" ] }, { "query_id": "TRIAL_0_QUERY_1", "query_text": "12月的消费者信心指数是多少？", "answers": [ "78.1", "78.1", "78.1" ] }, { "query_id": "TRIAL_0_QUERY_2", "query_text": "消费者信心指数由什么机构发布？", "answers": [ "工商协进会", "工商协进会", "工商协进会" ] } ] } ] }`

## CHID 中文阅读理解填空

下载地址：[https://github.com/CLUEbenchmark/CLUE](https://github.com/CLUEbenchmark/CLUE)   [论文](https://aclanthology.org/2020.coling-main.419/)   [一种实现思路](https://kexue.fm/archives/8739)

成语完形填空，文中多处成语被mask，候选项中包含了近义的成语。

数据量：训练集(84,709)，验证集(3,218)，测试集(3,231)

例子：

`{ "content": [ # 文段0 "……在热火22年的历史中，他们已经100次让对手得分在80以下，他们在这100次中都取得了胜利，今天他们希望能#idiom000378#再进一步。", # 文段1 "在轻舟发展过程之中，是和业内众多企业那样走相似的发展模式，去#idiom000379#？还是迎难而上，另走一条与众不同之路。诚然，#idiom000380#远比随大流更辛苦，更磨难，更充满风险。但是有一条道理却是显而易见的：那就是水往低处流，随波逐流，永远都只会越走越低。只有创新，只有发展科技，才能强大自己。", # 文段2 "最近十年间，虚拟货币的发展可谓#idiom000381#。美国著名经济学家林顿·拉鲁什曾预言：到2050年，基于网络的虚拟货币将在某种程度上得到官方承认，成为能够流通的货币。现在看来，这一断言似乎还嫌过于保守……", # 文段3 "“平时很少能看到这么多老照片，这次图片展把新旧照片对比展示，令人印象深刻。”现场一位参观者对笔者表示，大多数生活在北京的人都能感受到这个城市#idiom000382#的变化，但很少有人能具体说出这些变化，这次的图片展按照区域发展划分，展示了丰富的信息，让人形象感受到了60年来北京的变化和发展。", # 文段4 "从今天大盘的走势看，市场的热点在反复的炒作之中，概念股的炒作#idiom000383#，权重股走势较为稳健，大盘今日早盘的震荡可以看作是多头关前的蓄势行为。对于后市，大盘今日蓄势震荡后，明日将会在权重和题材股的带领下亮剑冲关。再创反弹新高无悬念。", # 文段5 "……其中，更有某纸媒借尤小刚之口指出“根据广电总局的这项要求，2009年的荧屏将很难出现#idiom000384#的情况，很多已经制作好的非主旋律题材电视剧想在卫视的黄金时段播出，只能等到2010年了……"], "candidates": [ "百尺竿头", "随波逐流", "方兴未艾", "身体力行", "一日千里", "三十而立", "逆水行舟", "日新月异", "百花齐放", "沧海一粟" ] }`

## C3 中文多选阅读理解

下载地址：[https://github.com/CLUEbenchmark/CLUE](https://github.com/CLUEbenchmark/CLUE)   [论文](https://aclanthology.org/2020.coling-main.419/)   [一种实现思路](https://kexue.fm/archives/8739)

中文多选阅读理解数据集，包含对话和长文等混合类型数据集。

数据量：训练集(11,869)，验证集(3,816)，测试集(3,892)

例子：

`[ [ "男：你今天晚上有时间吗?我们一起去看电影吧?", "女：你喜欢恐怖片和爱情片，但是我喜欢喜剧片，科幻片一般。所以……" ], [ { "question": "女的最喜欢哪种电影?", "choice": [ "恐怖片", "爱情片", "喜剧片", "科幻片" ], "answer": "喜剧片" } ], "25-35" ], [ [ "男：足球比赛是明天上午八点开始吧?", "女：因为天气不好，比赛改到后天下午三点了。" ], [ { "question": "根据对话，可以知道什么?", "choice": [ "今天天气不好", "比赛时间变了", "校长忘了时间" ], "answer": "比赛时间变了" } ], "31-109" ]`

## CLUENER 细粒度命名实体识别

下载地址：[https://github.com/CLUEbenchmark/CLUENER2020](https://github.com/CLUEbenchmark/CLUENER2020)   [文章](https://arxiv.org/abs/2001.04351)   [一种实现思路](https://kexue.fm/archives/8739)

本数据是在清华大学开源的文本分类数据集THUCTC基础上，选出部分数据进行细粒度命名实体标注，原数据来源于Sina News RSS.

  

任务详情：[CLUENER2020](https://github.com/CLUEbenchmark/CLUENER2020)

训练集：10748 验证集：1343

标签类别：  
数据分为10个标签类别，分别为: 地址（address），书名（book），公司（company），游戏（game），政府（goverment），电影（movie），姓名（name），组织机构（organization），职位（position），景点（scene）

  

cluener下载链接：[数据下载](https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip)

  

例子：

{"text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，", "label": {"name": {"叶老桂": [[9, 11]]}, "company": {"浙商银行": [[0, 3]]}}}  
{"text": "生生不息CSOL生化狂潮让你填弹狂扫", "label": {"game": {"CSOL": [[4, 7]]}}}

  

标签定义与规则：

  
地址（address）: **省**市**区**街**号，**路，**街道，**村等（如单独出现也标记），注意：地址需要标记完全, 标记到最细。  
书名（book）: 小说，杂志，习题集，教科书，教辅，地图册，食谱，书店里能买到的一类书籍，包含电子书。  
公司（company）: **公司，**集团，**银行（央行，中国人民银行除外，二者属于政府机构）, 如：新东方，包含新华网/中国军网等。  
游戏（game）: 常见的游戏，注意有一些从小说，电视剧改编的游戏，要分析具体场景到底是不是游戏。  
政府（goverment）: 包括中央行政机关和地方行政机关两级。 中央行政机关有国务院、国务院组成部门（包括各部、委员会、中国人民银行和审计署）、国务院直属机构（如海关、税务、工商、环保总局等），军队等。  
电影（movie）: 电影，也包括拍的一些在电影院上映的纪录片，如果是根据书名改编成电影，要根据场景上下文着重区分下是电影名字还是书名。  
姓名（name）: 一般指人名，也包括小说里面的人物，宋江，武松，郭靖，小说里面的人物绰号：及时雨，花和尚，著名人物的别称，通过这个别称能对应到某个具体人物。  
组织机构（organization）: 篮球队，足球队，乐团，社团等，另外包含小说里面的帮派如：少林寺，丐帮，铁掌帮，武当，峨眉等。  
职位（position）: 古时候的职称：巡抚，知州，国师等。现代的总经理，记者，总裁，艺术家，收藏家等。  
景点（scene）: 常见旅游景点如：长沙公园，深圳动物园，海洋馆，植物园，黄河，长江等。


## OCNLI 原生中文自然语言推理

下载地址：[https://github.com/CLUEbenchmark/OCNLI](https://github.com/CLUEbenchmark/OCNLI)   [论文](https://arxiv.org/abs/2010.05444)   [一种实现思路](https://kexue.fm/archives/8739)

OCNLI，即原生中文自然语言推理数据集，是第一个非翻译的、使用原生汉语的大型中文自然语言推理数据集。 OCNLI包含5万余训练数据，3千验证数据及3千测试数据。除测试数据外，我们将提供数据及标签。测试数据仅提供数据。OCNLI为中文语言理解基准测评（CLUE）的一部分。  
  
10月22日之后，中文原版数据集OCNLI替代了CMNLI，使用bert_base作为初始化分数；可以重新跑OCNLI，然后再上传新的结果。

  

任务详情：[OCNLI](https://github.com/CLUEbenchmark/OCNLI)

训练集：50000+ 验证集：3000 测试集：3000

标签：  
label0-label4的不同标注者打上的标签。使用投票机制确定最终的label标签，若未达成多数同意，则标签将为"-"，（已经在我们的基准代码中处理了）

  

OCLNLI数据链接：[OCLNLI数据](https://github.com/CLUEbenchmark/OCNLI/tree/main/data/ocnli)

  

数据示例：

{
    "level":"medium",
    "sentence1":"身上裹一件工厂发的棉大衣,手插在袖筒里",
    "sentence2":"身上至少一件衣服",
    "label":"entailment",
    "label0":"entailment","label1":"entailment", "label2":"entailment",
    "label3":"entailment","label4":"entailment",
    "genre":"lit",
    "prem_id":"lit_635",
    "id":0
}
                

字段说明：

level: 难易程度：easy, medium and hard
sentence1: 前提句子
sentence2: 假设句子
label: 投票表决最终标签，少数服从多数原则，无法确定的使用"-"标记（数据已处理好）。
label0 -- label4: 5个标注员打上的标签
genre: 主要是：gov, news, lit, tv, phone
prem_id: 前提id
id: 总id

