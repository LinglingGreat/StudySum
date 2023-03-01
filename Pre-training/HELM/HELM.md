---
title: HELM
created: 2023-02-27
tags: Benchmark
type: 论文
papername: Holistic Evaluation of Language Models
conference: ACL/EMNLP/NAACL/EACL/COLING/SIGIR/AAAI/ICML/ICDM
year: 2022
institution: 斯坦福
---

## 论文基本信息

标题：Holistic Evaluation of Language Models

作者：

链接： https://arxiv.org/abs/2211.09110

代码： https://crfm.stanford.edu/helm/latest/

框架图：

包含[Stanford CRFM的](https://crfm.stanford.edu/)**语言模型整体评估**项目（[论文](https://arxiv.org/abs/2211.09110)、[网站](https://crfm.stanford.edu/helm/v1.0/)）**`crfm-helm`**中使用的代码。包括以下功能：

-   标准格式的数据集集合（例如，NaturalQuestions）
-   可通过统一 API 访问的模型集合（例如 GPT-3、MT-NLG、OPT、BLOOM）
-   收集超出准确性的指标（效率、偏差、毒性等）
-   用于评估稳健性和公平性的扰动集合（例如，拼写错误、方言）
-   从数据集构建提示的模块化框架
-   用于管理账户和提供统一接口访问模型的代理服务器


场景

[Question answering](https://crfm.stanford.edu/helm/latest/?group=question_answering "In question answering, given a question and (optionally, in open-book settings) a passage, the goal is to produce the answer. QA is a general format that captures a wide range of tasks involving varying levels of world and commonsense knowledge and reasoning abilities.")

-   [MMLU](https://crfm.stanford.edu/helm/latest/?group=mmlu "The Massive Multitask Language Understanding (MMLU) benchmark for knowledge-intensive question answering across 57 domains [(Hendrycks et al., 2021)](https://openreview.net/forum?id=d7KBjmI3GmQ).")
-   [BoolQ](https://crfm.stanford.edu/helm/latest/?group=boolq "The BoolQ benchmark for binary (yes/no) question answering [(Clark et al., 2019)](https://aclanthology.org/N19-1300/).")
-   [NarrativeQA](https://crfm.stanford.edu/helm/latest/?group=narrative_qa "The NarrativeQA benchmark for reading comprehension over narratives [(Kočiský et al., 2017)](https://aclanthology.org/Q18-1023/).")
-   [NaturalQuestions (closed-book)](https://crfm.stanford.edu/helm/latest/?group=natural_qa_closedbook "The NaturalQuestions [(Kwiatkowski et al., 2019)](https://aclanthology.org/Q19-1026/) benchmark for question answering based on naturally-occurring queries through Google Search. The input does not include the Wikipedia page with the answer.")
-   [NaturalQuestions (open-book)](https://crfm.stanford.edu/helm/latest/?group=natural_qa_openbook_longans "The NaturalQuestions [(Kwiatkowski et al., 2019)](https://aclanthology.org/Q19-1026/) benchmark for question answering based on naturally-occurring queries through Google Search. The input includes the Wikipedia page with the answer.")
-   [QuAC](https://crfm.stanford.edu/helm/latest/?group=quac "The QuAC benchmark for question answering in the context of dialogues [(Choi et al., 2018)](https://aclanthology.org/D18-1241/).")
-   [HellaSwag](https://crfm.stanford.edu/helm/latest/?group=hellaswag "The HellaSwag benchmark for commonsense reasoning in question answering [(Zellers et al., 2019)](https://aclanthology.org/P19-1472/).")
-   [OpenbookQA](https://crfm.stanford.edu/helm/latest/?group=openbookqa "The OpenbookQA benchmark for commonsense-intensive open book question answering [(Mihaylov et al., 2018)](https://aclanthology.org/D18-1260/).")
-   [TruthfulQA](https://crfm.stanford.edu/helm/latest/?group=truthful_qa "The TruthfulQA benchmarking for measuring model truthfulness and commonsense knowledge in question answering [(Lin et al., 2022)](https://aclanthology.org/2022.acl-long.229/).")

[Information retrieval](https://crfm.stanford.edu/helm/latest/?group=information_retrieval "In information retrieval, given a query and a set of candidate documents, the goal is to produce a ranking of the documents.")

-   [MS MARCO (regular)](https://crfm.stanford.edu/helm/latest/?group=msmarco_regular "The MS MARCO benchmark's regular track for passage retrieval in information retrieval [(https://microsoft.github.io/msmarco/)](https://microsoft.github.io/msmarco/).")
-   [MS MARCO (TREC)](https://crfm.stanford.edu/helm/latest/?group=msmarco_trec "The MS MARCO benchmark's deep learning TREC track for passage retrieval in information retrieval [(https://trec.nist.gov)](https://microsoft.github.io/msmarco/).")

[Summarization](https://crfm.stanford.edu/helm/latest/?group=summarization "In text summarization, given a piece of text (paragraph or document), the goal is to produce a much shorter summary.")

-   [CNN/DailyMail](https://crfm.stanford.edu/helm/latest/?group=summarization_cnndm "The CNN/DailyMail benchmark for text summarization ([Hermann et al., 2015](https://papers.nips.cc/paper/2015/hash/afdec7005cc9f14302cd0474fd0f3c96-Abstract.html); [Nallapati et al.,2016](https://aclanthology.org/K16-1028/)).")
-   [XSUM](https://crfm.stanford.edu/helm/latest/?group=summarization_xsum "The XSUM benchmark for text summarization of BBC news articles [(Narayan et al., 2018)](https://aclanthology.org/D18-1206/).")

[Sentiment analysis](https://crfm.stanford.edu/helm/latest/?group=sentiment_analysis "In sentiment classification, given a text (e.g., movie review), the goal is to predict the sentiment (positive or negative).")

-   [IMDB](https://crfm.stanford.edu/helm/latest/?group=imdb "The IMDB benchmark for sentiment analysis in movie review [(Maas et al., 2011)](https://aclanthology.org/P11-1015/).")

[Toxicity detection](https://crfm.stanford.edu/helm/latest/?group=toxicity_detection "In toxicity detection, given a text, the goal is to predict whether the text has toxic content.")

-   [CivilComments](https://crfm.stanford.edu/helm/latest/?group=civil_comments "The CivilComments benchmark for toxicity detection [(Borkan et al., 2019)](https://arxiv.org/pdf/1903.04561.pdf).")

[Text classification](https://crfm.stanford.edu/helm/latest/?group=miscellaneous_text_classification "Text classification is a general format that aims to classify text into a set of categories. This includes a wide range of classification tasks where the input is text.")

-   [RAFT](https://crfm.stanford.edu/helm/latest/?group=raft "The Real-world annotated few-shot (RAFT) meta-benchmark of 11 real-world text classification tasks [(Alex et al., 2021)](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/ca46c1b9512a7a8315fa3c5a946e8265-Abstract-round2.html).")

[Aspirational scenarios](https://crfm.stanford.edu/helm/latest/?group=aspirational "Scenarios that we should support.")

-   [Data-to-text generation](https://crfm.stanford.edu/helm/latest/?group=data_to_text_generation "Currently, we prioritize user-facing tasks in our core scenarios, but don't implement data-to-text generation. Could be implemented via WebNLG, E2E, ToTTo, etc.")
-   [Fact verification](https://crfm.stanford.edu/helm/latest/?group=fact_verification "Currently, we prioritize user-facing tasks in our core scenarios, but don't implement fact verification. Could be implemented via FEVER.")
-   [Copywriting](https://crfm.stanford.edu/helm/latest/?group=copywriting "Currently, we prioritize user-facing tasks in our core scenarios, but don't implement tasks that have not been historically studied in the NLP research community like (ad) copywriting.")
-   [Story generation](https://crfm.stanford.edu/helm/latest/?group=story_generation "Currently, we prioritize user-facing tasks in our core scenarios, but don't implement more creative and interactive tasks like story generation.")
-   [Biomedical scenarios](https://crfm.stanford.edu/helm/latest/?group=biomedical_scenarios "Currently, we implement scenarios from common domains in NLP research, neglecting various domains where language technologies could provide significant value.")
-   [Clinical scenarios](https://crfm.stanford.edu/helm/latest/?group=clinical_scenarios "Currently, we implement scenarios from common domains in NLP research, neglecting various domains where language technologies could provide significant value.")
-   [Financial scenarios](https://crfm.stanford.edu/helm/latest/?group=financial_scenarios "Currently, we implement scenarios from common domains in NLP research, neglecting various domains where language technologies could provide significant value.")
-   [Customer services scenarios](https://crfm.stanford.edu/helm/latest/?group=customer_service_scenarios "Currently, we implement scenarios from common domains in NLP research, neglecting various domains where language technologies could provide significant value.")
-   [Educational scenarios](https://crfm.stanford.edu/helm/latest/?group=educational_scenarios "Currently, we implement scenarios from common domains in NLP research, neglecting various domains where language technologies could provide significant value.")
-   [Very recent scenarios](https://crfm.stanford.edu/helm/latest/?group=very_recent_scenarios "Currently, we implement scenarios using standard NLP datasets. However, to test temporal generalization as the world and language change, we should implement scenarios with very recent data (e.g., current world events) like StreamingQA.")
-   [Scenarios involving historic data](https://crfm.stanford.edu/helm/latest/?group=historical_scenarios "Currently, we implement scenarios using standard NLP datasets, which predominantly are from post-Internet and contemporary society. However, to test temporal generalization for using models in the digital humanities for historic data, we should implement scenarios with significantly older data (e.g., text from 1800s).")
-   [Scenarios involving non-native speakers](https://crfm.stanford.edu/helm/latest/?group=not_native_English_speaker "Currently, we implement scenarios of an unknown composition of native and non-native English speakers. We should implement scenarios to ensure coverage of language from non-native English speakers.")
-   [Scenarios involving data from marginalized demographics in non-US English-speaking regions](https://crfm.stanford.edu/helm/latest/?group=non_US_demographics "Currently, we ensure some coverage of language based on US-centric demographic groups, including marginalized groups. We should implement scenarios to ensure coverage of other socially-relevant groups beyond US demographics (e.g., caste in India).")
-   [Scenarios for languages beyond English.](https://crfm.stanford.edu/helm/latest/?group=non_english "Currently, we only implement English scenarios.")
-   [Scenarios with user-facing tasks on English dialects](https://crfm.stanford.edu/helm/latest/?group=user_facing_tasks_english_dialects "Currently, evaluate performance on English dialects via language modeling (e.g., TwitterAAE, ICE), but it would be good to implement user-facing tasks for these dialects.")

[Language](https://crfm.stanford.edu/helm/latest/?group=language "Targeted evaluation of linguistic capabilities.")

-   [The Pile](https://crfm.stanford.edu/helm/latest/?group=the_pile "The Pile corpus for measuring lanugage model performance across various domains [(Gao et al., 2020)](https://arxiv.org/pdf/2101.00027.pdf).")
-   [TwitterAAE](https://crfm.stanford.edu/helm/latest/?group=twitter_aae "The TwitterAAE corpus of [Blodgett et al. (2016)](https://aclanthology.org/D16-1120/) for measuring language model performance in tweets as a function of speaker dialect.")
-   [ICE](https://crfm.stanford.edu/helm/latest/?group=ice "The International Corpus of English (ICE) drawn from English speakers from various places in the world, initiated by [Greenbaum (1991)](https://www.cambridge.org/core/journals/english-today/article/abs/ice-the-international-corpus-of-english/47808205394C538393C3FD8E62E5E701).")
-   [BLiMP](https://crfm.stanford.edu/helm/latest/?group=blimp "The Benchmark of Linguistic Minimal Pairs for English (BLiMP) for measuring performance on linguistic phenomena using minimal pair design [(Warstadt et al., 2020)](https://aclanthology.org/2020.tacl-1.25/).")

[Knowledge](https://crfm.stanford.edu/helm/latest/?group=knowledge "Targeted evaluation of knowledge (e.g. factual, cultural, commonsense).")

-   [NaturalQuestions (closed-book)](https://crfm.stanford.edu/helm/latest/?group=natural_qa_closedbook "The NaturalQuestions [(Kwiatkowski et al., 2019)](https://aclanthology.org/Q19-1026/) benchmark for question answering based on naturally-occurring queries through Google Search. The input does not include the Wikipedia page with the answer.")
-   [HellaSwag](https://crfm.stanford.edu/helm/latest/?group=hellaswag "The HellaSwag benchmark for commonsense reasoning in question answering [(Zellers et al., 2019)](https://aclanthology.org/P19-1472/).")
-   [OpenbookQA](https://crfm.stanford.edu/helm/latest/?group=openbookqa "The OpenbookQA benchmark for commonsense-intensive open book question answering [(Mihaylov et al., 2018)](https://aclanthology.org/D18-1260/).")
-   [TruthfulQA](https://crfm.stanford.edu/helm/latest/?group=truthful_qa "The TruthfulQA benchmarking for measuring model truthfulness and commonsense knowledge in question answering [(Lin et al., 2022)](https://aclanthology.org/2022.acl-long.229/).")
-   [MMLU](https://crfm.stanford.edu/helm/latest/?group=mmlu "The Massive Multitask Language Understanding (MMLU) benchmark for knowledge-intensive question answering across 57 domains [(Hendrycks et al., 2021)](https://openreview.net/forum?id=d7KBjmI3GmQ).")
-   [WikiFact](https://crfm.stanford.edu/helm/latest/?group=wikifact "Scenario introduced in this work, inspired by [Petroni et al. (2019)](https://aclanthology.org/D19-1250/), to more extensively test factual knowledge.")

[Reasoning](https://crfm.stanford.edu/helm/latest/?group=reasoning "Targeted evaluation of reasoning capabilities (e.g. mathematical, hierarchical).")

-   [Synthetic reasoning (abstract symbols)](https://crfm.stanford.edu/helm/latest/?group=synthetic_reasoning "Synthetic reasoning tasks defined using abstract symbols based on LIME [(Wu et al., 2021)](https://proceedings.mlr.press/v139/wu21c.html).")
-   [Synthetic reasoning (natural language)](https://crfm.stanford.edu/helm/latest/?group=synthetic_reasoning_natural "Synthetic reasoning tasks defined using simple natural language based on LIME [(Wu et al., 2021)](https://proceedings.mlr.press/v139/wu21c.html).")
-   [bAbI](https://crfm.stanford.edu/helm/latest/?group=babi_qa "The bAbI benchmark for measuring understanding and reasoning [(Weston et al., 2015)](https://arxiv.org/pdf/1502.05698.pdf).")
-   [Dyck](https://crfm.stanford.edu/helm/latest/?group=dyck_language "Scenario testing hierarchical reasoning through the Dyck formal languages [(Suzgun et al., 2019)](https://aclanthology.org/W19-3905/).")
-   [GSM8K](https://crfm.stanford.edu/helm/latest/?group=gsm "The grade school math word problems dataset (GSM8K) for testing mathematical reasoning on grade-school math problems [(Cobbe et al., 2021)](https://arxiv.org/pdf/2110.14168.pdf).")
-   [MATH](https://crfm.stanford.edu/helm/latest/?group=math_regular "The MATH benchmark for measuring mathematical problem solving on competition math problems [(Hendrycks et al., 2021)](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html).")
-   [MATH (chain-of-thoughts)](https://crfm.stanford.edu/helm/latest/?group=math_chain_of_thought "The MATH benchmark for measuring mathematical problem solving on competition math problems with chain-of-thoughts style reasoning [(Hendrycks et al., 2021)](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html).")
-   [APPS (Code)](https://crfm.stanford.edu/helm/latest/?group=code_apps "The APPS benchmark for measuring competence on code challenges [(Hendrycks et al., 2021)](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/c24cd76e1ce41366a4bbe8a49b02a028-Abstract-round2.html).")
-   [HumanEval (Code)](https://crfm.stanford.edu/helm/latest/?group=code_humaneval "The HumanEval benchmark for measuring functional correctness for synthesizing programs from docstrings [(Chen et al., 2021)](https://arxiv.org/pdf/2107.03374.pdf).")
-   [LSAT](https://crfm.stanford.edu/helm/latest/?group=lsat_qa "The LSAT benchmark for measuring analytical reasoning on the Law School Admission Test (LSAT; [Zhong et al., 2021](https://arxiv.org/pdf/2104.06598.pdf)).")
-   [LegalSupport](https://crfm.stanford.edu/helm/latest/?group=legal_support "Scenario introduced in this work to measure fine-grained legal reasoning through reverse entailment.")
-   [Data imputation](https://crfm.stanford.edu/helm/latest/?group=entity_data_imputation "Scenario from [Mei et al. (2021)](https://ieeexplore.ieee.org/document/9458712/) that tests the ability to impute missing entities in a data table.")
-   [Entity matching](https://crfm.stanford.edu/helm/latest/?group=entity_matching "Scenario from Magellan [(Konda et al., 2016)](https://dl.acm.org/doi/10.14778/3007263.3007314) that tests the ability to determine if two entities match.")

[Harms](https://crfm.stanford.edu/helm/latest/?group=harms "Targeted evaluation of social harms (e.g., copyright, disinformation, social bias, toxicity).")

-   [Copyright (text)](https://crfm.stanford.edu/helm/latest/?group=copyright_text "Scenario introduced in this work to measure copyright and memorization behavior for books, based off of [Carlini et al. (2021)](https://www.usenix.org/biblio-11958).")
-   [Copyright (code)](https://crfm.stanford.edu/helm/latest/?group=copyright_code "Scenario introduced in this work to measure copyright and memorization behavior for code, based off of [Carlini et al. (2021)](https://www.usenix.org/biblio-11958).")
-   [Disinformation (reiteration)](https://crfm.stanford.edu/helm/latest/?group=disinformation_reiteration "Scenario from [Buchanan et al. (2021)](https://cset.georgetown.edu/publication/truth-lies-and-automation/) that tests the ability to reiterate disinformation content.")
-   [Disinformation (wedging)](https://crfm.stanford.edu/helm/latest/?group=disinformation_wedging "Scenario from [Buchanan et al. (2021)](https://cset.georgetown.edu/publication/truth-lies-and-automation/) that tests the ability to generate divisive and wedging content.")
-   [BBQ](https://crfm.stanford.edu/helm/latest/?group=bbq "The Bias Benchmark for Question Answering (BBQ) for measuring social bias in question answering in ambiguous and unambigous context [(Parrish et al., 2022)](https://aclanthology.org/2022.findings-acl.165/).")
-   [BOLD](https://crfm.stanford.edu/helm/latest/?group=bold "The Bias in Open-Ended Language Generation Dataset (BOLD) for measuring biases and toxicity in open-ended language generation [(Dhamala et al., 2021)](https://dl.acm.org/doi/10.1145/3442188.3445924).")
-   [RealToxicityPrompts](https://crfm.stanford.edu/helm/latest/?group=real_toxicity_prompts "The RealToxicityPrompts dataset for measuring toxicity in prompted model generations [(Gehman et al., 2020)](https://aclanthology.org/2020.findings-emnlp.301/).")

[Efficiency](https://crfm.stanford.edu/helm/latest/?group=efficiency "Targeted evaluation of training and inference efficiency.")

-   [Synthetic efficiency](https://crfm.stanford.edu/helm/latest/?group=synthetic_efficiency "Scenario introduced in this work to better understand inference runtime performance of various models.")

[Calibration](https://crfm.stanford.edu/helm/latest/?group=calibration "Extended calibration metrics.")

-   [MMLU](https://crfm.stanford.edu/helm/latest/?group=mmlu "The Massive Multitask Language Understanding (MMLU) benchmark for knowledge-intensive question answering across 57 domains [(Hendrycks et al., 2021)](https://openreview.net/forum?id=d7KBjmI3GmQ).")
-   [IMDB](https://crfm.stanford.edu/helm/latest/?group=imdb "The IMDB benchmark for sentiment analysis in movie review [(Maas et al., 2011)](https://aclanthology.org/P11-1015/).")
-   [RAFT](https://crfm.stanford.edu/helm/latest/?group=raft "The Real-world annotated few-shot (RAFT) meta-benchmark of 11 real-world text classification tasks [(Alex et al., 2021)](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/ca46c1b9512a7a8315fa3c5a946e8265-Abstract-round2.html).")
-   [CivilComments](https://crfm.stanford.edu/helm/latest/?group=civil_comments "The CivilComments benchmark for toxicity detection [(Borkan et al., 2019)](https://arxiv.org/pdf/1903.04561.pdf).")

[Vary number of in-context examples](https://crfm.stanford.edu/helm/latest/?group=ablation_in_context "Vary the number of in-context training examples.")

-   [NaturalQuestions (open-book)](https://crfm.stanford.edu/helm/latest/?group=natural_qa_openbook_longans "The NaturalQuestions [(Kwiatkowski et al., 2019)](https://aclanthology.org/Q19-1026/) benchmark for question answering based on naturally-occurring queries through Google Search. The input includes the Wikipedia page with the answer.")
-   [CNN/DailyMail](https://crfm.stanford.edu/helm/latest/?group=summarization_cnndm "The CNN/DailyMail benchmark for text summarization ([Hermann et al., 2015](https://papers.nips.cc/paper/2015/hash/afdec7005cc9f14302cd0474fd0f3c96-Abstract.html); [Nallapati et al.,2016](https://aclanthology.org/K16-1028/)).")
-   [IMDB](https://crfm.stanford.edu/helm/latest/?group=imdb "The IMDB benchmark for sentiment analysis in movie review [(Maas et al., 2011)](https://aclanthology.org/P11-1015/).")
-   [CivilComments](https://crfm.stanford.edu/helm/latest/?group=civil_comments "The CivilComments benchmark for toxicity detection [(Borkan et al., 2019)](https://arxiv.org/pdf/1903.04561.pdf).")

[Vary multiple-choice strategy](https://crfm.stanford.edu/helm/latest/?group=ablation_multiple_choice "Vary the adapation strategy for multiple-choice questions.")

-   [HellaSwag](https://crfm.stanford.edu/helm/latest/?group=hellaswag "The HellaSwag benchmark for commonsense reasoning in question answering [(Zellers et al., 2019)](https://aclanthology.org/P19-1472/).")
-   [OpenbookQA](https://crfm.stanford.edu/helm/latest/?group=openbookqa "The OpenbookQA benchmark for commonsense-intensive open book question answering [(Mihaylov et al., 2018)](https://aclanthology.org/D18-1260/).")
-   [TruthfulQA](https://crfm.stanford.edu/helm/latest/?group=truthful_qa "The TruthfulQA benchmarking for measuring model truthfulness and commonsense knowledge in question answering [(Lin et al., 2022)](https://aclanthology.org/2022.acl-long.229/).")
-   [MMLU](https://crfm.stanford.edu/helm/latest/?group=mmlu "The Massive Multitask Language Understanding (MMLU) benchmark for knowledge-intensive question answering across 57 domains [(Hendrycks et al., 2021)](https://openreview.net/forum?id=d7KBjmI3GmQ).")
-   [BLiMP](https://crfm.stanford.edu/helm/latest/?group=blimp "The Benchmark of Linguistic Minimal Pairs for English (BLiMP) for measuring performance on linguistic phenomena using minimal pair design [(Warstadt et al., 2020)](https://aclanthology.org/2020.tacl-1.25/).")
-   [LegalSupport](https://crfm.stanford.edu/helm/latest/?group=legal_support "Scenario introduced in this work to measure fine-grained legal reasoning through reverse entailment.")
-   [LSAT](https://crfm.stanford.edu/helm/latest/?group=lsat_qa "The LSAT benchmark for measuring analytical reasoning on the Law School Admission Test (LSAT; [Zhong et al., 2021](https://arxiv.org/pdf/2104.06598.pdf)).")
-   [BBQ](https://crfm.stanford.edu/helm/latest/?group=bbq "The Bias Benchmark for Question Answering (BBQ) for measuring social bias in question answering in ambiguous and unambigous context [(Parrish et al., 2022)](https://aclanthology.org/2022.findings-acl.165/).")

[Vary prompting](https://crfm.stanford.edu/helm/latest/?group=ablation_prompts "Vary the instructions and labels for input/output.")

-   [NaturalQuestions (open-book)](https://crfm.stanford.edu/helm/latest/?group=natural_qa_openbook_longans "The NaturalQuestions [(Kwiatkowski et al., 2019)](https://aclanthology.org/Q19-1026/) benchmark for question answering based on naturally-occurring queries through Google Search. The input includes the Wikipedia page with the answer.")
-   [CNN/DailyMail](https://crfm.stanford.edu/helm/latest/?group=summarization_cnndm "The CNN/DailyMail benchmark for text summarization ([Hermann et al., 2015](https://papers.nips.cc/paper/2015/hash/afdec7005cc9f14302cd0474fd0f3c96-Abstract.html); [Nallapati et al.,2016](https://aclanthology.org/K16-1028/)).")
-   [IMDB](https://crfm.stanford.edu/helm/latest/?group=imdb "The IMDB benchmark for sentiment analysis in movie review [(Maas et al., 2011)](https://aclanthology.org/P11-1015/).")
-   [CivilComments](https://crfm.stanford.edu/helm/latest/?group=civil_comments "The CivilComments benchmark for toxicity detection [(Borkan et al., 2019)](https://arxiv.org/pdf/1903.04561.pdf).")

[Robustness to contrast sets](https://crfm.stanford.edu/helm/latest/?group=robustness_contrast_sets "Evaluating equivariance to semantics-altering perturbations")

-   [IMDB](https://crfm.stanford.edu/helm/latest/?group=imdb "The IMDB benchmark for sentiment analysis in movie review [(Maas et al., 2011)](https://aclanthology.org/P11-1015/).")
-   [BoolQ](https://crfm.stanford.edu/helm/latest/?group=boolq "The BoolQ benchmark for binary (yes/no) question answering [(Clark et al., 2019)](https://aclanthology.org/N19-1300/).")



