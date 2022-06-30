## **机器学习应用领域**

经济相关：股市、房价等
能源相关：产能预测、分配与合理利用
NLP相关：检索、分类、主题、相似度
互联网用户行为：CTR预测
销量预测：电商、连锁店、超市…
深度学习应用：图像内容理解
推荐系统相关：电商推荐
其他预测：气候、社交网络分析

## **机器学习常用算法**

![1538445320110](../img/1538445320110.png)

SVM-中小型数据集，NB-NLP大型数据集

![1538446047789](../img/1538446047789.png)



## **常用工具**

scikit-learn，NLP的gensim,Natural Language Toolkit

## **建模与问题解决流程**

了解场景和目标
了解评估准则
认识数据：平衡否？
数据预处理(清洗，调权)
特征工程
模型调参
模型状态分析
模型融合

http://blog.csdn.net/han_xiaoyang/article/details/50469334
http://blog.csdn.net/han_xiaoyang/article/details/52910022

**数据清洗**

- 不可信的样本丢掉
- 缺省值极多的字段考虑不用

**数据采样**

- 下/上采样
- 保证样本均衡

**工具**

- hive sql/spark sql
- pandas：数据量少的时候

### **特征工程**

![1538447293961](../img/1538447293961.png)

**特征处理**

数值型
类别型
时间类：可以变成间隔型；或者组合型，如一周内登录网页多少次；饭点，非饭点、工作日，非工作日
文本型：n-gram，bag of words，TF-IDF
统计型：相对值
组合特征

参考
课程提供特征工程PDF
http://scikit-learn.org/stable/modules/preprocessing.html
http://scikit-learn.org/stable/modules/classes.html#modulesklearn.feature_extraction

特征选择
http://scikit-learn.org/stable/modules/feature_selection.html
过滤型：用得少
sklearn.feature_selection.SelectKBest

包裹型
sklearn.feature_selection.RFE

嵌入型
feature_selection.SelectFromModel
Linear model，L1正则化

### 模型选择

sklearn cheetsheet提供的候选
课程案例经验
交叉验证（cross validation）

- K折交叉验证（K-fold cross validation）
- http://scikit-learn.org/stable/modules/cross_validation.html

**模型参数选择**

交叉验证（cross validation）
http://scikit-learn.org/stable/modules/grid_search.html
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

### 模型状态评估

模型状态

- 过拟合(overfitting/high variance)
- 欠拟合(underfitting/high bias)

Learning curve:学习曲线

plot learning curve:绘制学习曲线
https://www.zybuluo.com/hanxiaoyang/note/545131

### 模型融合
简单说来，我们信奉几条信条

群众的力量是伟大的，集体智慧是惊人的

- Bagging：http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
- 随机森林/Random forest

站在巨人的肩膀上，能看得更远

- 模型stacking：用多种predictor结果作为特征训练

一万小时定律

- Adaboost：http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble
- 逐步增强树/Gradient Boosting Tree



## **kaggle wiki**

看

## 经济金融相关问题

解决高维数据分类/回归问题

- 线性/非线性回归方程：如预测股票收盘价等，可以用前10天的数据预测当天的

- 决策树
  - 优势：非黑盒；轻松去除无关attribute(Gain=0)；Test起来很快(O(depth))
  - 劣势：只能线性分隔数据；贪婪算法(可能找不到最好的树)

- 弱分类器的进化-Ensemble：Bagging，Random Forest，Boosting

- 神经网络

kaggle竞赛：房价预测

https://www.kaggle.com/c/house-prices-advanced-regression-techniques



非标准数据的处理

- 文本处理：记录每个单词的出现次数，记录每个单词的出现频率，用语义网络表示

- 图片处理：用RGB点阵表示

- 视频处理：
  - 音轨：声波；语音识别：文本
  - 视频：一堆图片：图片；图片识别：文本

## 排序与CTR预估问题

### Online advertising and click through rate prediction

**Types of online advertising**

**Retargeting** – Using cookies, track if a user left a webpage without making a purchase and retarget the user with ads from that site
**Behavioral targeting** – Data related to user’s online activity is collected from multiple websites, thus creating a detailed picture of the user’s interests to deliver more targeted advertising
**Contextual advertising** – Display ads related to the content of the webpage

广告主，平台方，用户方

CPM (Cost-Per-Mille): is an inventory based pricing model. Is when the price is based on 1,000
impressions. 按照曝光
CPC (Cost-Per-Click): Is a performance-based metric.This means the Publisher only gets paid when (and if)
a user clicks on an ad 按照点击
CPA (Cost Per Action): Best deal of all for Advertisers in terms of risk because they only pay for media when it results in a sale 

Click-through rate (CTR)

Ratio of users who click on an ad to the number of total users who view the ad

CTR=Clicks / Impressions * 100%

Today, typical click through rate is less than 1%

In a pay-per-click setting, revenue can be maximized by choosing to display ads that have the maximum CTR, hence the problem of CTR Prediction.

**Predict CTR – a Scalable Machine Learning success story**

Predict conditional probability that the ad will be clicked by the user given the predictive features of ads
Predictive features are:
– Ad’s historical performance
– Advertiser and ad content info
– Publisher info
– User Info (eg: search/ click history)
Data set is high dimensional, sparse and skewed:
– Hundreds of millions of online users
– Millions of unique publisher pages to display ads
– Millions of unique ads to display
– Very few ads get clicked by users

### Data set and features
Sample of the dataset used for the Display Advertising Challenge hosted by Kaggle:
https://www.kaggle.com/c/criteo-display-ad-challenge/

链接: https://pan.baidu.com/s/1qYVhaJq 密码: 8fyn

### Spark MLlib and the Pipeline API



### MLlib pipeline for Click Through Rate Prediction
Step 1 - Parse Data into Spark SQL DataFrames

Step 2 – Feature Transformer using StringIndexer类别型转成0,1,2,3,...这种

Step 3 - Feature Transformer using One Hot Encoding

Step 4 – Feature Selector using Vector Assembler将特征列表转化成一个向量

Step 5 – Train a model using Estimator Logistic Regression

Apply the pipeline to make predictions

### Random Forest/GBDT/FM/FFM/DNN

Some alternatives:

Use Hashed features instead of OHE
Use Log loss evaluation or ROC to evaluate Logistic Regression
Perform feature selection
Use Naïve Bayes or other binary classification algorithms

Random Forest
GBDT
FM（factorization machine）
请参见比赛
https://www.kaggle.com/c/avazu-ctrprediction
Rank 2nd Owen Zhang的解法：
https://github.com/owenzhang/kaggle-avazu



**要用格点搜索，交叉验证找到最好的参数**

**LibFM，LibMF，SVDfeature**



FFM（field-aware factorization machine）

工业界数据与代码
数据可以在百度云下载到
链接: https://pan.baidu.com/s/1qYRM2cs 密码: twa7

Google Wide && Deep model
说明：
https://www.tensorflow.org/versions/r0.10/tutorials/wide_and_deep/index.html
代码：
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/wide_n_deep_tutorial.py

FNN
说明：
http://arxiv.org/pdf/1601.02376.pdf
代码：
https://github.com/wnzhang/deep-ctr

About Data
需要大规模数据做实验/学习的同学，可以在Cretio实验数据下载1TB的CTR预估所需数据。
http://labs.criteo.com/downloads/download-terabyte-click-logs

## 自然语言处理类问题

### NLP的基本思路与技法

**NLTK**

http://www.nltk.org/
NLTK是Python上著名的⾃然语⾔处理库

⾃带语料库，词性分类库
⾃带分类，分词，等等功能
强⼤的社区⽀持
还有N多的简单版wrapper

![1538620835230](../img/1538620835230.png)

**文本处理流程**

Preprocess——Tokenize——Make Featurees——ML

**Tokenize**：把长句⼦拆成有“意义”的⼩部件

英文：

```
import nltk
sentence = “hello, world"
tokens = nltk.word_tokenize(sentence)
```

中文：

启发式Heuristic

机器学习/统计方法：HMM，CRF

jieba，讯飞的，斯坦福corenlp



社交⽹络语⾔的tokenize，如@某⼈, 表情符号, URL, `#话题符号`，难以分隔

```
from nltk.tokenize import word_tokenize
tweet = 'RT @angelababy: love you baby! :D http://ah.love #168cm'
print(word_tokenize(tweet))
# ['RT', '@', 'angelababy', ':', 'love', 'you', 'baby', '!', ':',
# ’D', 'http', ':', '//ah.love', '#', '168cm']
```

```
import re
emoticons_str = r"""
(?:
[:=;] # 眼睛
[oO\-]? # ⿐子
[D\)\]\(\]/\\OpP] # 嘴
)"""
regex_str = [
emoticons_str,
r'<[^>]+>', # HTML tags
r'(?:@[\w_]+)', # @某⼈人
r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # 话题标签
r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
# URLs
r'(?:(?:\d+,?)+(?:\.?\d+)?)', # 数字
r"(?:[a-z][a-z'\-_]+[a-z])", # 含有 - 和 ‘ 的单词
r'(?:[\w_]+)', # 其他
r'(?:\S)' # 其他
]
```

正则表达式对照表
http://www.regexlab.com/zh/regref.htm

```
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
def tokenize(s):
return tokens_re.findall(s)
def preprocess(s, lowercase=False):
tokens = tokenize(s)
if lowercase:
tokens = [token if emoticon_re.search(token) else token.lower() for token in
tokens]
return tokens
tweet = 'RT @angelababy: love you baby! :D http://ah.love #168cm'
print(preprocess(tweet))
# ['RT', '@angelababy', ':', 'love', 'you', 'baby',
# ’!', ':D', 'http://ah.love', '#168cm']
```



纷繁复杂的词形

Inflection变化: walk => walking => walked
不影响词性

derivation 引申: nation (noun) => national (adjective) => nationalize (verb)
影响词性

词形归⼀化

Stemming 词⼲提取：⼀般来说，就是把不影响词性的inflection的⼩尾巴砍掉
walking 砍ing = walk
walked 砍ed = walk

Lemmatization 词形归⼀：把各种类型的词的变形，都归为⼀个形式
went 归⼀ = go
are 归⼀ = be

```
>>> from nltk.stem.porter import PorterStemmer
>>> porter_stemmer = PorterStemmer()
>>> porter_stemmer.stem(‘maximum’)
u’maximum’
>>> porter_stemmer.stem(‘presumably’)
u’presum’
>>> porter_stemmer.stem(‘multiply’)
u’multipli’
>>> porter_stemmer.stem(‘provision’)
u’provis’
>>> from nltk.stem import SnowballStemmer
>>> snowball_stemmer = SnowballStemmer(“english”)
>>> snowball_stemmer.stem(‘maximum’)
u’maximum’
>>> snowball_stemmer.stem(‘presumably’)
u’presum’
>>> from nltk.stem.lancaster import LancasterStemmer
>>> lancaster_stemmer = LancasterStemmer()
>>> lancaster_stemmer.stem(‘maximum’)
‘maxim’
>>> lancaster_stemmer.stem(‘presumably’)
‘presum’
>>> lancaster_stemmer.stem(‘presumably’)
‘presum’
>>> from nltk.stem.porter import PorterStemmer
>>> p = PorterStemmer()
>>> p.stem('went')
'went'
>>> p.stem('wenting')
'went'
```

```
>>> from nltk.stem import WordNetLemmatizer
>>> wordnet_lemmatizer = WordNetLemmatizer()
>>> wordnet_lemmatizer.lemmatize(‘dogs’)
u’dog’
>>> wordnet_lemmatizer.lemmatize(‘churches’)
u’church’
>>> wordnet_lemmatizer.lemmatize(‘aardwolves’)
u’aardwolf’
>>> wordnet_lemmatizer.lemmatize(‘abaci’)
u’abacus’
>>> wordnet_lemmatizer.lemmatize(‘hardrock’)
‘hardrock’
```

Lemma的⼩问题

Went
v. go的过去式
n. 英⽂名：温特

NLTK更好地实现Lemma

```
# ⽊木有POS Tag，默认是NN 名词
>>> wordnet_lemmatizer.lemmatize(‘are’)
‘are’
>>> wordnet_lemmatizer.lemmatize(‘is’)
‘is’
# 加上POS Tag
>>> wordnet_lemmatizer.lemmatize(‘is’, pos=’v’)
u’be’
>>> wordnet_lemmatizer.lemmatize(‘are’, pos=’v’)
u’be’
```

![1538621891813](../img/1538621891813.png)

```
>>> import nltk
>>> text = nltk.word_tokenize('what does the fox say')
>>> text
['what', 'does', 'the', 'fox', 'say']
>>> nltk.pos_tag(text)
[('what', 'WDT'), ('does', 'VBZ'), ('the', 'DT'), ('fox', 'NNS'), ('say', 'VBP')]
```

Stopwords

⼀千个HE有⼀千种指代，⼀千个THE有⼀千种指事
对于注重理解⽂本『意思』的应⽤场景来说，歧义太多
全体stopwords列表 http://www.ranks.nl/stopwords

NLTK去除stopwords
⾸先记得在console⾥⾯下载⼀下词库
或者 nltk.download(‘stopwords’)

```
from nltk.corpus import stopwords
# 先token⼀一把，得到⼀一个word_list
# ...
# 然后filter⼀一把
filtered_words =
[word for word in word_list if word not in stopwords.words('english')]
```

⼀条typical的⽂本预处理流⽔线

Raw_Text——Tokenize(——POS Tag)——Lemma/Stemming——stopwords——Word_List

原始文本得到有意义的单词列表

**NLTK在NLP上的经典应⽤**

**情感分析**

最简单的 sentiment dictionary，类似于关键词打分机制

⽐如：AFINN-111
http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010

```
sentiment_dictionary = {}
for line in open('data/AFINN-111.txt')
word, score = line.split('\t')
sentiment_dictionary[word] = int(score)
# 把这个打分表记录在一个Dict上以后
# 跑⼀一遍整个句子，把对应的值相加
total_score = sum(sentiment_dictionary.get(word, 0) for word in words)
# 有值就是Dict中的值，没有就是0
# 于是你就得到了一个 sentiment score
```

配上ML的情感分析

```
from nltk.classify import NaiveBayesClassifier
# 随手造点训练集
s1 = 'this is a good book'
s2 = 'this is a awesome book'
s3 = 'this is a bad book'
s4 = 'this is a terrible book'
def preprocess(s):
# Func: 句子处理理
# 这里简单的用了split(), 把句子中每个单词分开
# 显然 还有更多的processing method可以用
return {word: True for word in s.lower().split()}
# return⻓长这样:
# {'this': True, 'is':True, 'a':True, 'good':True, 'book':True}
# 其中, 前⼀一个叫fname, 对应每个出现的文本单词;
# 后⼀一个叫fval, 指的是每个文本单词对应的值。
# 这里我们用最简单的True,来表示,这个词『出现在当前的句子中』的意义。
# 当然啦, 我们以后可以升级这个方程, 让它带有更加牛逼的fval, 比如 word2vec

# 把训练集给做成标准形式
training_data = [[preprocess(s1), 'pos'],
[preprocess(s2), 'pos'],
[preprocess(s3), 'neg'],
[preprocess(s4), 'neg']]
# 喂给model吃
model = NaiveBayesClassifier.train(training_data)
# 打出结果
print(model.classify(preprocess('this is a good book')))
```



**文本相似度**

⽤元素频率表⽰⽂本特征

余弦定理

Frequency 频率统计

```
import nltk
from nltk import FreqDist
# 做个词库先
corpus = 'this is my sentence ' \
'this is my life ' \
'this is the day'

# 随便便tokenize⼀一下
# 显然, 正如上文提到,
# 这里可以根据需要做任何的preprocessing:
# stopwords, lemma, stemming, etc.
tokens = nltk.word_tokenize(corpus)
print(tokens)
# 得到token好的word list
# ['this', 'is', 'my', 'sentence',
# 'this', 'is', 'my', 'life', 'this',
# 'is', 'the', 'day']

# 借用NLTK的FreqDist统计一下文字出现的频率
fdist = FreqDist(tokens)
# 它就类似于一个Dict
# 带上某个单词, 可以看到它在整个文章中出现的次数
print(fdist['is'])
# 3

# 好, 此刻, 我们可以把最常用的50个单词拿出来
standard_freq_vector = fdist.most_common(50)
size = len(standard_freq_vector)
print(standard_freq_vector)
# [('is', 3), ('this', 3), ('my', 2),
# ('the', 1), ('day', 1), ('sentence', 1),
# ('life', 1)

# Func: 按照出现频率大小, 记录下每一个单词的位置
def position_lookup(v):
    res = {}
    counter = 0
    for word in v:
    	res[word[0]] = counter
    	counter += 1
    return res
# 把标准的单词位置记录下来
standard_position_dict = position_lookup(standard_freq_vector)
print(standard_position_dict)
# 得到一个位置对照表
# {'is': 0, 'the': 3, 'day': 4, 'this': 1,
# 'sentence': 5, 'my': 2, 'life': 6}

# 这时, 如果我们有个新句子:
sentence = 'this is cool'
# 先新建一个跟我们的标准vector同样大小的向量
freq_vector = [0] * size
# 简单的Preprocessing
tokens = nltk.word_tokenize(sentence)
# 对于这个新句子里的每一个单词
for word in tokens:
    try:
        # 如果在我们的词库里出现过
        # 那么就在"标准位置"上+1
    	freq_vector[standard_position_dict[word]] += 1
    except KeyError:
        # 如果是个新词
        # 就pass掉
        continue
print(freq_vector)
# [1, 1, 0, 0, 0, 0, 0]
# 第一个位置代表 is, 出现了一次
# 第二个位置代表 this, 出现了一次
# 后面都木有
```



**文本分类**

TF-IDF

TF: Term Frequency, 衡量⼀个term在⽂档中出现得有多频繁。
TF(t) = (t出现在⽂档中的次数) / (⽂档中的term总数).
IDF: Inverse Document Frequency, 衡量⼀个term有多重要。
有些词出现的很多，但是明显不是很有卵⽤。⽐如’is'，’the‘，’and‘之类的。
为了平衡，我们把罕见的词的重要性（weight）搞⾼，把常见词的重要性搞低。
IDF(t) = log_e(⽂档总数 / 含有t的⽂档总数).
TF-IDF = TF * IDF

```
from nltk.text import TextCollection
# 首先, 把所有的文档放到TextCollection类中。
# 这个类会自动帮你断句, 做统计, 做计算
corpus = TextCollection(['this is sentence one',
'this is sentence two',
'this is sentence three'])
# 直接就能算出tfidf
# (term: 一句话中的某个term, text: 这句话)
print(corpus.tf_idf('this', 'this is sentence four'))
# 0.444342
# 同理, 怎么得到一个标准大小的vector来表示所有的句子?
# 对于每个新句子
new_sentence = 'this is sentence five'
# 遍历一遍所有的vocabulary中的词:
for word in standard_vocab:
	print(corpus.tf_idf(word, new_sentence))
	# 我们会得到一个巨长(=所有vocab长度)的向量
```

接下来ML

可能的ML模型：SVM，LR，RF，MLP，LSTM，RNN，...

## 能源资源相关问题

