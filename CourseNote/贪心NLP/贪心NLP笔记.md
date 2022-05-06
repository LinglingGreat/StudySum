## NLP定义以及歧义性

**NLP是什么？**NLP=NLU+NLG

- NLU:语音/文本——>意思(meaning)

- NLG:意思——>文本/语音

**NLP为什么难？**

图片：看到什么就是什么，而文本还需要去理解内在的含义。

- 一个意思可以有多种表达方式

- 一词多义(Ambiguity)，解决方法：每种含义有一种概率（利用历史统计数据，利用上下文）

## 案例：机器翻译

**问题：如果现在让你写一个机器翻译系统，怎么实现？**

方法：根据语料做一个统计模型（大量双语语料中，哪两个单词总是一起出现，就很有可能是翻译），每个词对应的翻译是哪个。有什么缺点？

慢，没解决语义，上下文没考虑，语法不对，需要大量的规则统计

假设中文翻译成英文。

中文分词，根据词典把每个词翻译成英文(得到Broken English，可能不通顺)

那怎么让它通顺呢？最笨的方法就是对所有的词做排列组合，在所有可能的选择里选出最合适的句子。那怎么选出最合适的呢？假设有一个模型LM，输入句子，它就输出这个句子的概率，这样就能选出概率最高的句子作为最合适的。LM是语言模型。

缺点？计算量大

可不可以把翻译模型和语言模型两个步骤做成一个步骤？同时考虑而不是分开考虑。Decoding Model。维特比算法。

$max_{en}p(en|cn)$

根据贝叶斯定理，上式等价于$max_{en}p(cn|en)p(en)$，两个部分分别叫做翻译模型和语言模型。

维特比算法可以同时考虑2个部分。

语言模型（Language Model）

- 给定一句句英文e, 计算概率（e)
- 如果是符合英文语法的，p(e)会高
- 如果是随机语句，p(e)会低

翻译模型

- 给定一对<c,e>，计算p(c|e)
- 语义相似度高, 则p(c|e)高
- 语义相似度低，则p(c|e)低

Decoding Algorithm

- 给定语言模型，翻译模型和f, 找出最优的使得p(e)p(f|e)最大

语言模型

![image-20211129220847851](img/image-20211129220847851.png)

## NLP的关键技术

自然语言处理技术四个维度

- Semantic(语义)
- Syntax(句子结构)：句法分析，依存分析，
- Morphology(单词)：分词，词性，NER
- Phonetics(声音)

https://www.quora.com/What-are-the-major-open-problems-in-natural-language-understanding

## 复杂度

![image-20211129231756243](img/image-20211129231756243.png)

![image-20211129232231218](img/image-20211129232231218.png)

取大的



斐波那契数的时间复杂度

T(n) = T(n-1)+T(n-2)

不能用主定理，画一棵树（递归树）

或者找规律，用数学归纳法证明



空间复杂度

O(N)，堆栈分析

![image-20211130232151199](img/image-20211130232151199.png)



指数级复杂度p^n：不可以解决的问题（NP hard/NP complete，NP complete是hard的子集）

- 方法一：对于小型的问题(n小)，可仍然采用
- 方法二：近似算法，不保证获得精确解
  - 提出近似算法
  - 指出时间复杂度
  - 给出近似算法最后给出的解离我们想要的最优解有多远
- 方法三：量子计算机

多项式复杂度n^p：可以解决的问题（P问题）

NP：可以多项式复杂度内能verify（给出一个解，能够判断是不是想要的）的问题，不考虑问题能否解决

P是NP的子集

## 问答系统

基于搜索的问题系统

1.分词

2.预处理：spell correction, stem/lemma, stop-words, word filtering, 同义词

3.文本的表示：bool vector, count vector, tfidf, word2vec, seq2seq

4.计算相似度：欧几里得，cosine，jaccard。倒排索引

5.排序，过滤，返回结果

## 文本预处理

![image-20211201221715519](img/image-20211201221715519.png)

### 分词

分词：最大匹配

![image-20211201222636951](img/image-20211201222636951.png)

![image-20211201223451449](img/image-20211201223451449.png)

![image-20211201224055159](img/image-20211201224055159.png)



考虑语义

![image-20211212132607382](img/image-20211212132607382.png)

![image-20211212133610238](img/image-20211212133610238.png)

![image-20211212133624620](img/image-20211212133624620.png)

2步，怎么解决效率问题？

维特比算法（核心：动态规划）

![image-20211212174700025](img/image-20211212174700025.png)

![image-20211212180229611](img/image-20211212180229611.png)

### 拼写纠错

语言模型，困惑度，语言模型的平滑

博客：https://norvig.com/spell-correct.html

```python
# 词典库
vocab = set([line.rstrip() for line in open('vocab.txt')])

# 需要生成所有候选集合
def generate_candidates(word):
    """
    word: 给定的输入（错误的输入） 
    返回所有(valid)候选集合
    """
    # 生成编辑距离为1的单词
    # 1.insert 2. delete 3. replace
    # appl: replace: bppl, cppl, aapl, abpl... 
    #       insert: bappl, cappl, abppl, acppl....
    #       delete: ppl, apl, app
    
    # 假设使用26个字符
    letters = 'abcdefghijklmnopqrstuvwxyz' 
    
    splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
    # insert操作
    inserts = [L+c+R for L, R in splits for c in letters]
    # delete
    deletes = [L+R[1:] for L,R in splits if R]
    # replace
    replaces = [L+c+R[1:] for L,R in splits if R for c in letters]
    
    candidates = set(inserts+deletes+replaces)
    
    # 过来掉不存在于词典库里面的单词
    return [word for word in candidates if word in vocab] 
    
generate_candidates("apple")

from nltk.corpus import reuters

# 读取语料库
categories = reuters.categories()
corpus = reuters.sents(categories=categories)

# 构建语言模型: bigram
term_count = {}
bigram_count = {}
for doc in corpus:
    doc = ['<s>'] + doc
    for i in range(0, len(doc)-1):
        # bigram: [i,i+1]
        term = doc[i]
        bigram = doc[i:i+2]
        
        if term in term_count:
            term_count[term]+=1
        else:
            term_count[term]=1
        bigram = ' '.join(bigram)
        if bigram in bigram_count:
            bigram_count[bigram]+=1
        else:
            bigram_count[bigram]=1

# sklearn里面有现成的包

# 用户打错的概率统计 - channel probability
channel_prob = {}

for line in open('spell-errors.txt'):
    items = line.split(":")
    correct = items[0].strip()
    mistakes = [item.strip() for item in items[1].strip().split(",")]
    channel_prob[correct] = {}
    for mis in mistakes:
        channel_prob[correct][mis] = 1.0/len(mistakes)

print(channel_prob)
```



## 其它

逻辑回归：http://cseweb.ucsd.edu/~elkan/250B/logreg.pdf

ElasticNet，凸优化原理：https://web.stanford.edu/~hastie/Papers/B67.2%20(2005)%20301-320%20Zou%20&%20Hastie.pdf

MLE与MAP https://zhuanlan.zhihu.com/p/37215276
高斯分布与L2正则 http://qwone.com/~jason/writing/l2gaussian.pdf

凸优化原理https://web.stanford.edu/class/ee364a/lectures.html

非凸优化问题，梯度下降法收敛分析

Set Cover Problem  https://en.wikipedia.org/wiki/Set_cover_problem
LP Relaxation   http://www.cs.cmu.edu/~anupamg/adv-approx/lecture2.pdf
Convergence Analaysis   https://www.cs.cmu.edu/~ggordon/10725-F12/slides/05-gd-revisited.pdf

SVM

SVM Intro   http://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf
CMU's SVM   https://www.cs.cmu.edu/~cga/ai-course/svm.pdf
Nice Paper on SVM  https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/svmtutorial.pdf

CRFhttp://cseweb.ucsd.edu/~elkan/250Bfall2007/loglinear.pdf

word2vechttps://arxiv.org/pdf/1301.3781.pdf

词向量https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb

VAEhttps://arxiv.org/abs/1312.6114





