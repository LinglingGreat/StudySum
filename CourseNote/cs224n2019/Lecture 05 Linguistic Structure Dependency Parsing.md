### Constituency Parsing

**part of speech=pos 词性**

- `Det` 指的是 **Determiner**，在语言学中的含义为 **限定词**
- `NP` 指的是 **Noun Phrase** ，在语言学中的含义为 **名词短语**
- `VP` 指的是 **Verb Phrase** ，在语言学中的含义为 **动词短语**
- `P` 指的是 **Preposition** ，在语言学中的含义为 **介词**
  - `PP` 指的是 **Prepositional Phrase** ，在语言学中的含义为 **介词短语**
- NP→Det NNP→Det N
- NP→Det (Adj) NNP→Det (Adj) N
- NP→Det (Adj) N PPNP→Det (Adj) N PP
  - PP→P NPPP→P NP
- VP→V PPVP→V PP
  - 中文中，介词短语会出现在动词之前
- Example : The cat by the large crate on the large table by the door

### Dependency Parsing

不是使用各种类型的短语，而是直接通过单词与其他的单词关系表示句子的结构，显示哪些单词依赖于(修饰或是其参数)哪些其他单词

例如：Look in the large crate in the kitchen by the door.

look是整个句子的根源，look依赖于crate(或者说crate是look的依赖)

- in, the, large都是crate的依赖
- in the kitchen是crate的修饰
- in, the都是kitchen的依赖
- by the door是crate的依赖

**介词短语依附歧义**：San Jose cops kill man with knife

面对复杂的句子结构，我们需要考虑 **指数级** 的可能结构，这个序列被称为 **Catalan numbers**

**Catalan numbers** : $C_n=(2n)!/[(n+1)!n!]$

- 一个指数增长的序列，出现在许多类似树的环境中
  - 例如，一个 n+2 边的多边形可能的三角剖分的数量
    - 出现在概率图形模型的三角剖分中(CS228)

**协调范围模糊**

例句：Shuttle veteran and longtime NASA executive Fred Gregory appointed to board

- 一个人：[[Shuttle veteran and longtime NASA executive] Fred Gregory] appointed to board
- 两个人：[Shuttle veteran] and [longtime NASA executive Fred Gregory] appointed to board

**形容词修饰语歧义**

Students get first hand job experience

- first hand表示 第一手的，直接的，即学生获得了直接的工作经验
  - `first` 是 `hand` 的形容词修饰语(amod)
- first 修饰 experience, hand 修饰 job 

**动词短语(VP)依存歧义**

例句：Mutilated body washes up on Rio beach to be used for Olympic beach volleyball.

- `to be used for Olympic beach volleyball` 是 动词短语 (VP)
- 修饰的是 `body` 还是 `beach`

**依赖路径识别语义关系**



### Dependency Grammar and Dependency Structure

Dependency Structure有两种表现形式

- 一种是直接在句子上标出依存关系箭头及语法关系
- 另一种是将其做成树状机构（Dependency Tree Graph）

### Greedy transition-based parsing

- 贪婪判别依赖解析器 **greedy discriminative dependency parser** 的一种简单形式
- 解析器执行一系列自底向上的操作
  - 大致类似于[shift-reduce解析器](https://en.wikipedia.org/wiki/Shift-reduce_parser)中的“shift”或“reduce”，但“reduce”操作专门用于创建头在左或右的依赖项

### Why train a neural dependency parser? Indicator Features Revisited

Indicator Features的问题

- 稀疏
- 不完整
- 计算复杂
  - 超过95%的解析时间都用于特征计算





### 参考资料

[https://looperxx.github.io/CS224n-2019-05-Linguistic%20Structure%20Dependency%20Parsing/](https://looperxx.github.io/CS224n-2019-05-Linguistic Structure Dependency Parsing/)