### Background

Structural vs. Reduced-Form Models

![image](img/researchmodel.png)



https://economics.stackexchange.com/questions/180/what-is-structural-estimation-compared-to-reduced-form-estimation



#### Reduced-form models:

目的是通过已有数据建立消费者或公司的历史决策模型，再将其用于预测这些人的未来行为。

但是，如果策略变化导致的影响与历史数据相差太大，或者如果这样的变化会导致决策者的决策方式发生改变（即决策模型的参数甚至函数形势发生变化），这样的分析可能会导致错误的预测。

#### Structural Models:

捕获数据反映的关系的潜在经济过程。

structral参数经常是从用户行为的"最优性"经济假设和市场均衡模型中推导而来的。

可以用于预测无法观测到的经济参数（例如边际成本，规模效应，需求的价格弹性）。

可以用于反事实模拟或政策模拟（要求structural model不会随着经济环境中的预期变化(contemplated change)而变化）。

可以用于比较两种竞争理论的预期表现。

##### Choice Model

Discrete Choice Model离散选择模型是大多数structural models的基本模型之一。

它为每个个体（消费者、投资者、公司等）简历内在选择行为模型。

可以用于理解和预测决策者面临选择集的一个离散选择。

![image](img/CDP.png)

Choice Decision Process的元素：

###### Decision maker

不同的决策者根据他们所处的环境不同可能有不同的选择集(choice sets)

不同的决策者有不同的偏好

###### Alternatives (the Choice Set)

choice set的三个特点：

–The alternatives must be **mutually  exclusive** (from the decision maker’s perspective). 

方案必须是互斥的

–The choice set must be **exhaustive** (all possible alternatives should be included).

选择集必须是穷举的（可以用other表示剩余方案）

–The number of alternatives must be **finite**.

可选择的是有限的

选择过程中的可选择方案被一个属性值集刻画，包括一般属性和方案的特殊属性。

###### Decision rule

Decision rule is a mechanism to process information andevaluate alternatives.
An individual invokes a decision rule to select an alternativefrom a choice set with two or more alternatives.
There are a variety of decision rules:
–**Dominance**: An alternative is dominant with respect to another if it is dominant forat least one attribute and no worse for all other attributes.
–**Satisfaction**: An alternative can be eliminated if it does not meet the “satisfactioncriterion (defined by decision maker) of at least one attribute.
–**Lexicographic**: Attributes are rank ordered by their level of “importance”. Thealternative that is the most attractive for the most important attribute is chosen.
*Utility maximization* is one of the widely used methods.



Utility-Based Choice Theory

U(X~i~, S~n~) ≥ U(X~j~, S~n~) 对于任意j  → i > j 任意j ∈ C

U是效用函数，X~i~ 是描述可选方案i和j的属性向量，S~n~ 是描述个体n的特征向量

两个广泛使用的离散选择模型：The logit model and the probit model

### Logit Model

思想：有多个方案，每个方案的效用U = V + ε，V=Xβ，其中ε之间是i.i.d的，且服从双指数分布。

利用极大似然估计，最大化观测数据中方案被选中的概率P(P^D的连乘，其中D表明方案是否被选中)(概率P即该方案的效用大于其它方案效用的概率，利用ε的分布可求得)。

![image](img/logit1.jpg)



很多Logit模型的缺陷都来源于i.i.d的假设以及双指数分布的假设

![image](img/logit2.jpg)



在J个方案中选择i的概率 / 在给定J个方案中i的效用是最大的概率

![image](img/logit3.jpg)



![image](img/logit4.jpg)



![image](img/logit5.jpg)



Estimation

-maximum likelihood estimation(MLE)

![image](img/logit6.jpg)



-where, D~nj~ = 1 if j is selected, 0 otherwise

#### Model Identification

无法识别，如Y = AX，A不是满秩矩阵时，方程的解不唯一，即不可识别。

![image](img/logit7.jpg)



Model Interpretation & Effects

![image](img/logit8.jpg)



产品的属性改变导致销量的变化。“自己的弹性”，“交叉弹性”

![image](img/logit9.jpg)



方案i与j的市场占有率之比只与回归系数、方案个体特性差异有关（不太合理）

#### IIA Assumption

![image](img/logit10.jpg)



![image](img/logit11.jpg)



![image](img/logit12.jpg)



#### 优缺点

![image](img/logit13.jpg)



### Probit Model

允许ε之间存在相关性，协方差矩阵。

思想：有多个方案，每个方案的效用U = V + ε，V=Xβ，其中ε之间是i.i.d的，且服从联合正态分布。

概率P即该方案的效用大于其它方案效用的概率，利用ε的分布可求得。

(X~2~ - X~1~ )β ,其它方案与基准方案之差 。利用simulation求系数。

![image](img/probit1.jpg)



![image](img/probit2.jpg)



![image](img/probit3.jpg)



![image](img/probit4.jpg)



![image](img/probit5.jpg)



![image](img/probit6.jpg)



其它方案与方案1的效用差异

![image](img/probit7.jpg)



![image](img/probit8.jpg)



![image](img/Probit9.jpg)



![image](img/probit10.jpg)



### 宽数据与长数据

**wide**: in this case, there is one row for each choice situation

| mode    | price.beach | price.pier | price.boat | price.charter | catch.beach | catch.pier | catch.boat | catch.charter | income   |
| ------- | ----------- | ---------- | ---------- | ------------- | ----------- | ---------- | ---------- | ------------- | -------- |
| charter | 157.93      | 157.93     | 157.93     | 182.93        | 0.0678      | 0.0503     | 0.2601     | 0.5391        | 7083.332 |
| charter | 15.114      | 15.114     | 10.534     | 34.534        | 0.1049      | 0.0451     | 0.1574     | 0.4671        | 1250     |
| boat    | 161.874     | 161.874    | 24.334     | 59.334        | 0.5333      | 0.4522     | 0.2413     | 1.0266        | 3750     |

**long**: in this case, there is one row for each alternative and,
therefore, as many rows as there are alternatives for each
choice situation.

| individual | mode  | choice | wait | vcost | travel | gcost | income | size |
| ---------- | ----- | ------ | ---- | ----- | ------ | ----- | ------ | ---- |
| 1          | air   | no     | 69   | 59    | 100    | 70    | 35     | 1    |
| 1          | train | no     | 34   | 31    | 372    | 71    | 35     | 1    |
| 1          | bus   | no     | 35   | 25    | 417    | 70    | 35     | 1    |
| 1          | car   | yes    | 0    | 10    | 180    | 30    | 35     | 1    |

两个模型估计的系数差别不大，但是预测结果可能不一样。

补充资料：Guadagni and Little’s Paper

### 补充资料

http://blog.sina.com.cn/s/blog_1611442d10102wt31.html

https://zhuanlan.zhihu.com/logit?author=glfkuan