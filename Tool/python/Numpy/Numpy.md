# Numpy

## 数组构造

【a】等差序列： `np.linspace, np.arange`

```python
In [31]: np.linspace(1,5,11) # 起始、终止（包含）、样本个数
Out[31]: array([1. , 1.4, 1.8, 2.2, 2.6, 3. , 3.4, 3.8, 4.2, 4.6, 5. ])

In [32]: np.arange(1,5,2) # 起始、终止（不包含）、步长
Out[32]: array([1, 3])
```


【b】特殊矩阵： `zeros, eye, full`

```python
In [33]: np.zeros((2,3)) # 传入元组表示各维度大小
Out[33]: 
array([[0., 0., 0.],
       [0., 0., 0.]])

In [34]: np.eye(3) # 3*3的单位矩阵
Out[34]: 
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])

In [35]: np.eye(3, k=1) # 偏移主对角线1个单位的伪单位矩阵
Out[35]: 
array([[0., 1., 0.],
       [0., 0., 1.],
       [0., 0., 0.]])

In [36]: np.full((2,3), 10) # 元组传入大小，10表示填充数值
Out[36]: 
array([[10, 10, 10],
       [10, 10, 10]])

In [37]: np.full((2,3), [1,2,3]) # 每行填入相同的列表
Out[37]: 
array([[1, 2, 3],
       [1, 2, 3]])
```


【c】随机矩阵： `np.random<br />`
最常用的随机生成函数为 `rand, randn, randint, choice` ，它们分别表示0-1均匀分布的随机数组、标准正态的随机数组、随机整数组和随机列表抽样：

```python
In [38]: np.random.rand(3) # 生成服从0-1均匀分布的三个随机数
Out[38]: array([0.99390903, 0.25854328, 0.13560598])

In [39]: np.random.rand(3, 3) # 注意这里传入的不是元组，每个维度大小分开输入
Out[39]: 
array([[0.62634822, 0.06747959, 0.76049576],
       [0.21826591, 0.71708638, 0.98481069],
       [0.38071365, 0.82645691, 0.25598288]])
```


对于服从区间 a 到 b 上的均匀分布可以如下生成：

```python
In [40]: a, b = 5, 15

In [41]: (b - a) * np.random.rand(3) + a
Out[41]: array([ 6.40061821,  6.72343487, 10.49412407])

# 一般的，可以选择已有的库函数：
In [42]: np.random.uniform(5, 15, 3)
Out[42]: array([11.10830186,  7.35193797,  8.46971257])

```


`randn` 生成了 N(0,I)的标准正态分布：

```python
In [43]: np.random.randn(3)
Out[43]: array([ 1.2642241 , -1.04640246,  0.05297258])

In [44]: np.random.randn(2, 2)
Out[44]: 
array([[2.65755302, 0.12266858],
       [0.29899713, 0.40504878]])

      
# 一元正态分布
In [45]: sigma, mu = 2.5, 3

In [46]: mu + np.random.randn(3) * sigma
Out[46]: array([6.46031228, 0.57297935, 5.2692226 ]) 

# 已有函数
In [47]: np.random.normal(3, 2.5, 3)
Out[47]: array([2.72546019, 7.42390272, 3.71079215])

```


`randint` 可以指定生成随机整数的最小值最大值（不包含）和维度大小：

```python
In [48]: low, high, size = 5, 15, (2,2) # 生成5到14的随机整数

In [49]: np.random.randint(low, high, size)
Out[49]: 
array([[ 6, 10],
       [11, 11]])
```


`choice` 可以从给定的列表中，以一定概率和方式抽取结果，当不指定概率时为均匀采样，默认抽取方式为有放回抽样：

```python
In [50]: my_list = ['a', 'b', 'c', 'd']

In [51]: np.random.choice(my_list, 2, replace=False, p=[0.1, 0.7, 0.1 ,0.1])
Out[51]: array(['b', 'd'], dtype='<U1')

In [52]: np.random.choice(my_list, (3,3))
Out[52]: 
array([['a', 'b', 'b'],
       ['a', 'a', 'b'],
       ['c', 'c', 'b']], dtype='<U1')
```


当返回的元素个数与原列表相同时，不放回抽样等价于使用 `permutation` 函数，即打散原列表：

```python
In [53]: np.random.permutation(my_list)
Out[53]: array(['b', 'd', 'a', 'c'], dtype='<U1')
```


随机种子，它能够固定随机数的输出结果：

```python
In [54]: np.random.seed(0)

In [55]: np.random.rand()
Out[55]: 0.5488135039273248

In [56]: np.random.seed(0)

In [57]: np.random.rand()
Out[57]: 0.5488135039273248

```


## 数组的变形与合并

【b】合并操作： `r_, c_<br />`
对于二维数组而言， `r_` 和 `c_` 分别表示上下合并和左右合并：

```python
n [59]: np.r_[np.zeros((2,3)),np.zeros((2,3))]
Out[59]: 
array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]])

In [60]: np.c_[np.zeros((2,3)),np.zeros((2,3))]
Out[60]: 
array([[0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0.]])
```


一维数组和二维数组进行合并时，应当把其视作列向量，在长度匹配的情况下只能够使用左右合并的 `c_` 操作：

```python
In [61]: try:
   ....:     np.r_[np.array([0,0]),np.zeros((2,1))]
   ....: except Exception as e:
   ....:     Err_Msg = e
   ....: 

In [62]: Err_Msg
Out[62]: ValueError('all the input arrays must have same number of dimensions, but the array at index 0 has 1 dimension(s) and the array at index 1 has 2 dimension(s)')

In [63]: np.r_[np.array([0,0]),np.zeros(2)]
Out[63]: array([0., 0., 0., 0.])

In [64]: np.c_[np.array([0,0]),np.zeros((2,3))]
Out[64]: 
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.]])
```


【c】维度变换： `reshape<br />`
`reshape` 能够帮助用户把原数组按照新的维度重新排列。在使用时有两种模式，分别为 `C` 模式和 `F` 模式，分别以逐行和逐列的顺序进行填充读取。

```python
In [65]: target = np.arange(8).reshape(2,4)

In [66]: target
Out[66]: 
array([[0, 1, 2, 3],
       [4, 5, 6, 7]])

In [67]: target.reshape((4,2), order='C') # 按照行读取和填充
Out[67]: 
array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7]])

In [68]: target.reshape((4,2), order='F') # 按照列读取和填充
Out[68]: 
array([[0, 2],
       [4, 6],
       [1, 3],
       [5, 7]])
```


特别地，由于被调用数组的大小是确定的， reshape 允许有一个维度存在空缺，此时只需填充-1即可：

```python
In [69]: target.reshape((4,-1))
Out[69]: 
array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7]])
```


下面将 `n*1` 大小的数组转为1维数组的操作是经常使用的：

```python
In [70]: target = np.ones((3,1))

In [71]: target
Out[71]: 
array([[1.],
       [1.],
       [1.]])

In [72]: target.reshape(-1)
Out[72]: array([1., 1., 1.])

```


## 数组的切片与索引

数组的切片模式支持使用 `slice` 类型的 `start:end:step` 切片，还可以直接传入列表指定某个维度的索引进行切片：

```python
In [73]: target = np.arange(9).reshape(3,3)

In [74]: target
Out[74]: 
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])

In [75]: target[:-1, [0,2]]
Out[75]: 
array([[0, 2],
       [3, 5]])
```


此外，还可以利用 `np.ix_` 在对应的维度上使用布尔索引，但此时不能使用 `slice` 切片：

```python
In [76]: target[np.ix_([True, False, True], [True, False, True])]
Out[76]: 
array([[0, 2],
       [6, 8]])

In [77]: target[np.ix_([1,2], [True, False, True])]
Out[77]: 
array([[3, 5],
       [6, 8]])
```


当数组维度为1维时，可以直接进行布尔索引，而无需 `np.ix_` ：

```python
In [78]: new = target.reshape(-1)

In [79]: new[new%2==0]
Out[79]: array([0, 2, 4, 6, 8])

```


## 常用函数

where

```python
In [80]: a = np.array([-1,1,-1,0])

In [81]: np.where(a>0, a, 5) # 对应位置为True时填充a对应元素，否则填充5
Out[81]: array([5, 1, 5, 5])
```


nonzero, argmax, argmin

这三个函数返回的都是索引， `nonzero` 返回非零数的索引， `argmax, argmin` 分别返回最大和最小数的索引：

```python
In [82]: a = np.array([-2,-5,0,1,3,-1])

In [83]: np.nonzero(a)
Out[83]: (array([0, 1, 3, 4, 5], dtype=int64),)

In [84]: a.argmax()
Out[84]: 4

In [85]: a.argmin()
Out[85]: 1
```


【c】 `any, all<br />`
`any` 指当序列至少 存在一个 `True` 或非零元素时返回 `True` ，否则返回 `False<br />`
`all` 指当序列元素 全为 `True` 或非零元素时返回 `True` ，否则返回 `False`

```python
In [86]: a = np.array([0,1])

In [87]: a.any()
Out[87]: True

In [88]: a.all()
Out[88]: False
```


【d】 `cumprod, cumsum, diff<br />`
`cumprod, cumsum` 分别表示累乘和累加函数，返回同长度的数组， `diff` 表示和前一个元素做差，由于第一个元素为缺失值，因此在默认参数情况下，返回长度是原数组减1

```python
In [89]: a = np.array([1,2,3])

In [90]: a.cumprod()
Out[90]: array([1, 2, 6], dtype=int32)

In [91]: a.cumsum()
Out[91]: array([1, 3, 6], dtype=int32)

In [92]: np.diff(a)
Out[92]: array([1, 1])
```


【e】 统计函数

常用的统计函数包括 `max, min, mean, median, std, var, sum, quantile` ，其中分位数计算是全局方法，因此不能通过 `array.quantile` 的方法调用：

```python
In [93]: target = np.arange(5)

In [94]: target
Out[94]: array([0, 1, 2, 3, 4])

In [95]: target.max()
Out[95]: 4

In [96]: np.quantile(target, 0.5) # 0.5分位数
Out[96]: 2.0
```


但是对于含有缺失值的数组，它们返回的结果也是缺失值，如果需要略过缺失值，必须使用 `nan*` 类型的函数，上述的几个统计函数都有对应的 `nan*` 函数。

```python
In [97]: target = np.array([1, 2, np.nan])

In [98]: target
Out[98]: array([ 1.,  2., nan])

In [99]: target.max()
Out[99]: nan

In [100]: np.nanmax(target)
Out[100]: 2.0

In [101]: np.nanquantile(target, 0.5)
Out[101]: 1.5
```


对于协方差和相关系数分别可以利用 cov, corrcoef 如下计算：

```python
In [102]: target1 = np.array([1,3,5,9])

In [103]: target2 = np.array([1,5,3,-9])

In [104]: np.cov(target1, target2)
Out[104]: 
array([[ 11.66666667, -16.66666667],
       [-16.66666667,  38.66666667]])

In [105]: np.corrcoef(target1, target2)
Out[105]: 
array([[ 1.        , -0.78470603],
       [-0.78470603,  1.        ]])
```


最后，需要说明二维 `Numpy` 数组中统计函数的 `axis` 参数，它能够进行某一个维度下的统计特征计算，当 `axis=0` 时结果为列的统计指标，当 `axis=1` 时结果为行的统计指标：

```python
In [106]: target = np.arange(1,10).reshape(3,-1)

In [107]: target
Out[107]: 
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])

In [108]: target.sum(0)
Out[108]: array([12, 15, 18])

In [109]: target.sum(1)
Out[109]: array([ 6, 15, 24])

```


## 广播机制

【a】标量和数组的操作

当一个标量和数组进行运算时，标量会自动把大小扩充为数组大小，之后进行逐元素操作

【b】二维数组之间的操作

当两个数组维度完全一致时，使用对应元素的操作，否则会报错，除非其中的某个数组的维度是 m×1 或者 1×n ，那么会扩充其具有 1的维度为另一个数组对应维度的大小。例如， 1×2 数组和 3×2 数组做逐元素运算时会把第一个数组扩充为 3×2 ，扩充时的对应数值进行赋值。但是，需要注意的是，如果第一个数组的维度是 1×3 ，那么由于在第二维上的大小不匹配且不为 1 ，此时报错。

```python
In [114]: res = np.ones((3,2))

In [115]: res
Out[115]: 
array([[1., 1.],
       [1., 1.],
       [1., 1.]])

In [116]: res * np.array([[2,3]]) # 第二个数组扩充第一维度为3
Out[116]: 
array([[2., 3.],
       [2., 3.],
       [2., 3.]])

In [117]: res * np.array([[2],[3],[4]]) # 第二个数组扩充第二维度为2
Out[117]: 
array([[2., 2.],
       [3., 3.],
       [4., 4.]])

In [118]: res * np.array([[2]]) # 等价于两次扩充，第二个数组两个维度分别扩充为3和2
Out[118]: 
array([[2., 2.],
       [2., 2.],
       [2., 2.]])
```


【c】一维数组与二维数组的操作

当一维数组 Ak与二维数组 Bm,n 操作时，等价于把一维数组视作 A1,k 的二维数组，使用的广播法则与【b】中一致，当 k!=n 且 k,n 都不是 1时报错。

```python
In [119]: np.ones(3) + np.ones((2,3))
Out[119]: 
array([[2., 2., 2.],
       [2., 2., 2.]])

In [120]: np.ones(3) + np.ones((2,1))
Out[120]: 
array([[2., 2., 2.],
       [2., 2., 2.]])

In [121]: np.ones(1) + np.ones((2,3))
Out[121]: 
array([[2., 2., 2.],
       [2., 2., 2.]])

```


## 向量与矩阵的计算

【a】向量内积： `dot`

```python
In [122]: a = np.array([1,2,3])

In [123]: b = np.array([1,3,5])

In [124]: a.dot(b)
Out[124]: 22
```


【b】向量范数和矩阵范数： `np.linalg.norm<br />`
在矩阵范数的计算中，最重要的是 `ord` 参数

|ord|norm for matrices|norm for vectors|
|---|---|---|
|None|Frobenius norm|2-norm|
|‘fro’|Frobenius norm|–|
|‘nuc’|nuclear norm|–|
|inf|max(sum(abs(x), axis=1))|max(abs(x))|
|-inf|min(sum(abs(x), axis=1))|min(abs(x))|
|0|–|sum(x != 0)|
|1|max(sum(abs(x), axis=0))|as below|
|-1|min(sum(abs(x), axis=0))|as below|
|2|2-norm (largest sing. value)|as below|
|-2|smallest singular value|as below|
|other|–|sum(abs(x)**ord)** (1./ord)|



```python
In [125]: matrix_target =  np.arange(4).reshape(-1,2)

In [126]: matrix_target
Out[126]: 
array([[0, 1],
       [2, 3]])

In [127]: np.linalg.norm(matrix_target, 'fro')
Out[127]: 3.7416573867739413

In [128]: np.linalg.norm(matrix_target, np.inf)
Out[128]: 5.0

In [129]: np.linalg.norm(matrix_target, 2)
Out[129]: 3.702459173643833
```


```python
In [130]: vector_target =  np.arange(4)

In [131]: vector_target
Out[131]: array([0, 1, 2, 3])

In [132]: np.linalg.norm(vector_target, np.inf)
Out[132]: 3.0

In [133]: np.linalg.norm(vector_target, 2)
Out[133]: 3.7416573867739413

In [134]: np.linalg.norm(vector_target, 3)
Out[134]: 3.3019272488946263
```


【c】矩阵乘法： @

```python
In [135]: a = np.arange(4).reshape(-1,2)

In [136]: a
Out[136]: 
array([[0, 1],
       [2, 3]])

In [137]: b = np.arange(-4,0).reshape(-1,2)

In [138]: b
Out[138]: 
array([[-4, -3],
       [-2, -1]])

In [139]: a@b
Out[139]: 
array([[ -2,  -1],
       [-14,  -9]])
```



## Numpy中的ascontiguousarray

在使用Numpy的时候，有时候会遇到下面的错误：

`AttributeError: incompatible shape for a non-contiguous array`

**C order vs Fortran order** 

所谓C order，指的是行优先的顺序（Row-major Order)，即内存中同行的元素存在一起，而Fortran Order则指的是列优先的顺序（Column-major Order)，即内存中同列的元素存在一起。Pascal, C，C++，Python都是行优先存储的，而Fortran，MatLab是列优先存储的。

所谓**contiguous array** ，指的是数组在内存中存放的地址也是连续的（注意内存地址实际是一维的），即访问数组中的下一个元素，直接移动到内存中的下一个地址就可以。

由于arr是C连续的，因此对其进行行操作比进行列操作速度要快

Numpy中，随机初始化的数组默认都是C连续的，经过不规则的`slice`

操作，则会改变连续性，可能会变成既不是C连续，也不是Fortran连续的。

Numpy可以通过`.flags`属性查看一个数组是C连续还是Fortran连续的。

可以这样认为，ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。

## 参考资料

[从Numpy中的ascontiguousarray说起](https://zhuanlan.zhihu.com/p/59767914)

[http://joyfulpandas.datawhale.club/Content/ch1.html](http://joyfulpandas.datawhale.club/Content/ch1.html)

n [59]: np.r_[np.zeros((2,3)),np.zeros((2,3))]

Out[59]: 

array([[0., 0., 0.],

&ensp;&ensp;&ensp;&ensp;[0., 0., 0.],

&ensp;&ensp;&ensp;&ensp;[0., 0., 0.],

&ensp;&ensp;&ensp;&ensp;[0., 0., 0.]])

In [60]: np.c_[np.zeros((2,3)),np.zeros((2,3))]

Out[60]: 

array([[0., 0., 0., 0., 0., 0.],

&ensp;&ensp;&ensp;&ensp;[0., 0., 0., 0., 0., 0.]])

